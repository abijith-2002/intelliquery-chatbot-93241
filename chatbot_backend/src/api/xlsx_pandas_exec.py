import io
import re
import ast
import textwrap
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
from openpyxl import load_workbook

from .config_utils import get_gemini_api_key
import google.generativeai as genai
from rapidfuzz import fuzz


SAFE_BUILTINS = {
    "range": range,
    "len": len,
    "min": min,
    "max": max,
    "sum": sum,
    "sorted": sorted,
    "abs": abs,
    "round": round,
    "enumerate": enumerate,
    "zip": zip,
    "list": list,
    "dict": dict,
    "set": set,
    "tuple": tuple,
    "any": any,
    "all": all,
}

# PUBLIC_INTERFACE
def extract_xlsx_schema_from_bytes(content: bytes) -> Dict[str, Any]:
    """
    PUBLIC_INTERFACE
    Parse an XLSX file and extract a simple schema describing sheets and headers.

    Args:
        content (bytes): The raw XLSX content.

    Returns:
        dict: { "sheets": [ { "name": str, "columns": [str, ...], "rows": int } , ... ],
                "all_columns": [str, ...] (unique across sheets) }
    """
    wb = load_workbook(io.BytesIO(content), data_only=True, read_only=True)
    sheets_info: List[Dict[str, Any]] = []
    all_cols_ordered: List[str] = []
    seen_cols = set()

    for ws in wb.worksheets:
        # find first non-empty row for headers
        header_row = None
        total_rows = 0
        columns: List[str] = []
        for i, row in enumerate(ws.iter_rows(values_only=True), start=1):
            if row and any((str(c).strip() if c is not None else "") for c in row):
                if header_row is None:
                    header_row = row
                    # capture columns
                    for c in header_row:
                        name = (str(c).strip() if c is not None else "")
                        if name:
                            columns.append(name)
                            if name not in seen_cols:
                                seen_cols.add(name)
                                all_cols_ordered.append(name)
                else:
                    total_rows += 1
        # If no header found, default empty
        sheets_info.append({"name": ws.title, "columns": columns, "rows": total_rows})

    return {"sheets": sheets_info, "all_columns": all_cols_ordered}


def _normalize_text(s: str) -> str:
    """Lowercase and collapse whitespace for matching."""
    return " ".join((s or "").strip().lower().split())

def _tokenize(s: str) -> List[str]:
    """Simple alnum tokenization."""
    return re.findall(r"[A-Za-z0-9]+", (s or "").lower())

def _lexical_overlap_score(query: str, candidate: str) -> float:
    """Token overlap ratio as simple lexical score."""
    qset = set(_tokenize(query))
    cset = set(_tokenize(candidate))
    if not qset or not cset:
        return 0.0
    shared = qset.intersection(cset)
    return len(shared) / max(len(qset), len(cset))

def _fuzzy_ratio_score(query: str, candidate: str) -> float:
    """RapidFuzz partial ratio normalized to 0..1."""
    if not query or not candidate:
        return 0.0
    return fuzz.partial_ratio(_normalize_text(query), _normalize_text(candidate)) / 100.0

def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Cosine similarity between vectors, zero-safe."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    import math
    dot = sum(x * y for x, y in zip(vec_a, vec_b))
    a_norm = math.sqrt(sum(x * x for x in vec_a))
    b_norm = math.sqrt(sum(y * y for y in vec_b))
    if a_norm == 0 or b_norm == 0:
        return 0.0
    return dot / (a_norm * b_norm)

def _embed_one(text: str) -> Optional[List[float]]:
    """
    Local wrapper to embed a single text using Gemini; returns None if unavailable.
    We re-define here to avoid importing from main to keep module isolation.
    """
    key = get_gemini_api_key()
    if not key:
        return None
    try:
        genai.configure(api_key=key)
        model_name = "models/text-embedding-004"
        result = genai.embed_content(model=model_name, content=text)
        vec = result.get("embedding", {}).get("values")
        if isinstance(vec, list) and vec and isinstance(vec[0], (int, float)):
            return [float(v) for v in vec]
        return None
    except Exception:
        return None

def _semantic_similarity_score(query: str, candidate: str) -> float:
    """Cosine similarity of embeddings if available; else 0.0."""
    qv = _embed_one(query)
    cv = _embed_one(candidate)
    if qv and cv:
        return _cosine_similarity(qv, cv)
    return 0.0

def _rank_columns_by_combined_scores(query: str, columns: List[str], k: int = 5) -> List[str]:
    """
    Combine exact match, fuzzy match, and semantic similarity to rank columns.

    Scoring approach:
    - exact_boost: 1.0 if normalized exact string match, else 0
    - fuzzy_score: RapidFuzz partial ratio in 0..1
    - lexical_score: token overlap 0..1 (fallback signal)
    - semantic_score: cosine similarity on embeddings 0..1 if embeddings available; else 0

    Final score = exact_boost*2.0 + 0.6*fuzzy_score + 0.3*semantic_score + 0.2*lexical_score

    We also give an additional small boost if candidate is a substring of query or vice versa.
    """
    if not columns:
        return []
    nq = _normalize_text(query)

    # Try to compute embedding for query once to avoid repeated calls
    # but keep _semantic_similarity_score for candidates (most lightweight approach here)
    q_embed = _embed_one(query)

    scored: List[Tuple[float, str]] = []
    for c in columns:
        if not c:
            continue
        nc = _normalize_text(c)
        exact_boost = 1.0 if nq == nc else 0.0
        fuzzy_score = _fuzzy_ratio_score(query, c)
        lexical_score = _lexical_overlap_score(query, c)

        # semantic: prefer using precomputed query embedding if available
        semantic_score = 0.0
        if q_embed:
            c_embed = _embed_one(c)
            if c_embed:
                semantic_score = _cosine_similarity(q_embed, c_embed)
        # small containment boost
        containment_boost = 0.1 if (nc in nq or nq in nc) and not exact_boost else 0.0

        combined = exact_boost * 2.0 + 0.6 * fuzzy_score + 0.3 * semantic_score + 0.2 * lexical_score + containment_boost
        scored.append((combined, c))

    # Sort by score desc and return top k unique
    scored.sort(key=lambda x: x[0], reverse=True)
    seen = set()
    out: List[str] = []
    for _, name in scored:
        if name not in seen:
            out.append(name)
            seen.add(name)
        if len(out) >= k:
            break
    return out

def _pick_top_k_columns(query: str, columns: List[str], k: int = 5) -> List[str]:
    """
    Backward-compatible wrapper that now uses combined ranking (exact + fuzzy + semantic).
    """
    return _rank_columns_by_combined_scores(query, columns, k=k)


def _build_gemini_pandas_prompt(user_query: str, top_columns: List[str], schema: Dict[str, Any]) -> str:
    """
    Construct an instruction prompt for Gemini to emit SAFE pandas code that:
      - Reads the already loaded DataFrames from 'dfs' (dict of sheet_name -> DataFrame)
      - Is read-only and DOES NOT write files, touch network, or import modules
      - Returns a small result (<= 20 rows) into a variable named RESULT
      - Avoids mutations on 'dfs' objects; use copies
    """
    schema_lines = []
    for s in schema.get("sheets", []):
        cols_preview = ", ".join(s.get("columns", [])[:20])
        schema_lines.append(f"- Sheet '{s.get('name', '')}': {s.get('rows', 0)} data rows; columns: {cols_preview}")
    schema_text = "\n".join(schema_lines)
    top_cols_text = ", ".join(top_columns) if top_columns else "(none)"

    instruction = f"""
You are to produce ONLY a Python code block (no explanations) that uses pandas to answer the user query.

Rules:
- DataFrames are already provided in a dict named dfs where keys are sheet names and values are pandas DataFrames.
- NEVER import any library or modules.
- NEVER write to disk, modify environment, open files, or access network.
- Read-only analysis only. Do not modify dfs; create copies if needed.
- The code MUST assign the final output (DataFrame or short summary string) to a variable named RESULT.
- If returning a DataFrame, ensure it is at most 20 rows (head(20) or equivalent).
- Prefer using these relevant columns if applicable: {top_cols_text}.
- If multiple sheets are relevant, you may merge or concatenate by common columns, but keep under 20 rows result.
- The user question is:

{user_query}

XLSX schema:
{schema_text}

Return ONLY the code, without backticks.
"""
    return textwrap.dedent(instruction).strip()


# PUBLIC_INTERFACE
def generate_pandas_code_with_gemini(user_query: str, top_columns: List[str], schema: Dict[str, Any]) -> str:
    """
    PUBLIC_INTERFACE
    Ask Gemini to produce safe pandas code based on the query, relevant columns, and schema.
    Ensures generation uses minimal, read-only operations and assigns result to RESULT.

    Returns:
        str: The generated Python code as text (no backticks).
    """
    key = get_gemini_api_key()
    if not key:
        raise RuntimeError("Gemini API key not configured")
    prompt = _build_gemini_pandas_prompt(user_query, top_columns, schema)
    genai.configure(api_key=key)
    model = genai.GenerativeModel("gemini-2.5-flash")
    resp = model.generate_content([{"role": "user", "parts": [prompt]}])
    code = (resp.text or "").strip()
    # strip code fences if any
    code = re.sub(r"^```(?:python)?\s*", "", code, flags=re.IGNORECASE)
    code = re.sub(r"\s*```$", "", code)
    return code.strip()


def _ast_check_safe(code: str) -> None:
    """
    Static checks: prevent dangerous syntax like import, attribute dunder access, exec, eval, __builtins__ access, etc.
    Raises ValueError if unsafe construct found.
    """
    forbidden_calls = {"exec", "eval", "__import__", "open", "compile", "input"}
    forbidden_names = {"__builtins__", "__loader__", "__package__", "__spec__", "__file__", "__name__"}
    tree = ast.parse(code, mode="exec")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise ValueError("Import statements are not allowed")
        if isinstance(node, ast.Attribute):
            # ban any attribute name with double underscores to reduce bypass risk
            if isinstance(node.attr, str) and "__" in node.attr:
                raise ValueError("Dunder attribute access is not allowed")
        if isinstance(node, ast.Call):
            # function name check
            fn = node.func
            if isinstance(fn, ast.Name) and fn.id in forbidden_calls:
                raise ValueError(f"Forbidden call: {fn.id}")
            if isinstance(fn, ast.Attribute) and isinstance(fn.attr, str):
                if fn.attr in forbidden_calls:
                    raise ValueError(f"Forbidden call: {fn.attr}")
        if isinstance(node, ast.Name):
            if node.id in forbidden_names:
                raise ValueError(f"Forbidden name usage: {node.id}")


def _render_df_result(df: pd.DataFrame) -> str:
    """
    Render a small DataFrame (<= 20 rows) as a simple text table for chat display.
    """
    # Ensure capped rows and columns sanity
    df = df.head(20)
    # convert to pretty text table using pandas built-in to_string
    return df.to_string(index=False)


def _summarize_non_df_result(obj: Any) -> str:
    """
    For non-DataFrame results, coerce to short string.
    """
    if isinstance(obj, (str, int, float, bool)):
        return str(obj)
    if isinstance(obj, list):
        if len(obj) > 20:
            return str(obj[:20]) + " ... (truncated)"
        return str(obj)
    if isinstance(obj, dict):
        # show first 10 keys
        keys = list(obj.keys())[:10]
        preview = {k: obj[k] for k in keys}
        return str(preview)
    return str(obj)


# PUBLIC_INTERFACE
def execute_safe_pandas_code_on_xlsx(code: str, xlsx_bytes: bytes) -> str:
    """
    PUBLIC_INTERFACE
    Execute previously validated pandas code in a restricted environment against the uploaded XLSX.

    The runtime provides:
        - dfs: Dict[str, pandas.DataFrame] keyed by sheet name
        - pd: pandas module
        - RESULT: expected output variable filled by the code

    Safety:
        - No builtins except a small safe subset
        - No imports (checked via AST)
        - No file/network operations
        - Output limited to <= 20 rows for DataFrames

    Returns:
        str: Rendered table or summary as a string.
    """
    # Static validation
    _ast_check_safe(code)

    # Load all sheets into DataFrames
    all_sheets = pd.read_excel(io.BytesIO(xlsx_bytes), sheet_name=None, engine="openpyxl")
    # ensure basic dtype handling; leave as-is to let code filter/aggregate

    # Prepare restricted globals/locals
    safe_globals = {
        "__builtins__": SAFE_BUILTINS,
        "pd": pd,
        "dfs": all_sheets,
    }
    safe_locals: Dict[str, Any] = {}

    # Execute user code
    exec(code, safe_globals, safe_locals)  # noqa: S102 (controlled environment)

    # Retrieve RESULT from locals or globals
    result = safe_locals.get("RESULT", safe_globals.get("RESULT"))
    if result is None:
        # Try common accidental variable name like 'result'
        result = safe_locals.get("result", safe_globals.get("result"))

    if result is None:
        raise ValueError("Generated code did not assign RESULT")

    # Render
    if isinstance(result, pd.DataFrame):
        return _render_df_result(result)
    return _summarize_non_df_result(result)


# PUBLIC_INTERFACE
def get_top_k_relevant_columns_for_query(user_query: str, header_store: Dict[str, Any], k: int = 5) -> List[str]:
    """
    PUBLIC_INTERFACE
    Determine the top-k relevant columns from the stored header information by combining:
      1) Exact match (strong boost if query equals the column name post-normalization)
      2) Fuzzy match (RapidFuzz partial ratio)
      3) Semantic similarity (cosine similarity on Gemini embeddings)
    with lexical token-overlap as an additional weak signal.

    If embeddings are unavailable, semantic score gracefully degrades to 0 and the
    ranking relies on exact + fuzzy + lexical.

    Args:
        user_query (str): The user question or instruction.
        header_store (dict): Store from get_session_header_embeddings(session_id), containing:
            {
                "headers": List[str],
                "embeddings": Optional[List[List[float] or None]],  # kept for future extension
                "embedding_model": str
            }
        k (int): Number of top columns to return.

    Returns:
        List[str]: Up to k column names ranked by combined relevance.
    """
    headers = (header_store or {}).get("headers", []) or []
    return _pick_top_k_columns(user_query, headers, k=k)
