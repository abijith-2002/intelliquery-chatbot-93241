import io
import re
import ast
import textwrap
from typing import Dict, Any, List

import pandas as pd
from openpyxl import load_workbook

from .config_utils import get_gemini_api_key
import google.generativeai as genai


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


def _pick_top_k_columns(query: str, columns: List[str], k: int = 5) -> List[str]:
    """
    Pick top-k columns using simple lexical overlap as fallback relevance.
    """
    if not columns:
        return []
    q = (query or "").lower()
    scored = []
    for c in columns:
        name = c or ""
        tokens = re.findall(r"[A-Za-z0-9]+", name.lower())
        score = sum(1 for t in tokens if t and t in q)
        scored.append((score, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = [c for _, c in scored[:k]]
    # ensure uniqueness and with non-empty
    seen = set()
    out = []
    for c in top:
        if c and c not in seen:
            out.append(c)
            seen.add(c)
    return out


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
    Determine the top-k relevant columns from the stored header embedding store using lexical fallback.

    Args:
        user_query (str): The question.
        header_store (dict): From session header store (get_session_header_embeddings).
        k (int): Number to select.

    Returns:
        List[str]: up to k column names
    """
    headers = (header_store or {}).get("headers", []) or []
    return _pick_top_k_columns(user_query, headers, k=k)
