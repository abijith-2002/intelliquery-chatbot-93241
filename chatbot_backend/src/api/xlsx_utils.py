from __future__ import annotations

import io
import re
from typing import Any, Dict, List, Tuple

import pandas as pd
from pydantic import BaseModel, Field

# Fuzzy matching
from rapidfuzz import fuzz

# Simple semantic similarity via TF-IDF cosine
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In-memory XLSX session store
# Structure:
#   XLSX_SESSIONS[session_id] = {
#       "files": {
#           filename: {
#               "sheets": {
#                   sheet_name: {
#                       "df": pandas.DataFrame,
#                       "columns": List[str]
#                   }
#               }
#           }
#       },
#       "default": (filename, sheet_name)  # last uploaded or explicitly chosen
#   }
XLSX_SESSIONS: Dict[str, Dict[str, Any]] = {}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9_]", "", re.sub(r"\s+", "_", (s or "").strip().lower()))


def _candidate_columns(df: pd.DataFrame) -> List[str]:
    return [str(c) for c in df.columns]


def _tfidf_semantic_score(query: str, candidates: List[str]) -> List[Tuple[str, float]]:
    if not query or not candidates:
        return []
    texts = [query] + candidates
    vec = TfidfVectorizer().fit_transform(texts)
    qv = vec[0:1]
    cv = vec[1:]
    sims = cosine_similarity(qv, cv).flatten().tolist()
    return list(zip(candidates, sims))


def _fuzzy_scores(query: str, candidates: List[str]) -> List[Tuple[str, float]]:
    out: List[Tuple[str, float]] = []
    for c in candidates:
        score = fuzz.token_set_ratio(query, c) / 100.0
        out.append((c, score))
    return out


def _blend_scores(query: str, candidates: List[str]) -> List[Tuple[str, float]]:
    # Blend fuzzy and semantic equally
    fuzzy = dict(_fuzzy_scores(query, candidates))
    sem = dict(_tfidf_semantic_score(query, candidates))
    blended: List[Tuple[str, float]] = []
    for c in candidates:
        blended.append((c, 0.5 * fuzzy.get(c, 0.0) + 0.5 * sem.get(c, 0.0)))
    blended.sort(key=lambda x: x[1], reverse=True)
    return blended


def _extract_requested_ops(query: str) -> Dict[str, Any]:
    """
    Parse common intents like filters and aggregations from the query.
    Recognized:
      - simple filters: col op value (>, <, >=, <=, =, ==)
      - contains filter: col: value
      - aggregations: sum/avg/mean/count/max/min of column
      - selections: top N, first N, last N
      - sorting: sort by/ order by col asc|desc
    """
    ops: Dict[str, Any] = {
        "filters": [],
        "contains": [],
        "aggs": [],
        "select": [],
        "sort": None,
        "limit": None,
    }
    # contains style: "status: open"
    for m in re.finditer(r"([A-Za-z0-9_ ]+)\s*:\s*([^\n,;]+)", query):
        ops["contains"].append({"field": m.group(1).strip(), "value": m.group(2).strip()})

    # comparison filters: col >= 10.5 etc.
    for m in re.finditer(r"([A-Za-z0-9_ ]+)\s*(>=|<=|>|<|==|=)\s*([0-9]+(?:\.[0-9]+)?)", query):
        ops["filters"].append({"field": m.group(1).strip(), "op": m.group(2), "value": float(m.group(3))})

    # aggregations
    for agg in ["sum", "avg", "average", "mean", "count", "max", "min"]:
        for m in re.finditer(rf"{agg}\s+of\s+([A-Za-z0-9_ ]+)", query, flags=re.IGNORECASE):
            ops["aggs"].append({"func": agg.lower(), "field": m.group(1).strip()})

    # top/first/last N
    for m in re.finditer(r"(top|first|last)\s+([0-9]+)", query, flags=re.IGNORECASE):
        ops["select"].append({"type": m.group(1).lower(), "n": int(m.group(2))})

    # sort by/order by
    m = re.search(r"(?:sort|order)\s+by\s+([A-Za-z0-9_ ]+)(?:\s+(asc|desc))?", query, flags=re.IGNORECASE)
    if m:
        ops["sort"] = {"field": m.group(1).strip(), "direction": (m.group(2) or "asc").lower()}

    # explicit limit
    m2 = re.search(r"limit\s+([0-9]+)", query, flags=re.IGNORECASE)
    if m2:
        ops["limit"] = int(m2.group(1))

    return ops


# PUBLIC_INTERFACE
class XLSXIngestResult(BaseModel):
    """Result of ingesting a single XLSX file."""
    filename: str = Field(..., description="Uploaded filename")
    sheets: List[str] = Field(default_factory=list, description="Sheet names ingested")
    rows_total: int = Field(0, description="Total rows across sheets (first 50k rows per sheet cap)")
    columns_per_sheet: Dict[str, List[str]] = Field(default_factory=dict, description="Column names per sheet")


# PUBLIC_INTERFACE
def ingest_xlsx_for_session(session_id: str, filename: str, content: bytes) -> XLSXIngestResult:
    """
    PUBLIC_INTERFACE
    Load XLSX into memory for the session. Caps rows to 50k per sheet to avoid memory pressure.

    Robustness:
    - Parses with engine='openpyxl'
    - Reads all columns as strings (dtype=str) to avoid dtype inference issues
    - Skips sheets that fail to parse instead of failing the entire file
    """
    if not content or len(content) == 0:
        raise ValueError(f"Empty file received for '{filename}'.")
    bio = io.BytesIO(content)

    try:
        xl = pd.ExcelFile(bio, engine="openpyxl")
    except Exception as e:
        raise ValueError(f"Failed to open '{filename}' as XLSX: {e}")

    sess = XLSX_SESSIONS.setdefault(session_id, {"files": {}, "default": None})
    files = sess["files"].setdefault(filename, {"sheets": {}})

    rows_total = 0
    columns_per_sheet: Dict[str, List[str]] = {}
    parsed_sheets: List[str] = []

    for sheet in xl.sheet_names:
        try:
            # Read everything as text to ensure safe downstream processing/embeddings
            df = xl.parse(sheet, dtype=str)
        except Exception as e:
            # Skip problematic sheet but continue others
            continue
        if len(df) > 50000:
            df = df.iloc[:50000].copy()
        # Ensure column names are strings
        df.columns = [str(c) for c in df.columns]
        files["sheets"][sheet] = {"df": df, "columns": [str(c) for c in df.columns]}
        rows_total += len(df)
        columns_per_sheet[sheet] = [str(c) for c in df.columns]
        parsed_sheets.append(sheet)

    # set default to the first successfully parsed sheet of last uploaded file
    default_sheet = parsed_sheets[0] if parsed_sheets else (xl.sheet_names[0] if xl.sheet_names else None)
    sess["default"] = (filename, default_sheet)

    return XLSXIngestResult(
        filename=filename,
        sheets=parsed_sheets,
        rows_total=rows_total,
        columns_per_sheet=columns_per_sheet
    )


# PUBLIC_INTERFACE
def get_default_dataframe(session_id: str) -> Tuple[str, str, pd.DataFrame]:
    """
    PUBLIC_INTERFACE
    Get default dataframe (filename, sheet_name, df) for the session.
    Raises ValueError if not available.
    """
    sess = XLSX_SESSIONS.get(session_id) or {}
    default = sess.get("default")
    if not default:
        raise ValueError("No XLSX uploaded for this session.")
    filename, sheet = default
    filemeta = sess["files"][filename]
    dfmeta = filemeta["sheets"][sheet]
    return filename, sheet, dfmeta["df"]


# PUBLIC_INTERFACE
def choose_columns_for_query(query: str, df: pd.DataFrame, max_cols: int = 5) -> List[str]:
    """
    PUBLIC_INTERFACE
    Identify relevant dataframe columns for the natural language query using:
      - Fuzzy token match (rapidfuzz)
      - TF-IDF semantic match
    Only returns existing columns; never invents new names.
    """
    candidates = _candidate_columns(df)
    if not candidates:
        return []
    blended = _blend_scores(query, candidates)
    cols = [c for c, _ in blended[:max_cols]]

    # Also ensure columns referenced explicitly in query-like tokens appear
    explicit: List[str] = []
    tokens = re.findall(r"[A-Za-z0-9_]+", query)
    for t in tokens:
        # exact match ignoring case
        for c in candidates:
            if t.lower() == str(c).lower() and c not in explicit:
                explicit.append(c)
    # Merge explicit to front preserving order
    merged = []
    for c in explicit + cols:
        if c not in merged:
            merged.append(c)
    # Keep only real columns
    merged = [c for c in merged if c in candidates]
    return merged[:max_cols]


# PUBLIC_INTERFACE
def build_pandas_code(query: str, df_name: str, df: pd.DataFrame) -> str:
    """
    PUBLIC_INTERFACE
    Generate pandas code string to retrieve data as requested (Type 1).
    This code includes only valid column names from df; never invents.
    The returned string is code only (no explanations).
    """
    ops = _extract_requested_ops(query)
    cols = choose_columns_for_query(query, df, max_cols=6)

    lines: List[str] = []
    lines.append(f"# dataframe: {df_name}")
    lines.append(f"out = {df_name}.copy()")

    # contains filters (case-insensitive)
    for cond in ops["contains"]:
        field = cond["field"]
        # map to best column
        best = choose_columns_for_query(field, df, max_cols=1)
        if not best:
            continue
        col = best[0]
        # Escape single quotes in the value for safe string embedding
        val = cond["value"].replace("'", "\\'")
        lines.append(f"out = out[out['{col}'].astype(str).str.contains('{val}', case=False, na=False)]")

    # numeric filters
    for cond in ops["filters"]:
        field = cond["field"]
        best = choose_columns_for_query(field, df, max_cols=1)
        if not best:
            continue
        col = best[0]
        op = cond["op"]
        num = cond["value"]
        lines.append(f"out = out[pd.to_numeric(out['{col}'], errors='coerce').fillna(float('nan')) {op} {num}]")

    # sorting
    if ops["sort"]:
        best = choose_columns_for_query(ops["sort"]["field"], df, max_cols=1)
        if best:
            dir_desc = ops["sort"]["direction"] == "desc"
            lines.append(f"out = out.sort_values(by='{best[0]}', ascending={str(not dir_desc)})")

    # projection
    if cols:
        safe = [c for c in cols if c in df.columns]
        lines.append(f"out = out[{safe!r}]")

    # aggregations (apply per involved column and create a summary dict)
    if ops["aggs"]:
        lines.append("agg_result = {}")
        for a in ops["aggs"]:
            func = a["func"]
            field = a["field"]
            best = choose_columns_for_query(field, df, max_cols=1)
            if not best:
                continue
            col = best[0]
            if func in ("avg", "average"):
                func = "mean"
            if func in ("sum", "mean", "count", "max", "min"):
                if func == "count":
                    lines.append(f"agg_result['{func}_{col}'] = out['{col}'].count()")
                else:
                    lines.append(f"agg_result['{func}_{col}'] = pd.to_numeric(out['{col}'], errors='coerce').{func}()")
        lines.append("# agg_result contains computed aggregates")

    # selections (top/first/last N)
    limit_applied = False
    for sel in ops["select"]:
        n = sel["n"]
        if sel["type"] in ("top", "first"):
            lines.append(f"out = out.head({n})")
            limit_applied = True
        elif sel["type"] == "last":
            lines.append(f"out = out.tail({n})")
            limit_applied = True

    # explicit limit
    if ops["limit"] and not limit_applied:
        lines.append(f"out = out.head({int(ops['limit'])})")

    lines.append("out")
    return "\n".join(lines)
