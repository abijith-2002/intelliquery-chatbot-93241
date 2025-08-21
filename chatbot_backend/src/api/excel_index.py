"""
Excel column/metadata indexing utilities.

Maintains a per-session, per-file, per-sheet mapping of columns, dtypes, samples,
and (optionally) column-level embeddings to enable future semantic mapping between
LLM queries and DataFrame columns.

Design:
- EXCEL_STORE: high-level sheet metadata persisted in memory (and parquet paths).
- EXCEL_COLUMN_INDEX: fast lookup for columns with optional embeddings.
- Functions here update the index during Excel uploads and provide retrieval helpers.

Note:
This is an in-memory index intended for prototype/demo use. For production, consider
persisting to a durable store keyed by session_id, with eviction policies.
"""
from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
import google.generativeai as genai

from .config_utils import get_gemini_api_key


# Global in-memory column index
# Structure:
# EXCEL_COLUMN_INDEX[session_id] = {
#   "files": {
#       filename: {
#           sheet_name: {
#               "columns": [col1, col2, ...],
#               "dtypes": {col: dtype_str, ...},
#               "sample_rows": [ {col: val, ...}, ... ],  # small sample
#               "column_samples": {col: [v1, v2, ...]},  # up to N distinct/non-null values
#               "column_embeddings": {col: [float, ...]} or {},  # optional embeddings for column semantics
#           },
#           ...
#       },
#       ...
#   }
# }
EXCEL_COLUMN_INDEX: Dict[str, Dict[str, Any]] = {}


def _get_embedding_model_name() -> str:
    # Recommended Gemini embedding model
    return "models/text-embedding-004"


def _maybe_embed_text(text: str) -> Optional[List[float]]:
    """
    Optionally embed text using Gemini. Returns None if API key missing or on failure.
    """
    api_key = get_gemini_api_key()
    if not api_key:
        return None
    try:
        genai.configure(api_key=api_key)
        res = genai.embed_content(model=_get_embedding_model_name(), content=text)
        vec = res.get("embedding", {}).get("values")
        if isinstance(vec, list) and vec and isinstance(vec[0], (int, float)):
            return [float(v) for v in vec]
    except Exception:
        return None
    return None


def _compose_column_semantic_seed(
    col_name: str,
    dtype: Optional[str],
    sample_values: Optional[List[Any]],
    sample_rows: Optional[List[Dict[str, Any]]],
) -> str:
    """
    Build a concise descriptor string for a column that can be embedded.
    """
    dtype_part = f" (dtype: {dtype})" if dtype else ""
    vals_part = ""
    if sample_values:
        # Use up to 3 representative values
        preview_vals = [str(v) for v in sample_values[:3] if v is not None]
        if preview_vals:
            vals_part = f" e.g. {preview_vals}"
    # Optionally include a little context from sample rows
    row_hint = ""
    if sample_rows:
        try:
            # Grab first row and show this column's value if present
            first_row = sample_rows[0]
            if col_name in first_row and first_row[col_name] is not None:
                row_hint = f" first_row_value={first_row[col_name]}"
        except Exception:
            pass
    return f"Column: {col_name}{dtype_part}{vals_part}{row_hint}"


# PUBLIC_INTERFACE
def upsert_excel_column_index(
    session_id: str,
    filename: str,
    sheet_name: str,
    columns: List[str],
    dtypes: Dict[str, str],
    sample_rows: List[Dict[str, Any]],
    column_samples: Optional[Dict[str, List[Any]]] = None,
    build_embeddings: bool = False,
) -> None:
    """
    PUBLIC_INTERFACE
    Create or update the Excel column index for a given session/file/sheet.

    Args:
        session_id: Session identifier.
        filename: Original uploaded Excel filename.
        sheet_name: Sheet within the Excel file.
        columns: List of column names for the sheet.
        dtypes: Mapping of column to dtype string.
        sample_rows: Small sample of rows (e.g., df.head(5).to_dict(orient='records')).
        column_samples: Optional mapping column -> small list of representative values.
        build_embeddings: If True, attempt to build column-level embeddings using Gemini.
                          If no API key configured, skips silently.

    Returns:
        None
    """
    if not session_id or not filename or not sheet_name:
        return

    if session_id not in EXCEL_COLUMN_INDEX:
        EXCEL_COLUMN_INDEX[session_id] = {"files": {}}
    files_map = EXCEL_COLUMN_INDEX[session_id]["files"]
    if filename not in files_map:
        files_map[filename] = {}
    sheet_map = files_map[filename].get(sheet_name, {})

    # Prepare structure
    sheet_map["columns"] = columns[:]
    sheet_map["dtypes"] = dict(dtypes or {})
    sheet_map["sample_rows"] = list(sample_rows or [])
    sheet_map["column_samples"] = dict(column_samples or {})
    sheet_map["column_embeddings"] = sheet_map.get("column_embeddings", {})

    if build_embeddings:
        for col in columns:
            try:
                seed_text = _compose_column_semantic_seed(
                    col_name=col,
                    dtype=dtypes.get(col) if dtypes else None,
                    sample_values=(column_samples or {}).get(col),
                    sample_rows=sample_rows,
                )
                vec = _maybe_embed_text(seed_text)
                if vec:
                    sheet_map["column_embeddings"][col] = vec
            except Exception:
                # Ignore per-column embedding errors
                pass

    files_map[filename][sheet_name] = sheet_map


# PUBLIC_INTERFACE
def find_columns_by_name(
    session_id: str, name_query: str
) -> List[Tuple[str, str, str]]:
    """
    PUBLIC_INTERFACE
    Find columns in the session by case-insensitive name containment.

    Args:
        session_id: Session identifier.
        name_query: Substring query for column name.

    Returns:
        List of tuples: (filename, sheet_name, column_name)
    """
    results: List[Tuple[str, str, str]] = []
    if not name_query:
        return results
    session_idx = EXCEL_COLUMN_INDEX.get(session_id, {})
    files_map = session_idx.get("files", {})
    q = (name_query or "").lower()
    for fname, sheets in files_map.items():
        for sname, meta in sheets.items():
            for col in meta.get("columns", []):
                if q in str(col).lower():
                    results.append((fname, sname, col))
    return results


# PUBLIC_INTERFACE
def get_column_embedding(
    session_id: str, filename: str, sheet_name: str, column_name: str
) -> Optional[List[float]]:
    """
    PUBLIC_INTERFACE
    Retrieve the optional embedding vector for a specific column.

    Returns:
        List[float] if available, otherwise None.
    """
    session_idx = EXCEL_COLUMN_INDEX.get(session_id, {})
    files_map = session_idx.get("files", {})
    meta = files_map.get(filename, {}).get(sheet_name, {})
    col_embs = meta.get("column_embeddings") or {}
    return col_embs.get(column_name)
