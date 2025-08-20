"""
DuckDB integration utilities for per-session, in-memory analytics on uploaded tabular data.

Design:
- Create an in-memory DuckDB connection per chat session lazily on first use.
- Register each uploaded Excel sheet's pandas DataFrame as a DuckDB view/table for that session.
- Provide helpers to execute SQL safely and return compact results for chat usage.

Persistence:
- In-memory only. No persistent DuckDB database is created or stored on disk.

Notes:
- Requires dependency: duckdb>=0.9.2 (documented in requirements.txt)
"""

from typing import Any, Dict, List, Optional, Tuple
import duckdb
import pandas as pd


# Registry of per-session in-memory DuckDB connections
# Structure: { session_id: duckdb.DuckDBPyConnection }
_DUCKDB_SESSIONS: Dict[str, duckdb.DuckDBPyConnection] = {}


def _get_or_create_session_conn(session_id: str) -> duckdb.DuckDBPyConnection:
    """
    Get or create a DuckDB in-memory connection for a session.
    """
    conn = _DUCKDB_SESSIONS.get(session_id)
    if conn is None:
        # :memory: creates new in-memory database. Using python API, just call duckdb.connect()
        conn = duckdb.connect(database=":memory:")
        # Set some pragmatic defaults
        conn.execute("PRAGMA threads=2;")
        _DUCKDB_SESSIONS[session_id] = conn
    return conn


# PUBLIC_INTERFACE
def register_pandas_tables(session_id: str, filename: str, sheet_map: Dict[str, pd.DataFrame]) -> None:
    """
    PUBLIC_INTERFACE
    Register each sheet DataFrame under the session's DuckDB connection as a table.

    Table naming:
        {safe_base_filename}__{safe_sheet_name}

    Args:
        session_id: Chat session identifier.
        filename: Original uploaded filename (used to namespace tables).
        sheet_map: Dict of sheet_name -> pandas.DataFrame
    """
    if not sheet_map:
        return
    conn = _get_or_create_session_conn(session_id)

    base = _safe_ident(filename)
    for sheet_name, df in sheet_map.items():
        if df is None:
            continue
        safe_table = f"{base}__{_safe_ident(sheet_name)}"
        # Create or replace view to point to the pandas DataFrame
        # Using register allows direct query: select * from "table"
        conn.register(safe_table, df)


def _safe_ident(name: str) -> str:
    """
    Turn a file/sheet name into a simple safe identifier by:
      - lowercasing
      - replacing non-alnum with underscore
      - collapsing multiple underscores
      - trimming leading/trailing underscores
    """
    import re
    s = (name or "").lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s)
    s = s.strip("_")
    if not s:
        s = "tbl"
    return s


# PUBLIC_INTERFACE
def list_session_tables(session_id: str) -> List[str]:
    """
    PUBLIC_INTERFACE
    Return the list of registered table names for the session.
    """
    conn = _get_or_create_session_conn(session_id)
    try:
        res = conn.execute("SHOW TABLES").fetchall()
        return [r[0] for r in res]
    except Exception:
        return []


# PUBLIC_INTERFACE
def explain_schema(session_id: str) -> Dict[str, Any]:
    """
    PUBLIC_INTERFACE
    Provide a quick schema snapshot for the session's DuckDB tables.

    Returns:
        Dict like {"tables": [{ "name": str, "columns": [{"name": str, "type": str}, ...]}, ...]}
    """
    conn = _get_or_create_session_conn(session_id)
    out: Dict[str, Any] = {"tables": []}
    try:
        tables = list_session_tables(session_id)
        for t in tables:
            cols = conn.execute(f"DESCRIBE \"{t}\"").fetchall()
            # DuckDB DESCRIBE columns: [("column_name","type",...)]
            col_items = []
            for row in cols:
                # row[0]=column_name, row[1]=type
                if len(row) >= 2:
                    col_items.append({"name": str(row[0]), "type": str(row[1])})
            out["tables"].append({"name": t, "columns": col_items})
    except Exception:
        # best-effort; return whatever collected
        pass
    return out


# PUBLIC_INTERFACE
def try_parse_inline_sql(query: str) -> Optional[str]:
    """
    PUBLIC_INTERFACE
    Try to extract or detect SQL from a user query.

    Heuristics:
      - If query starts with common SQL leading keywords (select/with/explain/show),
        treat as SQL as-is.
      - If the query is wrapped in code fences ```sql ... ```, extract inner content.

    Returns:
        The SQL string if detected, otherwise None.
    """
    if not query:
        return None
    q = query.strip()
    lower = q.lower()
    # code fence
    if lower.startswith("```sql"):
        end = lower.rfind("```")
        if end > 0:
            inner = q[6:end].strip()
            return inner if inner else None
    # common SQL starts
    for lead in ("select", "with", "explain", "show", "describe", "pragma"):
        if lower.startswith(lead + " "):
            return q
    return None


# PUBLIC_INTERFACE
def execute_sql_compact(session_id: str, sql: str, row_limit: int = 50, col_limit: int = 24) -> Tuple[str, int, int]:
    """
    PUBLIC_INTERFACE
    Execute SQL against the session's DuckDB connection and return a compact TSV string.

    Args:
        session_id: Session identifier.
        sql: SQL text to execute.
        row_limit: Maximum number of rows to return in the preview.
        col_limit: Maximum number of columns to include.

    Returns:
        (tsv_text, total_rows, total_cols)
        - tsv_text: a compact TSV with headers + up to row_limit rows (and col_limit columns).
        - total_rows: the total number of rows in the result.
        - total_cols: the number of columns in the result.

    Raises:
        duckdb.Error on SQL issues (caller should catch and render message).
    """
    conn = _get_or_create_session_conn(session_id)
    # Wrap the user's SQL as a subquery to count and limit safely
    try:
        # Compute total rows with a count(*) on the subquery
        cnt_sql = f"SELECT COUNT(*) AS c FROM ({sql}) t"
        total_rows = int(conn.execute(cnt_sql).fetchone()[0])
    except Exception:
        # If count fails (e.g., DDL / pragma), still try to run the query with LIMIT
        total_rows = 0

    # Apply a limit for preview safety (only for SELECT-like result sets)
    preview_sql = sql
    if sql.strip().lower().startswith(("select", "with")):
        preview_sql = f"SELECT * FROM ({sql}) t LIMIT {max(1, row_limit)}"

    df = conn.execute(preview_sql).fetch_df()
    total_cols = int(df.shape[1]) if df is not None else 0
    if df is None or df.empty:
        return "No rows.", total_rows, total_cols

    # Limit columns for display
    if df.shape[1] > col_limit:
        df = df.iloc[:, :col_limit]

    # Convert to TSV
    headers = "\t".join([str(c) for c in df.columns.tolist()])
    rows = ["\t".join("" if (v is None) else str(v) for v in r) for r in df.values.tolist()]
    tsv = headers + ("\n" + "\n".join(rows) if rows else "")

    return tsv, total_rows, total_cols
