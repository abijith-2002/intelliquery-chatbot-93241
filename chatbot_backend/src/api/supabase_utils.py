import os
from typing import Any, Dict, List, Optional

# Lazily import to keep optional dependency behavior
try:
    from supabase import create_client, Client  # type: ignore
except Exception:  # pragma: no cover
    create_client = None
    Client = None  # type: ignore


# PUBLIC_INTERFACE
def get_supabase_client() -> Optional["Client"]:
    """
    PUBLIC_INTERFACE
    Initialize and return a Supabase client if environment variables are present and library is installed.

    The following environment variables are checked (in order of preference for key):
        - REACT_APP_SUPABASE_URL
        - REACT_APP_SUPABASE_SERVICE_ROLE_KEY (preferred for server-side writes)
        - REACT_APP_SUPABASE_ANON_KEY       (fallback if service role not provided)

    Returns:
        Optional[Client]: Supabase client instance or None if not configured/available.

    Notes:
        Do not hardcode configuration in code; rely on environment variables.
    """
    url = os.getenv("REACT_APP_SUPABASE_URL") or ""
    service_key = os.getenv("REACT_APP_SUPABASE_SERVICE_ROLE_KEY") or ""
    anon_key = os.getenv("REACT_APP_SUPABASE_ANON_KEY") or ""
    if not url:
        return None
    key = service_key or anon_key
    if not key:
        return None
    if create_client is None:
        return None
    try:
        return create_client(url, key)
    except Exception:
        return None


# PUBLIC_INTERFACE
def insert_json_embeddings(
    client: "Client",
    table: str,
    rows: List[Dict[str, Any]],
    batch_size: int = 200,
) -> int:
    """
    PUBLIC_INTERFACE
    Insert JSON embedding rows into Supabase in batches.

    Args:
        client (Client): Supabase client.
        table (str): Destination table name (e.g., "json_embeddings").
        rows (List[Dict[str, Any]]): List of rows; each row should include:
            {
              "session_id": str,
              "filename": str,
              "path": str,
              "value": str,
              "embedding": Optional[List[float]]
            }
        batch_size (int): Number of rows per batch to insert.

    Returns:
        int: Total number of rows attempted to insert (best-effort).

    Raises:
        Exception: If Supabase returns a fatal error. This function is best-effort and will
                   continue batches even if some fail.
    """
    total = 0
    if not rows:
        return 0
    # Split into batches
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        try:
            client.table(table).insert(batch).execute()
        except Exception:
            # Best-effort: continue with other batches
            pass
        total += len(batch)
    return total
