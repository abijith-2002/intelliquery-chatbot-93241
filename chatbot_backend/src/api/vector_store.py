import os
import uuid
import json
from typing import Any, Dict, Iterable, List, Optional

from pydantic import BaseModel, Field


# PUBLIC_INTERFACE
class VectorRecord(BaseModel):
    """A single embedding record to persist to a vector store."""
    id: str = Field(..., description="Unique id for the row/vector")
    session_id: str = Field(..., description="Chat session id used to scope retrieval")
    namespace: str = Field(..., description="Logical namespace, e.g., 'xlsx:filename:sheet'")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary metadata to store alongside the vector")
    vector: List[float] = Field(..., description="Embedding vector")


# PUBLIC_INTERFACE
class IVectorStore:
    """PUBLIC_INTERFACE
    Interface for a vector store implementation.
    """

    def upsert(self, records: Iterable[VectorRecord]) -> int:
        """PUBLIC_INTERFACE
        Insert or update a collection of vector records.

        Args:
            records (Iterable[VectorRecord]): Records to persist.

        Returns:
            int: Number of records upserted.
        """
        raise NotImplementedError()

    # PUBLIC_INTERFACE
    def search(self, session_id: str, query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """PUBLIC_INTERFACE
        Semantic search by cosine similarity.

        Args:
            session_id (str): Scope vectors to a chat session.
            query_vector (List[float]): Embedding for the query.
            top_k (int): Number of top rows to return.

        Returns:
            List[Dict[str, Any]]: Ranked results as list of {"id", "score", "namespace", "metadata"}.
        """
        raise NotImplementedError()

    # PUBLIC_INTERFACE
    def keyword_filter(self, session_id: str, where: Dict[str, Any], limit: int = 50) -> List[Dict[str, Any]]:
        """PUBLIC_INTERFACE
        Best-effort metadata lookup to filter records using exact/contains matches.

        Implementations may translate this to SQL JSONB filters (pgvector) or
        in-memory filtering (faiss demo backend).

        Args:
            session_id (str): Session scope.
            where (Dict[str, Any]): Field->value constraints to match in metadata.
            limit (int): Max records.

        Returns:
            List[Dict[str, Any]]: Matching records with {"id","namespace","metadata"}.
        """
        raise NotImplementedError()

    def close(self) -> None:
        """Close resources if needed."""
        pass


class _FaissIndexWrapper:
    """Lazy FAISS index wrapper with simple L2 index."""
    def __init__(self, dim: int):
        self.dim = dim
        try:
            import faiss  # type: ignore
        except Exception as e:
            raise RuntimeError(f"FAISS backend not available: {e}")
        self.faiss = faiss
        self.index = self.faiss.IndexFlatIP(dim)  # cosine via normalized vectors (we will norm)
        self.ids: List[str] = []
        self.meta: List[Dict[str, Any]] = []

    @staticmethod
    def _normalize(v: List[float]) -> List[float]:
        import math
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def add(self, ids: List[str], vecs: List[List[float]], metas: List[Dict[str, Any]]) -> int:
        import numpy as np  # faiss expects numpy arrays
        if not vecs:
            return 0
        dim = len(vecs[0])
        if self.index.d != dim:
            # rebuild index if dimension changes (shouldn't, but safety)
            self.index = self.faiss.IndexFlatIP(dim)
        normed = [self._normalize(v) for v in vecs]
        arr = np.array(normed, dtype="float32")
        self.index.add(arr)
        self.ids.extend(ids)
        self.meta.extend(metas)
        return len(vecs)


class _FaissVectorStore(IVectorStore):
    """In-memory FAISS vector store. Good for demos and small-scale runs."""
    def __init__(self):
        self._idx: Optional[_FaissIndexWrapper] = None

    def upsert(self, records: Iterable[VectorRecord]) -> int:
        items = list(records)
        if not items:
            return 0
        # Ensure consistent dims
        dim = len(items[0].vector)
        ids = []
        vecs = []
        metas = []
        for r in items:
            if len(r.vector) != dim:
                continue  # skip inconsistent
            ids.append(r.id)
            vecs.append(r.vector)
            metas.append({"session_id": r.session_id, "namespace": r.namespace, **r.metadata})
        if not vecs:
            return 0
        if self._idx is None:
            self._idx = _FaissIndexWrapper(dim)
        return self._idx.add(ids, vecs, metas)

    def search(self, session_id: str, query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        if self._idx is None or not self._idx.ids:
            return []
        # cosine via inner product on normalized vectors
        import numpy as np
        norm_q = self._idx._normalize(query_vector)
        q = np.array([norm_q], dtype="float32")
        D, I = self._idx.index.search(q, min(top_k, len(self._idx.ids)))
        results: List[Dict[str, Any]] = []
        for score, idx in zip(D[0], I[0]):
            if idx == -1:
                continue
            meta = self._idx.meta[idx]
            if meta.get("session_id") != session_id:
                continue
            results.append({
                "id": self._idx.ids[idx],
                "score": float(score),
                "namespace": meta.get("namespace"),
                "metadata": {k: v for k, v in meta.items() if k not in {"session_id", "namespace"}},
            })
        # Sort by score desc and trim
        results.sort(key=lambda r: r.get("score", 0.0), reverse=True)
        return results[:top_k]

    def keyword_filter(self, session_id: str, where: Dict[str, Any], limit: int = 50) -> List[Dict[str, Any]]:
        if self._idx is None or not self._idx.ids:
            return []
        def matches(meta: Dict[str, Any]) -> bool:
            # all constraints must match
            for key, val in (where or {}).items():
                mval = meta.get(key)
                if isinstance(val, str):
                    # case-insensitive contains for strings
                    if mval is None or str(val).lower() not in str(mval).lower():
                        return False
                else:
                    if mval != val:
                        return False
            return True
        out: List[Dict[str, Any]] = []
        for _id, meta in zip(self._idx.ids, self._idx.meta):
            if meta.get("session_id") != session_id:
                continue
            if matches(meta):
                out.append({"id": _id, "namespace": meta.get("namespace"), "metadata": {k: v for k, v in meta.items() if k not in {"session_id","namespace"}}})
            if len(out) >= limit:
                break
        return out

    def close(self) -> None:
        self._idx = None


class _PgVectorStore(IVectorStore):
    """pgvector-backed implementation using SQLAlchemy Core."""
    def __init__(self, url: str):
        from sqlalchemy import create_engine
        self.engine = create_engine(url)
        with self.engine.begin() as conn:
            # Ensure extension and table exist. The dimension can vary, so we store as vector without fixed dim constraint.
            conn.exec_driver_sql("CREATE EXTENSION IF NOT EXISTS vector;")
            conn.exec_driver_sql(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    namespace TEXT NOT NULL,
                    metadata JSONB DEFAULT '{}'::jsonb,
                    vector vector
                );
                """
            )

    def upsert(self, records: Iterable[VectorRecord]) -> int:
        from sqlalchemy import text
        items = list(records)
        if not items:
            return 0

        # pgvector expects arrays; we'll send as Python list to SQLAlchemy.
        sql = text("""
            INSERT INTO embeddings (id, session_id, namespace, metadata, vector)
            VALUES (:id, :session_id, :namespace, CAST(:metadata AS JSONB), :vector)
            ON CONFLICT (id) DO UPDATE
            SET session_id = EXCLUDED.session_id,
                namespace = EXCLUDED.namespace,
                metadata = EXCLUDED.metadata,
                vector = EXCLUDED.vector
        """)
        cnt = 0
        with self.engine.begin() as conn:
            for r in items:
                conn.execute(sql, {
                    "id": r.id,
                    "session_id": r.session_id,
                    "namespace": r.namespace,
                    "metadata": json.dumps(r.metadata),
                    "vector": r.vector,
                })
                cnt += 1
        return cnt

    def search(self, session_id: str, query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        from sqlalchemy import text
        # Use cosine similarity via <#> operator if available (pgvector supports distance operators)
        # We order by distance ascending and convert to score = 1 - distance (best-effort).
        sql = text("""
            SELECT id, namespace, metadata, 1 - (vector <#> :qvec) AS score
            FROM embeddings
            WHERE session_id = :sid
            ORDER BY vector <#> :qvec ASC
            LIMIT :k
        """)
        with self.engine.begin() as conn:
            rows = conn.execute(sql, {"sid": session_id, "qvec": query_vector, "k": int(top_k)}).mappings().all()
        out: List[Dict[str, Any]] = []
        for r in rows:
            out.append({
                "id": r["id"],
                "score": float(r["score"]) if r["score"] is not None else 0.0,
                "namespace": r["namespace"],
                "metadata": r["metadata"] or {},
            })
        return out

    def keyword_filter(self, session_id: str, where: Dict[str, Any], limit: int = 50) -> List[Dict[str, Any]]:
        # Build JSONB filter predicates. We use ->> cast to text for string contains; equality for non-strings.
        if not where:
            return []
        clauses = ["session_id = :sid"]
        params: Dict[str, Any] = {"sid": session_id, "lim": int(limit)}
        idx = 0
        for key, val in where.items():
            idx += 1
            jpath = f"metadata->>'{key}'"
            if isinstance(val, str):
                param = f"p{idx}"
                clauses.append(f"LOWER({jpath}) LIKE LOWER('%' || :{param} || '%')")
                params[param] = val
            else:
                param = f"p{idx}"
                clauses.append(f"metadata->>'{key}' = :{param}")
                params[param] = str(val)
        from sqlalchemy import text
        sql = text(f"""
            SELECT id, namespace, metadata
            FROM embeddings
            WHERE {' AND '.join(clauses)}
            LIMIT :lim
        """)
        with self.engine.begin() as conn:
            rows = conn.execute(sql, params).mappings().all()
        return [{"id": r["id"], "namespace": r["namespace"], "metadata": r["metadata"] or {}} for r in rows]

    def close(self) -> None:
        # nothing
        pass


# PUBLIC_INTERFACE
def get_vector_store() -> IVectorStore:
    """PUBLIC_INTERFACE
    Resolve vector store based on env:
      - VECTOR_STORE=pgvector requires CHATBOT_PGVECTOR_URL
      - VECTOR_STORE=faiss (default if unset)

    Returns:
        IVectorStore: Vector store instance to use for upserts.
    """
    backend = (os.getenv("VECTOR_STORE") or "faiss").lower()
    if backend == "pgvector":
        url = os.getenv("CHATBOT_PGVECTOR_URL")
        if not url:
            raise RuntimeError("CHATBOT_PGVECTOR_URL required when VECTOR_STORE=pgvector")
        return _PgVectorStore(url)
    # default to faiss
    return _FaissVectorStore()


# PUBLIC_INTERFACE
def build_vector_id(prefix: str = "row") -> str:
    """PUBLIC_INTERFACE
    Build a unique vector id with given prefix.
    """
    return f"{prefix}-{uuid.uuid4()}"
