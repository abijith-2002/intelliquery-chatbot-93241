import os
from typing import List, Dict, Any, Optional, Iterable

try:
    # pinecone-client 5.x (new SDK)
    from pinecone import Pinecone  # type: ignore
    _PINECONE_AVAILABLE = True
except Exception:
    # If dependency is not installed yet, we gracefully handle it.
    _PINECONE_AVAILABLE = False

try:
    import backoff  # type: ignore
except Exception:
    # Fallback no-op backoff decorator
    def backoff_on_exception():
        def _decorator(fn):
            return fn
        return _decorator
    backoff = type("backoff", (), {"on_exception": lambda *args, **kwargs: backoff_on_exception()})


class PineconeVectorStore:
    """
    PUBLIC_INTERFACE
    Pinecone vector store wrapper for upserting and querying text chunk embeddings.

    This class is designed to be optional and non-intrusive:
    - If environment variables or dependency are missing, `is_configured` will be False and
      methods will no-op or return empty results.
    - It expects embeddings generated elsewhere (e.g., via Gemini).

    Required environment variables:
        - PINECONE_API_KEY
        - PINECONE_INDEX_NAME
        - One of:
            - PINECONE_HOST (serverless endpoint)
            - PINECONE_ENVIRONMENT (legacy/provisioned)

    Optional:
        - PINECONE_NAMESPACE (default: "default")
        - PINECONE_TOP_K (default: "3")
    """

    def __init__(
        self,
        api_key: Optional[str],
        index_name: Optional[str],
        host: Optional[str],
        environment: Optional[str],
        namespace: str = "default",
        top_k_default: int = 3,
    ) -> None:
        self.api_key = api_key or ""
        self.index_name = index_name or ""
        self.host = host or ""
        self.environment = environment or ""
        self.namespace = namespace or "default"
        try:
            self.top_k_default = int(top_k_default)
        except Exception:
            self.top_k_default = 3

        self.is_configured = (
            _PINECONE_AVAILABLE
            and bool(self.api_key)
            and bool(self.index_name)
            and (bool(self.host) or bool(self.environment))
        )

        self._pc = None
        self._index = None
        if self.is_configured:
            try:
                self._pc = Pinecone(api_key=self.api_key)
                # New SDK: get index via pc.Index(name, host=?)
                if self.host:
                    self._index = self._pc.Index(self.index_name, host=self.host)
                else:
                    # Environment usage may be legacy; host is preferred in serverless
                    self._index = self._pc.Index(self.index_name)
            except Exception:
                # If initialization fails, mark as not configured to avoid runtime errors
                self.is_configured = False
                self._pc = None
                self._index = None

    # PUBLIC_INTERFACE
    @classmethod
    def from_env(cls) -> "PineconeVectorStore":
        """Create an instance reading configuration from environment variables."""
        return cls(
            api_key=os.getenv("PINECONE_API_KEY"),
            index_name=os.getenv("PINECONE_INDEX_NAME"),
            host=os.getenv("PINECONE_HOST"),
            environment=os.getenv("PINECONE_ENVIRONMENT"),
            namespace=os.getenv("PINECONE_NAMESPACE", "default"),
            top_k_default=int(os.getenv("PINECONE_TOP_K", "3")),
        )

    def _iter_batches(self, items: List[Any], batch_size: int = 100) -> Iterable[List[Any]]:
        for i in range(0, len(items), batch_size):
            yield items[i : i + batch_size]

    # PUBLIC_INTERFACE
    def upsert_chunks(
        self,
        session_id: str,
        filename: str,
        chunks: List[str],
        embeddings: List[Optional[List[float]]],
    ) -> int:
        """
        PUBLIC_INTERFACE
        Upsert chunk embeddings into Pinecone for a given session/file.

        Args:
            session_id: Session identifier.
            filename: Source filename for the chunks.
            chunks: List of chunked text strings.
            embeddings: Embedding vectors corresponding to chunks. Any None vectors will be skipped.

        Returns:
            int: Number of vectors successfully upserted.
        """
        if not self.is_configured or not self._index:
            return 0
        if not chunks or not embeddings:
            return 0

        vectors = []
        for idx, (text, vec) in enumerate(zip(chunks, embeddings)):
            if not vec or not isinstance(vec, list) or len(vec) == 0:
                continue
            vector_id = f"{session_id}:{filename}:{idx}"
            metadata = {
                "session_id": session_id,
                "filename": filename,
                "chunk_index": idx,
                "text": text,
            }
            vectors.append({"id": vector_id, "values": vec, "metadata": metadata})

        if not vectors:
            return 0

        inserted = 0
        for batch in self._iter_batches(vectors, batch_size=100):
            try:
                # backoff on transient exceptions, 429s, etc.
                self._upsert_with_retry(batch)
                inserted += len(batch)
            except Exception:
                # If any batch fails, continue with others; caller can rely on in-memory fallback
                continue
        return inserted

    @backoff.on_exception(backoff.expo, Exception, max_tries=5)
    def _upsert_with_retry(self, batch: List[Dict[str, Any]]) -> None:
        if not self._index:
            raise RuntimeError("Pinecone index is not initialized.")
        self._index.upsert(vectors=batch, namespace=self.namespace)

    # PUBLIC_INTERFACE
    def query_top_k(
        self,
        session_id: str,
        query_embedding: List[float],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        PUBLIC_INTERFACE
        Query Pinecone for top-k similar chunks, filtered by session_id.

        Args:
            session_id: Session identifier to filter results.
            query_embedding: Embedding vector for the user's query.
            top_k: Override for number of results; defaults to env or 3.

        Returns:
            List[Dict[str, Any]]: A list of matches with 'score' and 'metadata' including 'text'.
        """
        if not self.is_configured or not self._index:
            return []
        if not query_embedding or not isinstance(query_embedding, list):
            return []

        k = int(top_k or self.top_k_default or 3)

        try:
            rsp = self._index.query(
                vector=query_embedding,
                top_k=k,
                namespace=self.namespace,
                include_metadata=True,
                filter={"session_id": {"$eq": session_id}},
            )
            matches = getattr(rsp, "matches", None)
            if not matches:
                return []
            results = []
            for m in matches:
                # New SDK match properties typically accessible via attributes; ensure dict-like
                md = getattr(m, "metadata", {}) or {}
                score = getattr(m, "score", 0.0)
                results.append({"score": float(score), "metadata": dict(md)})
            return results
        except Exception:
            return []
