from __future__ import annotations

from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from pinecone import Pinecone, ServerlessSpec


@dataclass
class VectorRecord:
    id: str
    values: List[float]
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class SearchMatch:
    id: str
    score: float
    metadata: Optional[Dict[str, Any]]


@dataclass
class SearchResults:
    matches: List[SearchMatch]


class PineconeVectorStore:
    """
    Pinecone vector store helper.

    Notes:
    - Ensure the index is created beforehand with correct dimension and metric (cosine).
      You may create the index serverlessly with:
        from pinecone import Pinecone, ServerlessSpec
        pc = Pinecone(api_key=...)
        pc.create_index(
            name="YOUR_INDEX",
            dimension=768,              # match your embedding model
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
      Or use provisioned/legacy environments with the environment param and without host.

    - If you are using serverless (recommended), pass host=<index_host> to connect:
        idx = pc.IndexHost("https://YOUR_INDEX-xxxx.svc.xxxxxx.pinecone.io")
      pinecone-client v5 provides pc.Index(name) and pc.Index(host) APIs; choose accordingly.
    """

    def __init__(
        self,
        api_key: str,
        index_name: Optional[str] = None,
        namespace: Optional[str] = None,
        host: Optional[str] = None,
        environment: Optional[str] = None,
        top_k: int = 3,
    ):
        if not api_key:
            raise ValueError("Pinecone API key is required")

        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        self.namespace = namespace or "default"
        self.top_k = top_k

        # Connect to index
        # If host provided (serverless), use it; otherwise use name (provisioned or same project)
        if host:
            self.index = self.pc.Index(host=host)
        else:
            if not index_name:
                raise ValueError("Either host or index_name must be provided to connect to Pinecone index")
            self.index = self.pc.Index(index_name)

    # CRUD operations
    def upsert(self, records: List[VectorRecord]) -> None:
        if not records:
            return
        vectors = []
        for r in records:
            item = {"id": r.id, "values": r.values}
            if r.metadata:
                item["metadata"] = r.metadata
            vectors.append(item)
        # Upsert in a single batch (consider batching for very large uploads)
        self.index.upsert(vectors=vectors, namespace=self.namespace)

    def delete_by_session(self, session_id: str) -> None:
        # Deletes all vectors for a given session_id using metadata filter
        self.index.delete(
            delete_all=False,
            namespace=self.namespace,
            filter={"session_id": {"$eq": session_id}},
        )

    def delete_by_ids(self, ids: List[str]) -> None:
        if not ids:
            return
        self.index.delete(ids=ids, namespace=self.namespace)

    def search(
        self,
        vector: List[float],
        top_k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True,
    ) -> SearchResults:
        k = top_k or self.top_k
        res = self.index.query(
            namespace=self.namespace,
            vector=vector,
            top_k=k,
            filter=filter,
            include_values=False,
            include_metadata=include_metadata,
        )
        matches: List[SearchMatch] = []
        for m in res.get("matches", []):
            matches.append(
                SearchMatch(
                    id=m.get("id"),
                    score=m.get("score", 0.0),
                    metadata=m.get("metadata"),
                )
            )
        return SearchResults(matches=matches)
