"""Persistent vector store for chatbot embeddings using SQLAlchemy.

This module defines the EmbeddingRecord model and helper functions to persist
embeddings per session. It reuses the shared SQLAlchemy engine/session from
auth_utils to keep a single DB file, controlled by the CHATBOT_SQLALCHEMY_DATABASE_URL
environment variable.

Tables are created when Base.metadata.create_all() is called in auth_utils.create_tables().
To ensure the embeddings table is created, make sure this module is imported before
create_tables() is executed.
"""
from __future__ import annotations

from typing import List, Optional, Dict, Any

from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    DateTime,
    Float,
    func,
)
from sqlalchemy.types import JSON

# Reuse Base, engine, and session factory from auth_utils
from .auth_utils import Base, SessionLocal


class EmbeddingRecord(Base):
    """ORM model to persist embeddings per chunk/document.

    Fields:
        id: Primary key.
        session_id: Session identifier to group embeddings.
        filename: Original source filename (if any).
        source_type: One of "file_text", "json_kv", "xlsx_row".
        key: Optional key for structured sources (e.g., JSON dotted key path).
        text: The chunk/document text that was embedded.
        embedding: The numeric vector as a JSON array (list[float]); may be None if embedding unavailable.
        model: Embedding model name (e.g., "models/text-embedding-004").
        vector_norm: Optional cached vector L2 norm for retrieval optimizations (not currently used).
        created_at: Timestamp of record creation.
    """
    __tablename__ = "embeddings"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(128), index=True, nullable=False)
    filename = Column(String(260), nullable=True)
    source_type = Column(String(32), nullable=False)
    key = Column(String(512), nullable=True)
    text = Column(Text, nullable=False)
    embedding = Column(JSON, nullable=True)
    model = Column(String(64), nullable=False, default="models/text-embedding-004")
    vector_norm = Column(Float, nullable=True)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)


# PUBLIC_INTERFACE
def add_embeddings(
    session_id: str,
    items: List[Dict[str, Any]],
    vectors: List[Optional[List[float]]],
    model: str,
) -> int:
    """PUBLIC_INTERFACE
    Persist a batch of embeddings for a session.

    Args:
        session_id: Session identifier.
        items: List of item dicts with keys: text (str), filename (str), source_type (str), key (Optional[str]).
        vectors: List of embedding vectors (list[float]) or None for each item.
        model: Embedding model name.

    Returns:
        int: Number of records inserted.
    """
    if not items:
        return 0
    # Lengths should match; zip will truncate to shortest for safety.
    to_insert: List[EmbeddingRecord] = []
    for item, vec in zip(items, vectors):
        rec = EmbeddingRecord(
            session_id=session_id,
            filename=item.get("filename"),
            source_type=item.get("source_type") or "file_text",
            key=item.get("key"),
            text=item.get("text") or "",
            embedding=vec if isinstance(vec, list) else None,
            model=model,
            vector_norm=_safe_norm(vec),
        )
        to_insert.append(rec)

    db = SessionLocal()
    try:
        db.add_all(to_insert)
        db.commit()
        return len(to_insert)
    finally:
        db.close()


# PUBLIC_INTERFACE
def delete_session_embeddings(session_id: str) -> int:
    """PUBLIC_INTERFACE
    Delete all embedding records for a given session.

    Args:
        session_id: Session identifier to clear.

    Returns:
        int: Number of rows deleted.
    """
    db = SessionLocal()
    try:
        q = db.query(EmbeddingRecord).filter(EmbeddingRecord.session_id == session_id)
        count = q.count()
        q.delete(synchronize_session=False)
        db.commit()
        return count
    finally:
        db.close()


# PUBLIC_INTERFACE
def get_session_embeddings_count(session_id: str) -> int:
    """PUBLIC_INTERFACE
    Return the number of embedding records stored for a session.

    Args:
        session_id: Session identifier.

    Returns:
        int: Count of records.
    """
    db = SessionLocal()
    try:
        return db.query(EmbeddingRecord).filter(EmbeddingRecord.session_id == session_id).count()
    finally:
        db.close()


# PUBLIC_INTERFACE
def get_session_embedding_items(
    session_id: str,
    source_types: Optional[List[str]] = None,
    max_records: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """PUBLIC_INTERFACE
    Fetch stored embedding items for a session from the persistent vector database.

    This is used during retrieval to include persisted items (e.g., per-row Excel documents)
    even after process restarts or when the in-memory index is empty.

    Args:
        session_id: Session identifier whose items should be returned.
        source_types: Optional list of source types to include (e.g., ["xlsx_row"]).
        max_records: Optional maximum number of records to return.

    Returns:
        List[Dict[str, Any]]: Each item has keys:
            - text (str)
            - filename (Optional[str])
            - source_type (str)
            - key (Optional[str])
            - embedding (Optional[List[float]])
            - model (str)
    """
    db = SessionLocal()
    try:
        q = db.query(EmbeddingRecord).filter(EmbeddingRecord.session_id == session_id)
        if source_types:
            q = q.filter(EmbeddingRecord.source_type.in_(source_types))
        # Order newest first; limit if requested
        q = q.order_by(EmbeddingRecord.id.desc())
        if max_records and max_records > 0:
            q = q.limit(int(max_records))
        rows = q.all()
        results: List[Dict[str, Any]] = []
        for r in rows:
            results.append(
                {
                    "text": r.text or "",
                    "filename": r.filename,
                    "source_type": r.source_type or "file_text",
                    "key": r.key,
                    "embedding": r.embedding if isinstance(r.embedding, list) else None,
                    "model": r.model or "models/text-embedding-004",
                }
            )
        return results
    finally:
        db.close()


def _safe_norm(vec: Optional[List[float]]) -> Optional[float]:
    """Compute L2 norm of the vector if available."""
    if not vec:
        return None
    try:
        import math
        return math.sqrt(sum(float(x) * float(x) for x in vec))
    except Exception:
        return None
