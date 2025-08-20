"""Chroma-backed persistent vector store for chatbot embeddings.

This module integrates ChromaDB as the vector database for storing embeddings and
associated metadata. It also maintains a separate SQLAlchemy mapping table for Excel
row/chunk IDs to support exact filtering and context aggregation.

Environment variables:
    CHATBOT_CHROMA_DIR           -> Directory for Chroma persistence (default: ./chroma_data)
    CHATBOT_CHROMA_COLLECTION    -> Chroma collection name (default: chatbot_embeddings)

Public functions preserved for compatibility:
    - add_embeddings(session_id, items, vectors, model) -> int
    - delete_session_embeddings(session_id) -> int
    - get_session_embeddings_count(session_id) -> int
    - get_session_embedding_items(session_id, source_types=None, max_records=None) -> List[dict]
"""
from __future__ import annotations

import os
import json
import hashlib
from typing import List, Optional, Dict, Any, Tuple

# SQLAlchemy base/session reused from auth_utils
from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    DateTime,
    Float,
    UniqueConstraint,
    func,
)
from sqlalchemy.types import JSON as SAJSON  # Distinct from Python json
from .auth_utils import Base, SessionLocal

# Try to import Chroma with graceful fallback
try:
    import chromadb  # type: ignore
    _CHROMA_AVAILABLE = True
except Exception:
    chromadb = None  # type: ignore
    _CHROMA_AVAILABLE = False


# =========================
# SQLAlchemy ORM (mapping)
# =========================

class RowChunkMap(Base):
    """Mapping table to track Excel row/chunk IDs and their corresponding Chroma record.

    Fields:
        id: Primary key.
        session_id: Session identifier.
        row_id: Logical Excel row identifier (e.g., "Sheet1:r12").
        chunk_id: Logical chunk identifier (e.g., "Sheet1:r12:c1of3").
        chroma_id: The record ID used in Chroma.
        filename: Source filename.
        sheet_name: Sheet name.
        row_number: Row number (1-based).
        chunk_index: Chunk index (0-based).
        chunk_count: Total chunks for the row.
        created_at: Row creation timestamp.
    """
    __tablename__ = "row_chunk_map"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(128), index=True, nullable=False)
    row_id = Column(String(256), nullable=True, index=True)
    chunk_id = Column(String(256), nullable=True, index=True)
    chroma_id = Column(String(256), nullable=False, index=True)
    filename = Column(String(260), nullable=True)
    sheet_name = Column(String(128), nullable=True)
    row_number = Column(Integer, nullable=True)
    chunk_index = Column(Integer, nullable=True)
    chunk_count = Column(Integer, nullable=True)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)

    __table_args__ = (
        UniqueConstraint("session_id", "chunk_id", name="uix_session_chunk"),
    )


# Backward-compatibility: keep the historical EmbeddingRecord model definition for migrations,
# but it is no longer used for new writes. It can be retained to avoid import errors and
# allow future data migration if necessary.
class EmbeddingRecord(Base):
    """Deprecated SQLAlchemy model. No longer used for persistence of new embeddings."""
    __tablename__ = "embeddings"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(128), index=True, nullable=False)
    filename = Column(String(260), nullable=True)
    source_type = Column(String(32), nullable=False)
    key = Column(String(512), nullable=True)
    text = Column(Text, nullable=False)
    embedding = Column(SAJSON, nullable=True)
    model = Column(String(64), nullable=False, default="models/text-embedding-004")
    vector_norm = Column(Float, nullable=True)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)


# =========================
# Chroma setup utilities
# =========================

def _get_chroma_dir() -> str:
    """Resolve Chroma persistence directory (creates it if missing)."""
    base = os.getenv("CHATBOT_CHROMA_DIR", os.path.join(".", "chroma_data"))
    os.makedirs(base, exist_ok=True)
    return base


def _get_chroma_collection_name() -> str:
    """Resolve Chroma collection name."""
    return os.getenv("CHATBOT_CHROMA_COLLECTION", "chatbot_embeddings")


_CHROMA_CLIENT = None
_CHROMA_COLLECTION = None


def _ensure_chroma() -> Tuple[Any, Any]:
    """Create/return a Chroma persistent client and collection, handling API variations."""
    if not _CHROMA_AVAILABLE:
        raise RuntimeError("ChromaDB is not installed. Please ensure 'chromadb' is available.")

    global _CHROMA_CLIENT, _CHROMA_COLLECTION
    if _CHROMA_CLIENT is not None and _CHROMA_COLLECTION is not None:
        return _CHROMA_CLIENT, _CHROMA_COLLECTION

    # Instantiate persistent client depending on the installed version.
    persist_dir = _get_chroma_dir()
    coll_name = _get_chroma_collection_name()

    client = None
    try:
        # Chroma >= 0.5
        if hasattr(chromadb, "PersistentClient"):
            client = chromadb.PersistentClient(path=persist_dir)  # type: ignore[attr-defined]
        else:
            # Chroma 0.4.x
            try:
                from chromadb.config import Settings  # type: ignore
            except Exception:
                Settings = None  # type: ignore
            if Settings is None:
                raise RuntimeError("chromadb.config.Settings not available in this version.")
            client = chromadb.Client(
                Settings(chroma_db_impl="duckdb+parquet", persist_directory=persist_dir)  # type: ignore
            )
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Chroma client: {e}")

    # Create or get collection (no embedding function; we pass precomputed embeddings)
    try:
        collection = client.get_or_create_collection(name=coll_name, metadata={"hnsw:space": "cosine"})
    except TypeError:
        # Some versions accept only 'name'
        collection = client.get_or_create_collection(name=coll_name)
    except Exception as e:
        raise RuntimeError(f"Failed to get/create Chroma collection '{coll_name}': {e}")

    _CHROMA_CLIENT = client
    _CHROMA_COLLECTION = collection
    return client, collection


def _deterministic_hash(*parts: str, max_len: int = 24) -> str:
    """Build a deterministic short hash for a set of string parts."""
    h = hashlib.sha1("||".join(parts).encode("utf-8", errors="ignore")).hexdigest()
    return h[:max_len]


def _make_chroma_id(session_id: str, item: Dict[str, Any], vec: Optional[List[float]]) -> str:
    """Create a stable Chroma record ID.

    If the item has an explicit 'chunk_id' (from Excel row-chunk), we prefer:
        f"{session_id}::chunk::{chunk_id}"
    Otherwise, create a deterministic hash from key fields.
    """
    # If item already includes chunk_id metadata, prefer that
    chunk_id = None
    # chunk_id may be nested in item["key"] as JSON (for persisted form) or as a direct field
    if "chunk_id" in item and item.get("chunk_id"):
        chunk_id = str(item.get("chunk_id"))
    else:
        key_val = item.get("key")
        if isinstance(key_val, str):
            try:
                meta = json.loads(key_val)
                chunk_id = meta.get("chunk_id")
            except Exception:
                chunk_id = None

    if chunk_id:
        return f"{session_id}::chunk::{chunk_id}"

    # Fall back to hash of relevant attributes
    filename = str(item.get("filename") or "")
    source_type = str(item.get("source_type") or "file_text")
    key = str(item.get("key") or "")
    text = str(item.get("text") or "")
    vec_tag = "v1" if vec else "v0"
    dh = _deterministic_hash(session_id, source_type, filename, key, text)
    return f"{session_id}::{source_type}::{dh}::{vec_tag}"


def _extract_rowchunk_meta(item: Dict[str, Any]) -> Dict[str, Any]:
    """Extract row/chunk metadata from item['key'] (JSON) or direct fields."""
    md: Dict[str, Any] = {}

    # Direct fields (in-memory entries from main.py carry these)
    for fld in ("row_id", "chunk_id", "sheet_name", "row_number", "chunk_index", "chunk_count"):
        if fld in item and item.get(fld) is not None:
            md[fld] = item.get(fld)

    # If not present, try to parse JSON in key
    key_val = item.get("key")
    if isinstance(key_val, str):
        try:
            parsed = json.loads(key_val)
            if isinstance(parsed, dict):
                for fld in ("row_id", "chunk_id", "sheet_name", "row_number", "chunk_index", "chunk_count"):
                    if fld in parsed and parsed.get(fld) is not None and fld not in md:
                        md[fld] = parsed.get(fld)
        except Exception:
            pass

    return md


def _upsert_rowchunk_mappings(session_id: str, mappings: List[Dict[str, Any]]) -> int:
    """Insert row/chunk mappings into SQLAlchemy table, ignoring duplicates by constraint."""
    if not mappings:
        return 0
    db = SessionLocal()
    inserted = 0
    try:
        for m in mappings:
            # Skip if missing identifiers
            if not m.get("chunk_id") or not m.get("chroma_id"):
                continue
            # Check existing to avoid raising on unique constraint for common backends
            exists = (
                db.query(RowChunkMap)
                .filter(
                    RowChunkMap.session_id == session_id,
                    RowChunkMap.chunk_id == m.get("chunk_id"),
                )
                .first()
            )
            if exists:
                continue

            row = RowChunkMap(
                session_id=session_id,
                row_id=m.get("row_id"),
                chunk_id=m.get("chunk_id"),
                chroma_id=m.get("chroma_id"),
                filename=m.get("filename"),
                sheet_name=m.get("sheet_name"),
                row_number=m.get("row_number"),
                chunk_index=m.get("chunk_index"),
                chunk_count=m.get("chunk_count"),
            )
            db.add(row)
            inserted += 1
        if inserted:
            db.commit()
        return inserted
    finally:
        db.close()


# =========================
# Public API (Chroma-backed)
# =========================

# PUBLIC_INTERFACE
def add_embeddings(
    session_id: str,
    items: List[Dict[str, Any]],
    vectors: List[Optional[List[float]]],
    model: str,
) -> int:
    """PUBLIC_INTERFACE
    Persist a batch of embeddings for a session into Chroma, and store row/chunk mappings.

    Args:
        session_id: Session identifier.
        items: List of item dicts with keys:
            text (str), filename (str), source_type (str), key (Optional[str]).
            For Excel row chunks, row/chunk metadata should be included either
            directly (row_id, chunk_id, sheet_name, row_number, chunk_index, chunk_count)
            or JSON-encoded in 'key'.
        vectors: Precomputed embedding vectors (list[float]) or None for each item.
                Only items with vectors are upserted into Chroma.
        model: Embedding model name.

    Returns:
        int: Number of records upserted into Chroma.
    """
    if not items:
        return 0

    if not _CHROMA_AVAILABLE:
        # Graceful no-op if Chroma not installed (keeps compatibility)
        return 0

    # Ensure Chroma collection exists
    _, collection = _ensure_chroma()

    # Build payloads; Chroma requires all lists to be same length
    ids: List[str] = []
    documents: List[str] = []
    metadatas: List[Dict[str, Any]] = []
    embeddings: List[List[float]] = []
    mappings: List[Dict[str, Any]] = []

    for item, vec in zip(items, vectors):
        # Only persist entries that have an embedding vector
        if not isinstance(vec, list) or not vec:
            continue

        text = str(item.get("text") or "")
        filename = item.get("filename")
        source_type = item.get("source_type") or "file_text"
        key = item.get("key")

        rowchunk_meta = _extract_rowchunk_meta(item)
        chroma_id = _make_chroma_id(session_id, {**item, **rowchunk_meta}, vec)

        meta = {
            "session_id": session_id,
            "model": model,
            "filename": filename,
            "source_type": source_type,
            "key": key,
            # Promote row/chunk fields for exact filtering/aggregation
            **rowchunk_meta,
        }

        ids.append(chroma_id)
        documents.append(text)
        metadatas.append(meta)
        embeddings.append([float(x) for x in vec])

        # If row/chunk metadata present, track a mapping record
        if "chunk_id" in rowchunk_meta and rowchunk_meta.get("chunk_id"):
            mappings.append({
                "row_id": rowchunk_meta.get("row_id"),
                "chunk_id": rowchunk_meta.get("chunk_id"),
                "chroma_id": chroma_id,
                "filename": filename,
                "sheet_name": rowchunk_meta.get("sheet_name"),
                "row_number": rowchunk_meta.get("row_number"),
                "chunk_index": rowchunk_meta.get("chunk_index"),
                "chunk_count": rowchunk_meta.get("chunk_count"),
            })

    if not ids:
        return 0

    # Upsert into Chroma
    try:
        collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )
    except Exception:
        # If upsert fails entirely, do not insert mappings
        # but do not raise to avoid breaking uploads.
        return 0

    # Update mapping table for row/chunk metadata
    try:
        if mappings:
            _upsert_rowchunk_mappings(session_id, mappings)
    except Exception:
        # Mapping failures should not break overall flow
        pass

    return len(ids)


# PUBLIC_INTERFACE
def delete_session_embeddings(session_id: str) -> int:
    """PUBLIC_INTERFACE
    Delete all embedding records for a given session from Chroma and clear mappings.

    Args:
        session_id: Session identifier to clear.

    Returns:
        int: Number of Chroma records deleted (best-effort; may be an estimate).
    """
    if not _CHROMA_AVAILABLE:
        return 0

    _, collection = _ensure_chroma()

    # Count first (best-effort)
    try:
        count_before = get_session_embeddings_count(session_id)
    except Exception:
        count_before = 0

    # Delete from Chroma
    try:
        collection.delete(where={"session_id": session_id})
    except Exception:
        pass

    # Delete mappings
    db = SessionLocal()
    try:
        q = db.query(RowChunkMap).filter(RowChunkMap.session_id == session_id)
        q.delete(synchronize_session=False)
        db.commit()
    finally:
        db.close()

    return int(count_before)


def _collection_get(collection, where: Dict[str, Any], limit: Optional[int]) -> Dict[str, Any]:
    """Compatibility wrapper around collection.get for various Chroma versions."""
    include = ["documents", "metadatas", "embeddings"]
    try:
        if limit and limit > 0:
            return collection.get(where=where, include=include, limit=int(limit))
        return collection.get(where=where, include=include)
    except TypeError:
        # Older versions may not support 'limit' parameter
        return collection.get(where=where, include=include)


# PUBLIC_INTERFACE
def get_session_embeddings_count(session_id: str) -> int:
    """PUBLIC_INTERFACE
    Return the number of embedding records stored for a session in Chroma.

    Args:
        session_id: Session identifier.

    Returns:
        int: Count of records (best-effort based on collection.get()).
    """
    if not _CHROMA_AVAILABLE:
        return 0

    _, collection = _ensure_chroma()
    try:
        res = _collection_get(collection, where={"session_id": session_id}, limit=None)
        ids = res.get("ids") or []
        return len(ids)
    except Exception:
        return 0


# PUBLIC_INTERFACE
def get_session_embedding_items(
    session_id: str,
    source_types: Optional[List[str]] = None,
    max_records: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """PUBLIC_INTERFACE
    Fetch stored embedding items for a session from Chroma.

    This is used during retrieval to include persisted items (e.g., per-row Excel documents)
    even after process restarts or when the in-memory index is empty.

    Args:
        session_id: Session identifier whose items should be returned.
        source_types: Optional list of source types to include (e.g., ["xlsx_row_chunk", "xlsx_row"]).
        max_records: Optional maximum number of records to return.

    Returns:
        List[Dict[str, Any]]: Each item has keys:
            - text (str)
            - filename (Optional[str])
            - source_type (str)
            - key (Optional[str])
            - embedding (Optional[List[float]])
            - model (str)
            - (optional) row_id, chunk_id, sheet_name, row_number, chunk_index, chunk_count
    """
    if not _CHROMA_AVAILABLE:
        return []

    _, collection = _ensure_chroma()

    where: Dict[str, Any] = {"session_id": session_id}
    if source_types:
        # Add source type filter
        where = {"$and": [where, {"source_type": {"$in": source_types}}]}

    try:
        res = _collection_get(collection, where=where, limit=max_records)
        ids = res.get("ids") or []
        docs = res.get("documents") or []
        metas = res.get("metadatas") or []
        embs = res.get("embeddings") or []

        out: List[Dict[str, Any]] = []
        for i in range(min(len(ids), len(docs), len(metas), len(embs))):
            md = metas[i] or {}
            out_item = {
                "text": docs[i] or "",
                "filename": md.get("filename"),
                "source_type": md.get("source_type") or "file_text",
                "key": md.get("key"),
                "embedding": embs[i] if isinstance(embs[i], list) else None,
                "model": md.get("model") or "models/text-embedding-004",
            }
            # Promote row/chunk metadata if available
            for fld in ("row_id", "chunk_id", "sheet_name", "row_number", "chunk_index", "chunk_count"):
                if fld in md:
                    out_item[fld] = md.get(fld)
            out.append(out_item)

        return out
    except Exception:
        return []


# PUBLIC_INTERFACE
def get_row_chunks(
    session_id: str,
    row_id: str,
    max_chunks: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """PUBLIC_INTERFACE
    Fetch all stored items for a specific Excel row (by row_id), ordered by chunk_index.

    Uses Chroma metadata filtering (session_id + row_id) to retrieve row chunks. Falls back
    to using the RowChunkMap table and fetching by stored Chroma IDs if direct filtering
    is unsupported on the installed ChromaDB version.

    Args:
        session_id: The session identifier.
        row_id: Logical row identifier (e.g., "Sheet1:r12").
        max_chunks: Optional maximum number of chunks to return.

    Returns:
        List[Dict[str, Any]]: Items with at least
            text, filename, source_type, key, (optional) row_id, chunk_id, sheet_name,
            row_number, chunk_index, chunk_count, embedding, model.
    """
    if not _CHROMA_AVAILABLE:
        return []

    _, collection = _ensure_chroma()

    def _sort_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        try:
            return sorted(items, key=lambda x: (x.get("chunk_index") if x.get("chunk_index") is not None else 0))
        except Exception:
            return items

    # Attempt direct metadata-filtered retrieval
    where = {"$and": [
        {"session_id": session_id},
        {"row_id": row_id},
        {"source_type": {"$in": ["xlsx_row_chunk", "xlsx_row"]}},
    ]}
    try:
        res = _collection_get(collection, where=where, limit=max_chunks)
        docs = res.get("documents") or []
        metas = res.get("metadatas") or []
        embs = res.get("embeddings") or []
        items: List[Dict[str, Any]] = []
        for i in range(min(len(docs), len(metas), len(embs))):
            md = metas[i] or {}
            item = {
                "text": docs[i] or "",
                "filename": md.get("filename"),
                "source_type": md.get("source_type") or "file_text",
                "key": md.get("key"),
                "embedding": embs[i] if isinstance(embs[i], list) else None,
                "model": md.get("model") or "models/text-embedding-004",
            }
            for fld in ("row_id", "chunk_id", "sheet_name", "row_number", "chunk_index", "chunk_count"):
                if fld in md:
                    item[fld] = md.get(fld)
            items.append(item)
        if items:
            return _sort_items(items)
    except Exception:
        # Fall through to mapping-based retrieval
        pass

    # Fallback: use mapping table to fetch chroma IDs for row_id then pull by ids
    db = SessionLocal()
    try:
        q = (
            db.query(RowChunkMap)
            .filter(RowChunkMap.session_id == session_id, RowChunkMap.row_id == row_id)
            .order_by(RowChunkMap.chunk_index.asc())
        )
        rows = q.all()
    finally:
        db.close()

    if not rows:
        return []

    # Limit number of chunks if requested
    selected = rows[: max_chunks] if max_chunks and max_chunks > 0 else rows
    ids = [r.chroma_id for r in selected if r.chroma_id]

    # Many Chroma versions support collection.get(ids=...)
    try:
        include = ["documents", "metadatas", "embeddings"]
        res = collection.get(ids=ids, include=include)
        docs = res.get("documents") or []
        metas = res.get("metadatas") or []
        embs = res.get("embeddings") or []
        out: List[Dict[str, Any]] = []
        for i in range(min(len(docs), len(metas), len(embs))):
            md = metas[i] or {}
            item = {
                "text": docs[i] or "",
                "filename": md.get("filename"),
                "source_type": md.get("source_type") or "file_text",
                "key": md.get("key"),
                "embedding": embs[i] if isinstance(embs[i], list) else None,
                "model": md.get("model") or "models/text-embedding-004",
            }
            for fld in ("row_id", "chunk_id", "sheet_name", "row_number", "chunk_index", "chunk_count"):
                if fld in md:
                    item[fld] = md.get(fld)
            out.append(item)
        return _sort_items(out)
    except Exception:
        return []


def _safe_norm(vec: Optional[List[float]]) -> Optional[float]:
    """Compute L2 norm of the vector if available. Retained for backward compatibility."""
    if not vec:
        return None
    try:
        import math
        return math.sqrt(sum(float(x) * float(x) for x in vec))
    except Exception:
        return None
