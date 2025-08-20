# ==============================================================================
# IMPORTANT: This backend requires a Google Gemini API key to function.
# Supported environment variable names (checked in this order):
#   - GEMINI_API_KEY
#   - REACT_APP_GEMINI_API_KEY
#   - GOOGLE_API_KEY
#   - GOOGLE_GEMINI_API_KEY
# Make sure to create a `.env` file or provide one of the above at deployment.
# Without this, Gemini responses will be unavailable and fallback messaging will appear.
# ==============================================================================

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, EmailStr
from typing import List, Optional, Dict, Any, Tuple
import google.generativeai as genai
import json

from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory

# Import authentication/database helpers
from .auth_utils import (
    create_tables,
    get_db,
    create_user,
    get_user_by_username,
    verify_password,
)
# Import the chat title router
from .chat_title import router as chat_title_router
# Config utilities
from .config_utils import get_gemini_api_key
# Ensure vector store models are registered before table creation.
from . import vector_store

# Load environment variables
load_dotenv()

# Memory store for chat contexts (keyed by session_id).
CONVERSATION_MEMORY: Dict[str, ConversationBufferMemory] = {}

# Per-session uploaded context store (legacy tracking for previews).
# Structure: { session_id: { "files": [ {filename, size, chars, preview, error?} ], "combined": str } }
CONTEXT_STORE: Dict[str, Dict[str, Any]] = {}

# Per-session in-memory vector index for RAG.
# Structure:
#   RAG_INDEX_STORE[session_id] = {
#       "chunks": [
#           {
#              "text": str,
#              "filename": str,
#              "source_type": "file_text" | "json_kv" | "xlsx_row" | "xlsx_row_chunk",
#              "key": Optional[str],  # present for json_kv or encoded metadata for xlsx_row_chunk
#              # Optional metadata for xlsx_row_chunk:
#              # "row_id", "chunk_id", "sheet_name", "row_number", "chunk_index", "chunk_count"
#           }, ...
#       ],
#       "embeddings": [ [float, ...] | None, ...],
#       "embedding_model": str
#   }
RAG_INDEX_STORE: Dict[str, Dict[str, Any]] = {}

app = FastAPI(
    title="IntelliQuery Chatbot API",
    version="1.0.0",
    description="FastAPI backend for the IntelliQuery chatbot, providing chat endpoints using user-uploaded context files and Google Gemini for answer generation."
)

openapi_tags = [
    {"name": "Health", "description": "Health check endpoint."},
    {"name": "Chat", "description": "Endpoints for chat, history/title generation, response retrieval, and context management."},
    {"name": "UserAuth", "description": "Endpoints for user registration and login."}
]

# Register chat title router (Gemini-powered title generator)
app.include_router(chat_title_router)

# Allow CORS from everywhere for demo/dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    """Schema for a chat request."""
    session_id: str = Field(..., description="Unique session identifier for the conversation context.")
    query: str = Field(..., description="User's query (question or message).")


class ChatAnswerResponse(BaseModel):
    """Schema for the chat response that returns only the final Gemini answer."""
    answer: str = Field(..., description="Final natural language answer returned by Gemini.")


# --- AUTH SCHEMAS FOR REGISTRATION & LOGIN ---

class RegisterRequest(BaseModel):
    """Pydantic schema for user registration request."""
    username: str = Field(..., min_length=3, max_length=50, description="Unique username for registration")
    email: EmailStr = Field(..., description="User's email address")
    password: str = Field(..., min_length=6, max_length=128, description="User's password (plaintext, will be hashed)")


class RegisterResponse(BaseModel):
    """Schema for success/failure of registration."""
    id: int
    username: str
    email: EmailStr


class LoginRequest(BaseModel):
    """Pydantic schema for user login request."""
    username: str = Field(..., description="Registered username")
    password: str = Field(..., description="Password")


class LoginResponse(BaseModel):
    """Schema for login response."""
    id: int
    username: str
    email: EmailStr


class UploadedFileResult(BaseModel):
    """Schema describing the processing result for a single uploaded file."""
    filename: str = Field(..., description="Original file name")
    size: int = Field(..., description="File size in bytes")
    content_chars: int = Field(..., description="Number of characters extracted from the file")
    preview: str = Field(..., description="A short preview/summary of extracted content")
    error: Optional[str] = Field(default=None, description="Error message if processing failed")


class UploadContextResponse(BaseModel):
    """Schema for the response from the context upload endpoint."""
    session_id: str = Field(..., description="Session ID associated with the uploaded context")
    files_processed: List[UploadedFileResult] = Field(..., description="Per-file processing results")
    total_chars: int = Field(..., description="Total number of characters added to session context")
    message: str = Field(..., description="Status message/acknowledgment")


def _clean_gemini_output(text: str) -> str:
    """
    Removes leading/trailing meta, KB source, or disclaimer information from Gemini output.
    Ensures only direct answers are delivered to the user.
    """
    import re
    META_PATTERNS = [
        # Remove variants at start of the answer
        r"(?i)^ *(?:based on (?:the )?(?:provided )?(?:knowledge ?base|context|sources)[^:]*:?)",
        r"(?i)^(?:as an?\s+[^\s:,]+ [^\s:,]+,?)[\s:.-]*",
        r"(?i)^ *(?:this information[^:]*:?)",
        r"(?i)^ *(?:note:)[^\n\r]*",
        r"(?i)^ *(?:please note)[^\n\r]*",
        r"(?i)^ *(?:source[sd]?:)[^\n\r]*",
        r"(?i)^ *(?:from the knowledge base[^:]*:?)",
        r"(?i)^ *(?:provided context[^:]*:?)",
        r"(?i)^\(?(?:based on|as an ai language model|this information|provided context)[^\)]*\)?",
    ]
    # Remove trailing variants
    TRAIL_PATTERNS = [
        r"(?i)\(? *(?:based on (?:the )?(?:provided )?(?:knowledge ?base|context|sources)[^)]*)\)?[.!]? *$",
        r"(?i)\(? *(?:from the knowledge base)[^)]*\)?[.!]? *$",
        r"(?i)\(? *(?:provided context)[^)]*\)?[.!]? *$",
    ]
    clean_text = text
    for pat in META_PATTERNS:
        clean_text = re.sub(pat, "", clean_text, flags=re.IGNORECASE | re.MULTILINE)
    for pat in TRAIL_PATTERNS:
        clean_text = re.sub(pat, "", clean_text, flags=re.IGNORECASE | re.MULTILINE)
    clean_text = clean_text.strip()
    return clean_text


# --- CONTEXT + RAG UTILITIES ---

def _tokenize(s: str) -> List[str]:
    import re
    return [t for t in re.findall(r"[A-Za-z0-9]+", (s or "").lower()) if t]


def _simple_similarity(a: str, b: str) -> float:
    """
    Simple similarity score based on token overlap (resource-light fallback).
    """
    set_a, set_b = set(_tokenize(a)), set(_tokenize(b))
    if not set_a or not set_b:
        return 0.0
    shared = set_a.intersection(set_b)
    return len(shared) / max(len(set_a), len(set_b))


def _split_into_chunks(text: str, chunk_size_words: int = 180, overlap_words: int = 40) -> List[str]:
    """
    Split text into overlapping chunks to improve retrieval granularity.

    Args:
        text: The input full text.
        chunk_size_words: Approximate number of words per chunk.
        overlap_words: Overlap between consecutive chunks to preserve context.

    Returns:
        List[str]: Chunked text segments.
    """
    if not text:
        return []
    words = text.split()
    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = min(len(words), start + chunk_size_words)
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        if end == len(words):
            break
        start = max(end - overlap_words, start + 1)
    return chunks


def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """
    Compute cosine similarity between two vectors. Zero-safe.
    """
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    import math
    dot = sum(x * y for x, y in zip(vec_a, vec_b))
    a_norm = math.sqrt(sum(x * x for x in vec_a))
    b_norm = math.sqrt(sum(y * y for y in vec_b))
    if a_norm == 0 or b_norm == 0:
        return 0.0
    return dot / (a_norm * b_norm)


def _get_embedding_model_name() -> str:
    """
    Resolve the embedding model name to use with Google Generative AI.
    """
    # text-embedding-004 is the current recommended Gemini embedding model.
    return "models/text-embedding-004"


def _embed_one(text: str) -> Optional[List[float]]:
    """
    Get embedding for a single text using Gemini embeddings.
    Returns None if embedding fails (e.g., missing key).
    """
    key = get_gemini_api_key()
    if not key:
        return None
    try:
        genai.configure(api_key=key)
        model = _get_embedding_model_name()
        # google-generativeai embed_content returns dict with "embedding": {"values": [...]}
        result = genai.embed_content(model=model, content=text)
        vec = result.get("embedding", {}).get("values")
        if isinstance(vec, list) and vec and isinstance(vec[0], (int, float)):
            return [float(v) for v in vec]
        return None
    except Exception:
        # Silently degrade to None to allow lexical fallback
        return None


def _embed_texts(texts: List[str]) -> List[Optional[List[float]]]:
    """
    Embed a batch of texts using Gemini embeddings with parallelization.

    Uses a ThreadPoolExecutor to speed up embedding IO calls. Concurrency can be tuned via
    CHATBOT_EMBED_MAX_WORKERS env var (default: 8). If embedding fails or key missing,
    returns list of None entries.
    """
    import os
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if not texts:
        return []

    try:
        max_workers = int(os.getenv("CHATBOT_EMBED_MAX_WORKERS", "8"))
    except Exception:
        max_workers = 8
    max_workers = max(1, max_workers)

    results: List[Optional[List[float]]] = [None] * len(texts)

    def _task(i: int, s: str):
        vec = _embed_one(s)
        results[i] = vec

    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="embed") as ex:
        futures = []
        for i, s in enumerate(texts):
            futures.append(ex.submit(_task, i, s))
        for _f in as_completed(futures):
            pass
    return results


def _ensure_session_index(session_id: str):
    """
    Ensure RAG index structure exists for the session.
    """
    if session_id not in RAG_INDEX_STORE:
        RAG_INDEX_STORE[session_id] = {
            "chunks": [],
            "embeddings": [],
            "embedding_model": _get_embedding_model_name(),
        }


def _index_text_for_session(session_id: str, filename: str, text: str):
    """
    Chunk, embed, and store vectors and their source chunks for a session.
    Falls back to storing chunks without embeddings if embeddings are unavailable.
    Processes in batches to reduce memory footprint for large documents.
    """
    _ensure_session_index(session_id)
    chunks = _split_into_chunks(text, chunk_size_words=180, overlap_words=40)
    if not chunks:
        return

    store = RAG_INDEX_STORE[session_id]
    BATCH = 256
    for start in range(0, len(chunks), BATCH):
        sub = chunks[start:start + BATCH]
        vectors = _embed_texts(sub)

        items_for_db = []
        for chunk, vec in zip(sub, vectors):
            store["chunks"].append({"text": chunk, "filename": filename, "source_type": "file_text"})
            store["embeddings"].append(vec)
            items_for_db.append(
                {"text": chunk, "filename": filename, "source_type": "file_text", "key": None}
            )

        # Persist to vector DB (best-effort; ignore failures)
        try:
            vector_store.add_embeddings(
                session_id=session_id,
                items=items_for_db,
                vectors=vectors,
                model=_get_embedding_model_name(),
            )
        except Exception:
            pass


def _index_json_kv_pairs_for_session(session_id: str, filename: str, pairs: List[Tuple[str, str]]):
    """
    Index flattened JSON key-value pairs for a session. Each (key, value) becomes a standalone chunk.

    Args:
        session_id: Session ID.
        filename: Source JSON filename.
        pairs: List of (dotted_key, value) pairs.
    """
    _ensure_session_index(session_id)
    # Filter pairs to those with a non-empty key and render "key: value"
    filtered_pairs = [(k, v) for (k, v) in pairs if k]
    if not filtered_pairs:
        return

    store = RAG_INDEX_STORE[session_id]
    BATCH = 512
    for start in range(0, len(filtered_pairs), BATCH):
        sub_pairs = filtered_pairs[start:start + BATCH]
        texts = [f"{k}: {v}" for k, v in sub_pairs]
        vectors = _embed_texts(texts)
        items_for_db = []
        for (k, v), chunk_text, vec in zip(sub_pairs, texts, vectors):
            store["chunks"].append({
                "text": chunk_text,
                "filename": filename,
                "source_type": "json_kv",
                "key": k
            })
            store["embeddings"].append(vec)
            items_for_db.append(
                {"text": chunk_text, "filename": filename, "source_type": "json_kv", "key": k}
            )

        # Persist to vector DB (best-effort; ignore failures)
        try:
            vector_store.add_embeddings(
                session_id=session_id,
                items=items_for_db,
                vectors=vectors,
                model=_get_embedding_model_name(),
            )
        except Exception:
            pass


def _index_xlsx_documents_for_session(session_id: str, filename: str, docs: List[str]):
    """
    Index per-row XLSX documents (already formatted natural-language strings).

    Args:
        session_id (str): Session ID.
        filename (str): Source XLSX filename.
        docs (List[str]): List of natural-language row documents.
    """
    _ensure_session_index(session_id)
    if not docs:
        return
    vectors = _embed_texts(docs)
    store = RAG_INDEX_STORE[session_id]
    items_for_db = []
    for doc_text, vec in zip(docs, vectors):
        store["chunks"].append({
            "text": doc_text,
            "filename": filename,
            "source_type": "xlsx_row",
        })
        store["embeddings"].append(vec)
        items_for_db.append(
            {"text": doc_text, "filename": filename, "source_type": "xlsx_row", "key": None}
        )

    # Persist to vector DB (best-effort; ignore failures)
    try:
        vector_store.add_embeddings(
            session_id=session_id,
            items=items_for_db,
            vectors=vectors,
            model=_get_embedding_model_name(),
        )
    except Exception:
        pass


def _index_xlsx_row_chunks_for_session(session_id: str, filename: str, row_chunks: List[Dict[str, Any]]):
    """
    Index XLSX row-chunks with metadata, in batches to control memory use and latencies.

    Each row_chunk dict must have at least the keys specified in the docstring.
    """
    _ensure_session_index(session_id)
    if not row_chunks:
        return

    store = RAG_INDEX_STORE[session_id]
    BATCH = 512
    for start in range(0, len(row_chunks), BATCH):
        sub = row_chunks[start:start + BATCH]
        texts = [rc.get("text", "") for rc in sub]
        vectors = _embed_texts(texts)
        items_for_db = []

        for rc, vec in zip(sub, vectors):
            # In-memory item includes full metadata for direct retrieval
            mem_item = {
                "text": rc.get("text", ""),
                "filename": filename,
                "source_type": "xlsx_row_chunk",
                "key": rc.get("chunk_id"),  # raw key for quick identification
                "row_id": rc.get("row_id"),
                "chunk_id": rc.get("chunk_id"),
                "sheet_name": rc.get("sheet_name"),
                "row_number": rc.get("row_number"),
                "chunk_index": rc.get("chunk_index"),
                "chunk_count": rc.get("chunk_count"),
            }
            store["chunks"].append(mem_item)
            store["embeddings"].append(vec)

            # Persisted 'key' stores JSON-encoded metadata for later reconstruction
            meta = {
                "row_id": rc.get("row_id"),
                "chunk_id": rc.get("chunk_id"),
                "sheet_name": rc.get("sheet_name"),
                "row_number": rc.get("row_number"),
                "chunk_index": rc.get("chunk_index"),
                "chunk_count": rc.get("chunk_count"),
            }
            items_for_db.append(
                {
                    "text": rc.get("text", ""),
                    "filename": filename,
                    "source_type": "xlsx_row_chunk",
                    "key": json.dumps(meta, ensure_ascii=False),
                }
            )

        # Persist to vector DB (best-effort; ignore failures)
        try:
            vector_store.add_embeddings(
                session_id=session_id,
                items=items_for_db,
                vectors=vectors,
                model=_get_embedding_model_name(),
            )
        except Exception:
            pass


def _vector_search(session_id: str, query: str, top_k: int = 3) -> List[str]:
    """
    Perform semantic vector search for the top_k relevant chunks using cosine similarity
    against Gemini embeddings. If embeddings are not available, falls back to lexical similarity.

    Returns:
        List[str]: The text of the top-k retrieved chunks.
    """
    items = _vector_search_items(session_id, query, top_k=top_k)
    return [it["text"] for it in items]


def _vector_search_items(
    session_id: str,
    query: str,
    top_k: int = 6,
    restrict_source_types: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieve the top-k items (with metadata) relevant to the user query.

    This function now combines:
      - In-memory session index items (built from current process uploads)
      - Persisted items from the vector database (e.g., per-row Excel row-chunks)

    Optionally restricts candidate items by source_type via `restrict_source_types`.

    Returns:
        List[Dict[str, Any]]: Items with fields {text, filename, source_type, key?, metadata?}
    """
    # Prepare query vector (if possible); otherwise lexical fallback will be used.
    query_vec = _embed_one(query)

    combined_items: List[Tuple[Dict[str, Any], Optional[List[float]]]] = []

    # 1) In-memory items
    index = RAG_INDEX_STORE.get(session_id)
    if index and index.get("chunks"):
        for item, vec in zip(index["chunks"], index["embeddings"]):
            if restrict_source_types and item.get("source_type") not in restrict_source_types:
                continue
            combined_items.append((item, vec))

    # 2) Persisted items from DB
    try:
        db_items = vector_store.get_session_embedding_items(
            session_id=session_id,
            source_types=restrict_source_types or None,
            max_records=5000,
        )
        # Deduplicate against in-memory by (text, source_type)
        seen = set((itm["text"], itm.get("source_type")) for itm, _ in combined_items)
        for dbi in db_items:
            key = (dbi.get("text") or "", dbi.get("source_type"))
            if key in seen:
                continue

            item_dict = {
                "text": dbi.get("text") or "",
                "filename": dbi.get("filename"),
                "source_type": dbi.get("source_type") or "file_text",
                "key": dbi.get("key"),
            }

            # Attempt to parse JSON metadata stored in 'key' (for xlsx_row_chunk)
            meta = None
            key_val = dbi.get("key")
            if isinstance(key_val, str):
                try:
                    meta = json.loads(key_val)
                except Exception:
                    meta = None

            if isinstance(meta, dict):
                # Promote metadata to top-level fields for downstream use
                item_dict.update({
                    "row_id": meta.get("row_id"),
                    "chunk_id": meta.get("chunk_id"),
                    "sheet_name": meta.get("sheet_name"),
                    "row_number": meta.get("row_number"),
                    "chunk_index": meta.get("chunk_index"),
                    "chunk_count": meta.get("chunk_count"),
                })

            vec = dbi.get("embedding") if isinstance(dbi.get("embedding"), list) else None
            combined_items.append((item_dict, vec))
            seen.add(key)
    except Exception:
        # If DB is unavailable or an error occurs, continue with in-memory results only.
        pass

    if not combined_items:
        return []

    # Score and rank
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for item, vec in combined_items:
        if query_vec and vec:
            score = _cosine_similarity(query_vec, vec)
        else:
            score = _simple_similarity(query, item.get("text", ""))

        # Boosts for structured sources
        st = item.get("source_type")
        if st == "json_kv":
            score *= 1.2  # 20% boost for JSON facts
        elif st in ("xlsx_row", "xlsx_row_chunk"):
            score *= 1.1  # 10% boost for Excel row-derived documents

        scored.append((score, item))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in scored[:top_k]]


# PUBLIC_INTERFACE
def get_gemini_response(
    query: str,
    memory: ConversationBufferMemory,
    row_docs: List[str],
) -> str:
    """
    PUBLIC_INTERFACE
    Generate an answer using Google Gemini API strictly based on the provided Excel row
    documents retrieved from the vector database.

    The model must NOT use external knowledge or assumptions; if the rows do not provide
    sufficient information to answer, it should respond:
        "I don't have sufficient information in the provided Excel rows to answer that."

    Args:
        query (str): User query.
        memory (ConversationBufferMemory): Conversation memory buffer. This is used only
            for conversational phrasing/continuity, not as a source of facts.
        row_docs (List[str]): The retrieved Excel row documents to use as the sole knowledge
            context. Each string should already include sheet/row info when available.

    Returns:
        str: Model answer derived only from the provided row documents.
    """
    mem_str = memory.buffer_as_str if hasattr(memory, "buffer_as_str") else ""
    # Limit number and size of row docs to keep prompt manageable
    MAX_ROWS = 10
    MAX_CONTEXT_CHARS = 12000

    selected_rows = (row_docs or [])[:MAX_ROWS]
    rows_block = "\n".join(f"- {doc}" for doc in selected_rows).strip()
    if len(rows_block) > MAX_CONTEXT_CHARS:
        rows_block = rows_block[:MAX_CONTEXT_CHARS]

    instruction = (
        "You are an assistant that answers questions using ONLY the following Excel row documents.\n"
        "Do not use any outside knowledge or assumptions. If the rows do not contain the answer,\n"
        "reply exactly: \"I don't have sufficient information in the provided Excel rows to answer that.\""
    )

    prompt = (
        f"{instruction}\n\n"
        f"Excel row documents:\n{rows_block if rows_block else '(none)'}\n\n"
        f"Conversation history (do not use as a source of facts):\n{mem_str}\n\n"
        f"User's question:\n{query}\n\n"
        "Now answer strictly using only the content of the Excel row documents. If needed, quote the relevant\n"
        "values from the rows. If the documents are insufficient, use the required fallback sentence exactly."
    )

    gemini_api_key = get_gemini_api_key()
    if not gemini_api_key:
        raise HTTPException(status_code=500, detail="Gemini API key is not set in environment variables.")
    try:
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content([{"role": "user", "parts": [prompt]}])
        raw_answer = (response.text or "").strip()
        return _clean_gemini_output(raw_answer)
    except Exception as e:
        # Fallback to minimal message if Gemini API fails
        return _clean_gemini_output(f"[Gemini enhancement unavailable: {e}]")


def _combine_row_chunks_text(chunks: List[Dict[str, Any]]) -> str:
    """
    Combine multiple XLSX row-chunk texts into a single row document string.

    Attempts to strip per-chunk headers like:
        "Sheet <sheet> | Row <n> | Chunk i/n | "
    and consolidates the remaining key-value parts. Falls back to joining the full
    chunk texts if header stripping fails.
    """
    if not chunks:
        return ""
    # Sort by chunk_index if present
    try:
        chunks_sorted = sorted(chunks, key=lambda c: (c.get("chunk_index") if c.get("chunk_index") is not None else 0))
    except Exception:
        chunks_sorted = chunks

    first = chunks_sorted[0]
    sheet_name = first.get("sheet_name") or "Sheet"
    row_number = first.get("row_number")
    header_base = f"Sheet {sheet_name}"
    if row_number is not None:
        header_base += f" | Row {row_number}"

    import re
    header_pat = re.compile(r"^Sheet [^|]+ \| Row \d+ \| Chunk \d+/\d+ \| ?", flags=re.IGNORECASE)

    parts: List[str] = []
    for ch in chunks_sorted:
        text = ch.get("text") or ""
        stripped = header_pat.sub("", text).strip()
        parts.append(stripped or text)

    combined_pairs = " | ".join([p for p in parts if p]).strip()
    if combined_pairs:
        return f"{header_base} | {combined_pairs}"
    # Fallback: join original texts if nothing matched
    return " ".join([ch.get("text") or "" for ch in chunks_sorted]).strip()


# Ensure user table exists on startup
create_tables()

@app.get("/", tags=["Health"])
def health_check():
    """
    PUBLIC_INTERFACE
    API health check endpoint.

    Returns:
        dict: Status message.
    """
    return {"message": "Healthy"}


@app.post(
    "/chat",
    response_model=ChatAnswerResponse,
    tags=["Chat"],
    summary="Chat with bot",
    description="Submit a chat query. Backend retrieves only top-k relevant segments via vector search from uploaded files (if any) and passes them to Gemini. Returns only Gemini's final answer."
)
def chat(request: ChatRequest):
    """
    PUBLIC_INTERFACE
    Handles user's chat request. All answers come directly from Gemini.
    If the user has uploaded files for this session, the most relevant snippets from those files are retrieved
    via semantic vector search and provided as additional context to Gemini. The API returns only Gemini's final answer.

    Updated retrieval behavior:
    - Uses ChromaDB to find top relevant XLSX row chunks.
    - Aggregates ALL chunks belonging to the same Excel row to reconstruct the full row document.
    - Passes the reconstructed row documents as the only context to Gemini for answering.
    """
    import traceback

    try:
        session_id = request.session_id
        if not session_id or not isinstance(session_id, str):
            raise HTTPException(status_code=400, detail="session_id must be a non-empty string.")

        if session_id not in CONVERSATION_MEMORY:
            CONVERSATION_MEMORY[session_id] = ConversationBufferMemory(
                return_messages=True,
                output_key="output"
            )
        memory = CONVERSATION_MEMORY[session_id]

        def _safe_str_output(val, fallback="No answer available"):
            if val is None or (isinstance(val, str) and val.strip() == ""):
                return fallback
            return str(val)

        # Log user input with a temp placeholder output
        try:
            memory.save_context({"input": request.query}, {"output": _safe_str_output("", "No answer available")})
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to save user context: {e}")

        # Retrieve top-k relevant Excel row-chunks and then aggregate all chunks for those rows
        top_items = _vector_search_items(
            session_id,
            request.query,
            top_k=8,
            restrict_source_types=["xlsx_row_chunk", "xlsx_row"],  # prefer new chunked rows, keep legacy compatibility
        )

        # Build row documents by aggregating chunks from the same row
        row_docs: List[str] = []
        seen_row_ids: set = set()

        # Helper to pull in-memory chunks for a row (fallback if Chroma returns nothing)
        def _gather_memory_row_chunks(sid: str, rid: str) -> List[Dict[str, Any]]:
            out: List[Dict[str, Any]] = []
            idx = RAG_INDEX_STORE.get(sid)
            if not idx:
                return out
            for it in idx.get("chunks", []):
                if it.get("source_type") == "xlsx_row_chunk" and it.get("row_id") == rid:
                    out.append(it)
            try:
                out.sort(key=lambda c: (c.get("chunk_index") if c.get("chunk_index") is not None else 0))
            except Exception:
                pass
            return out

        for it in top_items:
            if len(row_docs) >= 6:
                break
            st = it.get("source_type")

            if st == "xlsx_row":
                # Legacy per-row document already complete
                txt = it.get("text") or ""
                if txt:
                    row_docs.append(txt)
                continue

            if st == "xlsx_row_chunk":
                rid = it.get("row_id")
                if not rid or rid in seen_row_ids:
                    continue
                seen_row_ids.add(rid)

                # Fetch all chunks for this row from Chroma; fallback to in-memory if needed
                all_chunks: List[Dict[str, Any]] = []
                try:
                    all_chunks = vector_store.get_row_chunks(session_id=session_id, row_id=rid, max_chunks=None)
                except Exception:
                    all_chunks = []

                if not all_chunks:
                    all_chunks = _gather_memory_row_chunks(session_id, rid)

                # If still empty, at least include the selected chunk as-is
                if not all_chunks:
                    all_chunks = [it]

                combined_text = _combine_row_chunks_text(all_chunks)
                if combined_text:
                    row_docs.append(combined_text)

        if not row_docs:
            gemini_answer = "I don't have sufficient information in the provided Excel rows to answer that."
        else:
            try:
                gemini_answer = get_gemini_response(request.query, memory, row_docs=row_docs[:6])
            except Exception as e:
                gemini_answer = "[Gemini unavailable: {}]".format(e)

        # Final output cleaning and save in memory
        gemini_answer = _clean_gemini_output(_safe_str_output(gemini_answer))
        try:
            memory.save_context({"input": request.query}, {"output": _safe_str_output(gemini_answer)})
        except Exception:
            pass  # Do not raise for failed memory update

        return ChatAnswerResponse(answer=gemini_answer)

    except HTTPException:
        raise  # Allow FastAPI HTTPExceptions to propagate
    except Exception as e:
        tb = traceback.format_exc()
        print(f"Internal Server Error in /chat endpoint: {e}\nTraceback:\n{tb}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")


# PUBLIC_INTERFACE
@app.post(
    "/register",
    response_model=RegisterResponse,
    tags=["UserAuth"],
    summary="Register a new user",
    description="Endpoint for user registration using username, email, and password. Returns user details on success. Username and email must be unique.",
    responses={
        409: {"description": "Username/email already registered"},
        400: {"description": "Validation error"},
    },
)
def register(request: RegisterRequest, db=Depends(get_db)):
    """
    PUBLIC_INTERFACE
    Registers a new user, stores credentials securely in the database.

    Args:
        request (RegisterRequest): The registration data (username, email, password).
        db (Session): SQLAlchemy Session dependency.

    Returns:
        RegisterResponse: id, username, and email of the registered user.

    Raises:
        HTTPException: If username or email already exists, or validation fails.
    """
    try:
        user = create_user(db, username=request.username, email=request.email, password=request.password)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return RegisterResponse(id=user.id, username=user.username, email=user.email)


# PUBLIC_INTERFACE
@app.post(
    "/login",
    response_model=LoginResponse,
    tags=["UserAuth"],
    summary="User login",
    description="Endpoint for users to login with username and password.",
    responses={
        401: {"description": "Incorrect username or password"},
        400: {"description": "Validation error"},
    },
)
def login(request: LoginRequest, db=Depends(get_db)):
    """
    PUBLIC_INTERFACE
    Authenticate a user's credentials and return basic user info.

    Args:
        request (LoginRequest): Login data (username, password).
        db (Session): SQLAlchemy Session dependency.

    Returns:
        LoginResponse: id, username, and email of the authenticated user.

    Raises:
        HTTPException: If credentials are invalid.
    """
    user = get_user_by_username(db, request.username)
    if not user or not verify_password(request.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    return LoginResponse(id=user.id, username=user.username, email=user.email)


# Add endpoint doc for WebSocket and real-time (optional, can expand later)
@app.get("/chat/wsinfo", tags=["Chat"], summary="WebSocket usage info", description="Info about WebSocket/API support for real-time chat.")
def chat_wsinfo():
    """
    PUBLIC_INTERFACE
    Returns information about real-time chat support (WebSocket or usual polling).
    """
    return {"detail": "Current version supports REST API chat only. Real-time WebSocket may be added in future versions."}


# --- FILE UPLOAD ENDPOINTS FOR CONTEXT ---

# PUBLIC_INTERFACE
# Models for async job acceptance and status
class UploadJobAccepted(BaseModel):
    """Acknowledgement for async upload processing."""
    job_id: str = Field(..., description="Background job identifier")
    status_url: str = Field(..., description="URL to poll job status")
    message: str = Field(..., description="Acknowledgement message")


class UploadJobStatus(BaseModel):
    """Status for an async upload processing job."""
    job_id: str = Field(..., description="Background job identifier")
    status: str = Field(..., description="Job status: pending|running|succeeded|failed|canceled")
    progress: int = Field(..., ge=0, le=100, description="Progress percentage")
    message: str = Field(..., description="Status message")
    result: Optional[UploadContextResponse] = Field(default=None, description="Final result if completed")


# PUBLIC_INTERFACE
@app.post(
    "/chat/upload-context",
    response_model=UploadContextResponse,
    tags=["Chat"],
    summary="Upload context files for a chat session",
    description=(
        "Synchronous processing: extracts readable text from supported files and indexes them for retrieval. "
        "For very large uploads, prefer the async endpoint to avoid timeouts."
    ),
    responses={
        400: {"description": "Validation error or no files provided"},
        415: {"description": "Unsupported media type"},
    },
)
def upload_chat_context(
    session_id: str = Form(..., description="Session ID to associate uploaded context with"),
    files: List[UploadFile] = File(..., description="One or more files (.docx, .xlsx, .pdf, .txt, .json)"),
):
    """
    PUBLIC_INTERFACE
    Synchronous upload and processing. Suitable for small/medium files. For large/wide Excel files,
    use /chat/upload-context/async to prevent timeouts.
    """
    from .processing import process_files

    # Read file bytes upfront
    files_data: List[Tuple[str, bytes]] = []
    for f in files:
        filename = f.filename or "unnamed"
        try:
            data = f.file.read()
        except Exception:
            data = b""
        files_data.append((filename, data))

    result = process_files(session_id=session_id, files_data=files_data, progress_callback=None)
    # Convert dict to UploadContextResponse model
    return UploadContextResponse(
        session_id=result["session_id"],
        files_processed=[UploadedFileResult(**r) for r in result["files_processed"]],
        total_chars=result["total_chars"],
        message=result["message"],
    )


# PUBLIC_INTERFACE
@app.post(
    "/chat/upload-context/async",
    response_model=UploadJobAccepted,
    tags=["Chat"],
    summary="Upload context files (async)",
    description=(
        "Queues files for background processing (chunking, embeddings, ChromaDB storage) to avoid timeouts. "
        "Returns a job_id to poll status."
    ),
)
def upload_chat_context_async(
    session_id: str = Form(..., description="Session ID to associate uploaded context with"),
    files: List[UploadFile] = File(..., description="One or more files (.docx, .xlsx, .pdf, .txt, .json)"),
):
    """
    PUBLIC_INTERFACE
    Accepts files and schedules background processing to prevent gateway timeouts on large uploads.
    """
    from .background_jobs import get_job_manager

    if not session_id or not isinstance(session_id, str):
        raise HTTPException(status_code=400, detail="session_id must be provided as a non-empty string.")
    if not files:
        raise HTTPException(status_code=400, detail="At least one file must be provided.")

    # Read file bytes upfront (UploadFile cannot be used outside request context)
    files_data: List[Tuple[str, bytes]] = []
    for f in files:
        filename = f.filename or "unnamed"
        try:
            data = f.file.read()
        except Exception:
            data = b""
        files_data.append((filename, data))

    jm = get_job_manager()
    job_id = jm.submit_upload_job(session_id=session_id, files_data=files_data)

    # Construct status URL (best-effort)
    status_url = f"/chat/upload-context/jobs/{job_id}"
    return UploadJobAccepted(job_id=job_id, status_url=status_url, message="Accepted for background processing")


# PUBLIC_INTERFACE
@app.get(
    "/chat/upload-context/jobs/{job_id}",
    response_model=UploadJobStatus,
    tags=["Chat"],
    summary="Upload job status",
    description="Poll the status of a background upload processing job.",
)
def get_upload_job_status(job_id: str):
    """
    PUBLIC_INTERFACE
    Return the current status/progress of a queued/running upload job, and the final
    result if completed.
    """
    from .background_jobs import get_job_manager
    jm = get_job_manager()
    info = jm.get_job(job_id)
    if not info:
        raise HTTPException(status_code=404, detail="Job not found")

    # Coerce result to model if available
    result_model = None
    if info.get("result"):
        r = info["result"]
        result_model = UploadContextResponse(
            session_id=r["session_id"],
            files_processed=[UploadedFileResult(**x) for x in r["files_processed"]],
            total_chars=r["total_chars"],
            message=r["message"],
        )

    return UploadJobStatus(
        job_id=job_id,
        status=info.get("status", "unknown"),
        progress=int(info.get("progress", 0)),
        message=info.get("message", ""),
        result=result_model,
    )
