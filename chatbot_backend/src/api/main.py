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

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form, BackgroundTasks, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field, EmailStr
from typing import List, Optional, Dict, Any
import json
import google.generativeai as genai

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

# Load environment variables
load_dotenv()

# Memory store for chat contexts (keyed by session_id).
CONVERSATION_MEMORY: Dict[str, ConversationBufferMemory] = {}

# Per-session uploaded context store.
# Structure:
# {
#   session_id: {
#       "files": [ {filename, size, chars, preview, error?} ],
#       "combined": str,                          # legacy textual context
#       "xlsx_json_rows": [ {row}, {row}, ... ],  # primary JSON rows derived from Excel
#       "last_query": str,
#       "last_top_chunks_count": int,
#       "last_used_table_snippets": bool,
#       "last_context_chars": int,
#       "last_retrieved_context": str,
#   }
# }
CONTEXT_STORE: Dict[str, Dict[str, Any]] = {}

# Per-session in-memory vector index for RAG.
# Structure:
#   RAG_INDEX_STORE[session_id] = {
#       "chunks": [ {"text": str, "filename": str} , ...],
#       "embeddings": [ [float, ...], ...],
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


def _chunk_xlsx_text_special(text: str, max_chunk_chars: int = 2600) -> List[str]:
    """
    Preserve Excel sheet blocks and table previews when chunking, to keep row/column context intact.

    Strategy:
      - Split by blank lines to get sheet/section blocks.
      - If a block is larger than `max_chunk_chars`, split by lines without breaking TSV header+rows adjacency.
      - Favor keeping sections that include 'Columns', 'JSON sample', and 'TSV preview' together.

    This greatly improves retrieval accuracy for table questions.
    """
    if not text:
        return []
    blocks: List[str] = []
    current: List[str] = []
    lines = text.splitlines()
    def flush_block():
        if current:
            blocks.append("\n".join(current).strip())
            current.clear()
    for line in lines:
        if line.strip() == "":
            flush_block()
            continue
        current.append(line)
    flush_block()

    chunks: List[str] = []
    for block in blocks:
        if len(block) <= max_chunk_chars:
            chunks.append(block)
            continue
        # Split large block by lines
        blines = block.splitlines()
        buf: List[str] = []
        for ln in blines:
            # avoid splitting TSV header from the immediate next rows by keeping small groups together
            buf.append(ln)
            if sum(len(x) + 1 for x in buf) >= max_chunk_chars:
                chunks.append("\n".join(buf).strip())
                buf = []
        if buf:
            chunks.append("\n".join(buf).strip())
    # Return only non-empty chunks
    return [c for c in chunks if c.strip()]


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
    Embed a batch of texts (serially) using Gemini embeddings.
    If embedding fails or key missing, returns list of None entries.
    """
    vectors: List[Optional[List[float]]] = []
    for t in texts:
        vectors.append(_embed_one(t))
    return vectors


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

    For .xlsx files, use special chunking to preserve table blocks and TSV/JSON previews.
    """
    _ensure_session_index(session_id)
    # Prefer sheet/TSV-preserving chunking for Excel/JSON structured summaries
    lower_name = (filename or "").lower()
    if lower_name.endswith(".xlsx") or lower_name.endswith(".json"):
        chunks = _chunk_xlsx_text_special(text, max_chunk_chars=2600)
    else:
        chunks = _split_into_chunks(text, chunk_size_words=180, overlap_words=40)

    if not chunks:
        return
    vectors = _embed_texts(chunks)

    store = RAG_INDEX_STORE[session_id]
    for chunk, vec in zip(chunks, vectors):
        store["chunks"].append({"text": chunk, "filename": filename})
        store["embeddings"].append(vec)  # vec could be None; retrieval handles fallback


def _index_structured_chunks_for_session(session_id: str, chunks: List[Dict[str, Any]]):
    """
    Index pre-built structured chunks with metadata.

    Each item in 'chunks' should be of the form:
      {"text": "<chunk text>", "meta": {...}}

    This function embeds per chunk text and stores meta for downstream retrieval.
    """
    _ensure_session_index(session_id)
    texts: List[str] = []
    metas: List[Dict[str, Any]] = []
    for ch in (chunks or []):
        t = (ch.get("text") or "").strip()
        if not t:
            continue
        texts.append(t)
        metas.append(ch.get("meta") or {})

    if not texts:
        return

    vectors = _embed_texts(texts)
    store = RAG_INDEX_STORE[session_id]
    for t, vec, meta in zip(texts, vectors, metas):
        store["chunks"].append({"text": t, "filename": meta.get("filename") or "", "meta": meta})
        store["embeddings"].append(vec)


def _vector_search(session_id: str, query: str, top_k: int = 3) -> List[str]:
    """
    Perform semantic vector search for the top_k relevant chunks using cosine similarity
    against Gemini embeddings. If embeddings are not available, falls back to lexical similarity.

    Returns:
        List[str]: The text of the top-k retrieved chunks.
    """
    index = RAG_INDEX_STORE.get(session_id)
    if not index or not index.get("chunks"):
        return []

    # Try vector search
    query_vec = _embed_one(query)
    if query_vec:
        scored = []
        schema_list_query = _is_schema_or_list_query(query)
        for item, vec in zip(index["chunks"], index["embeddings"]):
            if vec:
                score = _cosine_similarity(query_vec, vec)
            else:
                # If a particular chunk lacks embedding, degrade to lexical fallback for that item
                score = _simple_similarity(query, item["text"])
            # Boost schema/field chunks when query appears schema-oriented
            if schema_list_query:
                m = item.get("meta") or {}
                if m.get("type") in {"json_field", "json_schema"}:
                    score *= 1.15
            scored.append((score, item["text"]))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [text for _, text in scored[:top_k]]

    # Fallback: lexical similarity if no query embedding
    scored_lex = [(_simple_similarity(query, item["text"]), item["text"]) for item in index["chunks"]]
    scored_lex.sort(key=lambda x: x[0], reverse=True)
    return [text for _, text in scored_lex[:top_k]]


def _is_debug_enabled() -> bool:
    """
    Check env var CHATBOT_DEBUG_CONTEXT to enable verbose context logging.
    """
    import os
    return os.getenv("CHATBOT_DEBUG_CONTEXT", "").lower() in ("1", "true", "yes", "on")

def _debug_log(msg: str) -> None:
    """
    Print debug logs only when debug is enabled to avoid noisy output in prod.
    """
    if _is_debug_enabled():
        try:
            print(f"[DEBUG] {msg}")
        except Exception:
            pass

def _is_schema_or_list_query(q: str) -> bool:
    """
    Heuristic to detect queries asking for schema/columns or listing unique values.
    Examples: "what are the customer names", "list all columns", "distinct model names", "what are the values of ..."
    """
    import re
    ql = (q or "").lower()
    patterns = [
        r"\bcolumns?\b",
        r"\bschema\b",
        r"\bwhat\s+are\s+the\s+.+\bnames?\b",
        r"\blist\s+all\b",
        r"\bunique\b",
        r"\bdistinct\b",
        r"\bvalues?\b\s+of\b",
        r"\bwhat\s+are\s+the\s+customer\s+names\b",
        r"\bmodel\s+names?\b",
    ]
    return any(re.search(p, ql) for p in patterns)

def _extract_table_snippets_from_combined(combined: str, max_chars: int = 12000) -> str:
    """
    Extracts the most useful Excel table snippets from the combined uploaded context:
      - [Sheet: ...] headers
      - Columns (...)
      - JSON sample (first rows):
      - TSV preview:
    Stops each section at the first blank line after it to avoid dragging unrelated text.
    Returns a trimmed string limited by max_chars.
    """
    if not combined:
        return ""
    lines = combined.splitlines()
    out: List[str] = []
    i = 0
    def add_until_blank(start_idx: int) -> int:
        j = start_idx
        while j < len(lines) and lines[j].strip() != "":
            out.append(lines[j])
            j += 1
        # append one blank line as separator
        out.append("")
        return j

    while i < len(lines) and sum(len(x) + 1 for x in out) < max_chars:
        line = lines[i]
        if line.startswith("[Sheet:"):
            # Always include the sheet line
            out.append(line)
            i += 1
            continue
        if line.startswith("Columns (") or line.startswith("JSON sample") or line.startswith("TSV preview") or line.startswith("Unique values by column"):
            i = add_until_blank(i)
            continue
        i += 1

    result = "\n".join(out).strip()
    if len(result) > max_chars:
        return result[:max_chars]
    return result


def _try_answer_trains_between_query(query: str, raw_json_docs: List[Dict[str, Any]]) -> Optional[str]:
    """
    Attempt a deterministic answer for queries like "which trains go from TVC to TCR"
    using uploaded raw JSON documents that contain train objects with 'route' arrays.

    Returns:
        A concise natural language answer string if a confident match is found; otherwise None.
    """
    import re

    if not query or not raw_json_docs:
        return None

    # Parse JSON docs into a list of train dicts
    trains: List[Dict[str, Any]] = []
    for d in raw_json_docs:
        try:
            js = d.get("json") or ""
            if not js:
                continue
            data = json.loads(js)
        except Exception:
            continue
        # Root array of trains
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and isinstance(item.get("route"), list):
                    trains.append(item)
        # Root object with a list of trains under some key
        elif isinstance(data, dict):
            for v in data.values():
                if isinstance(v, list):
                    for item in v:
                        if isinstance(item, dict) and isinstance(item.get("route"), list):
                            trains.append(item)

    if not trains:
        return None

    q = (query or "").strip()

    # 1) Try to extract explicit station codes like "TVC->TCR", "TVC to TCR", "from TVC to TCR"
    code_pair_patterns = [
        r"\bfrom\s+([A-Z]{2,5})\s+to\s+([A-Z]{2,5})\b",
        r"\b([A-Z]{2,5})\s*(?:-|->|—|–|>|to)\s*([A-Z]{2,5})\b",
    ]
    src_code: Optional[str] = None
    dst_code: Optional[str] = None
    for pat in code_pair_patterns:
        m = re.search(pat, q, flags=re.IGNORECASE)
        if m:
            src_code = m.group(1).upper()
            dst_code = m.group(2).upper()
            break

    # 2) Fallback: try to capture station names "from A to B" and map them to codes
    if not (src_code and dst_code):
        # Build a mapping of station name (lower) -> code from the uploaded trains
        name_to_code: Dict[str, str] = {}
        for t in trains:
            for stop in (t.get("route") or []):
                if not isinstance(stop, dict):
                    continue
                sname = (stop.get("station_name") or "").strip().lower()
                scode = (stop.get("station_code") or "").strip().upper()
                if sname and scode and sname not in name_to_code:
                    name_to_code[sname] = scode

        name_pair_patterns = [
            r"\bfrom\s+([A-Za-z][A-Za-z\s]+?)\s+to\s+([A-Za-z][A-Za-z\s]+?)\b",
        ]
        for pat in name_pair_patterns:
            m = re.search(pat, q, flags=re.IGNORECASE)
            if m:
                n1 = m.group(1).strip().lower()
                n2 = m.group(2).strip().lower()
                # Exact match first; fallback to substring contains if not found
                def _resolve_name(n: str) -> Optional[str]:
                    if n in name_to_code:
                        return name_to_code[n]
                    # substring heuristic
                    for k, v in name_to_code.items():
                        if n in k:
                            return v
                    return None
                src_code = src_code or _resolve_name(n1)
                dst_code = dst_code or _resolve_name(n2)
                break

    if not (src_code and dst_code):
        return None

    # Find trains that contain both stations in order
    matches: List[Dict[str, Any]] = []
    for t in trains:
        route = t.get("route")
        if not isinstance(route, list) or not route:
            continue
        codes = [str((stop or {}).get("station_code") or "").upper() for stop in route if isinstance(stop, dict)]
        if src_code in codes and dst_code in codes:
            i = codes.index(src_code)
            j = codes.index(dst_code)
            if i < j:
                # Collect times (if available)
                dep = ""
                arr = ""
                try:
                    src_stop = route[i] if i < len(route) else {}
                    dst_stop = route[j] if j < len(route) else {}
                    src_dep = (src_stop or {}).get("departure_time") or (src_stop or {}).get("dep_time") or ""
                    dst_arr = (dst_stop or {}).get("arrival_time") or (dst_stop or {}).get("arr_time") or ""
                    dep = f", dep {src_code} {src_dep}" if src_dep else ""
                    arr = f", arr {dst_code} {dst_arr}" if dst_arr else ""
                except Exception:
                    pass
                matches.append({
                    "name": (t.get("name") or t.get("train_name") or t.get("title") or "").strip(),
                    "number": (t.get("number") or t.get("train_number") or "").strip(),
                    "dep": dep,
                    "arr": arr,
                })

    if not matches:
        return f"No trains found from {src_code} to {dst_code} in the uploaded data."

    # Format a concise answer
    parts: List[str] = []
    for m in matches:
        label = f"{m['name']} ({m['number']})".strip()
        timing = f"{m['dep']}{m['arr']}".strip()
        parts.append(f"- {label}{(' ' + timing) if timing else ''}")
    header = f"Trains from {src_code} to {dst_code}:"
    return header + "\n" + "\n".join(parts)


# PUBLIC_INTERFACE
def get_gemini_response(
    query: str,
    memory: ConversationBufferMemory,
    extra_context: str = "",
) -> str:
    """
    PUBLIC_INTERFACE
    Enhance the answer using Google Gemini API, considering the chat context and any user-uploaded context.

    Never include statements about sources, knowledge base, RAG, or meta-assertions in the prompt or response.

    Args:
        query (str): User query.
        memory (ConversationBufferMemory): Conversation memory buffer.
        extra_context (str): Additional, retrieved context from uploaded files. May be empty.

    Returns:
        str: Model answer.
    """
    mem_str = memory.buffer_as_str if hasattr(memory, "buffer_as_str") else ""
    # Trim extra context to a reasonable size to avoid overwhelming the model
    MAX_CONTEXT_CHARS = 12000
    trimmed_extra = (extra_context or "").strip()
    if len(trimmed_extra) > MAX_CONTEXT_CHARS:
        trimmed_extra = trimmed_extra[: MAX_CONTEXT_CHARS]

    # Compose prompt with uploaded/retrieved context primarily.
    # Important: instruct Gemini to parse JSON arrays of objects (from Excel) precisely for data-based questions.
    guidance = (
        "If the provided context contains a JSON array of objects representing spreadsheet rows, treat each object as a "
        "row and each key as a column name. Use ONLY this JSON array for answering questions derived from spreadsheets; "
        "ignore any TSV or human-readable summaries if present. For listing-type questions (e.g., 'what are the customer "
        "names?' or 'list all model names'), derive the unique values directly from the JSON array. Use exact cell values, "
        "avoid fabricating values, and present results clearly. For calculation/lookup questions, compute answers precisely "
        "from the JSON data. Do not add disclaimers."
    )
    prompt = (
        f"You are an expert software assistant.\n"
        f"User's question: '{query}'\n"
        f"{f'Additional user-provided context (may be relevant):\n{trimmed_extra}\n' if trimmed_extra else ''}"
        f"Conversation history:\n{mem_str}\n"
        f"{guidance}\n"
        f"Please answer the user's latest question. Be concise and clear."
    )

    gemini_api_key = get_gemini_api_key()
    if not gemini_api_key:
        raise HTTPException(status_code=500, detail="Gemini API key is not set in environment variables.")
    try:
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content([{"role": "user", "parts": [prompt]}])
        raw_answer = response.text.strip()
        return _clean_gemini_output(raw_answer)
    except Exception as e:
        # Fallback to minimal message if Gemini API fails
        return _clean_gemini_output(f"[Gemini enhancement unavailable: {e}]")


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
def chat(request: ChatRequest, response: Response):
    """
    PUBLIC_INTERFACE
    Handles user's chat request. All answers come directly from Gemini.
    If the user has uploaded files for this session, the most relevant snippets from those files are retrieved
    via semantic vector search and provided as additional context to Gemini. The API returns only Gemini's final answer.
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

        # Build context for Gemini:
        # If Excel JSON rows exist for this session, use ONLY the JSON array as context.
        MAX_CONTEXT_CHARS = 12000

        def _rows_to_json_str(rows: List[Dict[str, Any]], max_chars: int = MAX_CONTEXT_CHARS) -> tuple[str, int]:
            """
            Build a compact JSON array string within the character budget.
            Returns:
                tuple[str, int]: (json_array_string, included_count)
            """
            if not rows:
                return "[]", 0
            out_parts: List[str] = []
            current = 2  # for [ ]
            included = 0
            for r in rows:
                s = json.dumps(r, ensure_ascii=False, separators=(",", ":"))
                sep = ", " if out_parts else ""
                if current + len(sep) + len(s) > max_chars:
                    break
                out_parts.append(s)
                current += len(sep) + len(s)
                included += 1
            return "[" + ", ".join(out_parts) + "]", included

        session_ctx = CONTEXT_STORE.get(session_id, {})
        raw_json_docs = session_ctx.get("raw_json_docs", []) or []
        json_rows: List[Dict[str, Any]] = session_ctx.get("xlsx_json_rows", []) or []
        used_table_snippets = False  # retained for debug metadata compatibility
        top_chunks: List[str] = []  # ensure defined for telemetry

        # Route query fast-path: If the user asks for trains between two stations and we have raw JSON,
        # compute the answer deterministically and return immediately.
        fast_answer = _try_answer_trains_between_query(request.query, raw_json_docs)
        if fast_answer:
            try:
                # Save to memory for chat continuity
                memory.save_context({"input": request.query}, {"output": fast_answer})
            except Exception:
                pass
            return ChatAnswerResponse(answer=fast_answer)

        def _pack_json_docs_for_prompt(docs: List[Dict[str, Any]], max_chars: int = MAX_CONTEXT_CHARS) -> tuple[str, bool, int, int]:
            """
            Build a valid JSON array string like:
              [{"source":"file.json","data": <JSON>}, ...]
            honoring a character cap. If a single doc cannot fit, include a truncated
            string sample to keep JSON valid:
              {"source":"file.json","data_truncated":true,"data_sample":"..."}
            Returns:
              (json_array_string, truncated_flag, included_docs, total_docs)
            """
            parts: List[str] = []
            current = 2  # for [ ]
            truncated = False
            included_docs = 0
            total_docs = len(docs)
            for d in docs:
                fn = d.get("filename") or "json"
                js = d.get("json") or ""
                entry = f'{{"source":{json.dumps(fn)}, "data": {js}}}'
                sep = ", " if parts else ""
                if current + len(sep) + len(entry) <= max_chars:
                    parts.append(entry)
                    current += len(sep) + len(entry)
                    included_docs += 1
                    continue
                # Fallback: truncated sample as a JSON string to keep overall structure valid
                budget = max_chars - current - len(sep) - len(fn) - 60  # room for keys and punctuation
                if budget > 0 and js:
                    sample = js[: max(0, budget)] + ("..." if budget < len(js) else "")
                else:
                    sample = ""
                safe_sample = json.dumps(sample, ensure_ascii=False)
                fallback_entry = f'{{"source":{json.dumps(fn)}, "data_truncated": true, "data_sample": {safe_sample}}}'
                if current + len(sep) + len(fallback_entry) <= max_chars:
                    parts.append(fallback_entry)
                    current += len(sep) + len(fallback_entry)
                    truncated = True
                    included_docs += 1  # count as represented (sample)
                # We break after attempting fallback; remaining docs omitted
                truncated = True
                break
            # If not all docs could be included, mark truncated
            if included_docs < total_docs:
                truncated = True
            return "[" + ", ".join(parts) + "]", truncated, included_docs, total_docs

        # Track truncation state and human-readable detail
        truncated_flag = False
        trunc_detail = ""

        if raw_json_docs:
            # Bypass RAG and directly provide raw JSON content to Gemini
            packed, truncated, included_docs, total_docs = _pack_json_docs_for_prompt(raw_json_docs, max_chars=MAX_CONTEXT_CHARS)
            retrieved_context = packed
            if truncated:
                truncated_flag = True
                trunc_detail = f"Included {included_docs} of {total_docs} JSON document(s) from uploaded files."
        elif json_rows:
            packed_rows, included_count = _rows_to_json_str(json_rows, max_chars=MAX_CONTEXT_CHARS)
            retrieved_context = packed_rows
            total_rows = len(json_rows)
            if included_count < total_rows:
                truncated_flag = True
                trunc_detail = f"Included {included_count} of {total_rows} row(s) from spreadsheet/JSON data."
        else:
            # No JSON docs or Excel rows: fall back to vector search over any uploaded text
            top_chunks = _vector_search(session_id, request.query, top_k=3)
            retrieved_context = "\n---\n".join(top_chunks).strip()
            # If retrieval is extremely weak and we have combined text, include a small portion as last resort
            combined = session_ctx.get("combined", "")
            if (not retrieved_context or len(retrieved_context) < 100) and combined:
                # Keep explicit trim so we can signal truncation
                if len(combined) > MAX_CONTEXT_CHARS:
                    truncated_flag = True
                    trunc_detail = f"Included a truncated portion (~{MAX_CONTEXT_CHARS} chars) of uploaded text to fit the prompt."
                retrieved_context = combined[:MAX_CONTEXT_CHARS]

        # Ensure final context respects prompt budget; detect if this trimming causes truncation
        if retrieved_context:
            if len(retrieved_context) > MAX_CONTEXT_CHARS:
                truncated_flag = True
                if not trunc_detail:
                    trunc_detail = f"Context trimmed to ~{MAX_CONTEXT_CHARS} characters to fit the model prompt."
                retrieved_context = retrieved_context[:MAX_CONTEXT_CHARS]

        # Debug/logging: store what reached Gemini for traceability
        try:
            # Ensure session context dictionary exists
            ctx = CONTEXT_STORE.get(session_id) or {}
            ctx["last_query"] = request.query
            ctx["last_top_chunks_count"] = len(top_chunks)
            ctx["last_used_table_snippets"] = used_table_snippets
            ctx["last_context_chars"] = len(retrieved_context or "")
            ctx["last_retrieved_context"] = (retrieved_context or "")[:4000]
            CONTEXT_STORE[session_id] = ctx
            _debug_log(
                f"session={session_id} top_chunks={len(top_chunks)} used_table_snips={used_table_snippets} "
                f"extra_len={len(retrieved_context or '')} "
                f"context_sample={(retrieved_context or '')[:300].replace('\\n',' ')[:300]}"
            )
        except Exception:
            # Non-fatal
            pass

        # Compose Gemini answer with retrieved context (if any)
        try:
            gemini_answer = get_gemini_response(request.query, memory, extra_context=retrieved_context or "")
        except Exception as e:
            gemini_answer = "[Gemini unavailable: {}]".format(e)

        # If context was truncated, prepend a clear notice and set response headers to inform the UI
        if truncated_flag:
            # Limit detail header length to avoid excessively large headers
            detail_header = trunc_detail[:400] if trunc_detail else "Context truncated due to prompt size limits."
            response.headers["X-Context-Truncated"] = "true"
            response.headers["X-Context-Truncation-Detail"] = detail_header
            notice_lines = [
                "Notice: The uploaded file content was too large to include fully in the AI prompt.",
                "The answer below is based on a truncated subset of your files and may be incomplete.",
            ]
            if trunc_detail:
                notice_lines.append(f"Details: {trunc_detail}")
            notice = "\n".join(notice_lines).strip()
        else:
            response.headers["X-Context-Truncated"] = "false"
            notice = ""

        # Final output cleaning and save in memory
        gemini_answer = _clean_gemini_output(_safe_str_output(gemini_answer))
        final_answer = (notice + "\n\n" + gemini_answer).strip() if notice else gemini_answer
        try:
            memory.save_context({"input": request.query}, {"output": _safe_str_output(final_answer)})
        except Exception:
            pass  # Do not raise for failed memory update

        return ChatAnswerResponse(answer=final_answer)

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
@app.post(
    "/chat/upload-context",
    response_model=UploadContextResponse,
    tags=["Chat"],
    summary="Upload context files for a chat session",
    description=(
        "Accepts one or more files via multipart/form-data and extracts readable text from supported types "
        "(.docx, .xlsx, .pdf, .txt, .json). The extracted content is stored per session and used as additional context "
        "when answering subsequent chat queries. Builds a vector index (Gemini embeddings) for semantic retrieval. "
        "To improve reliability and avoid timeouts on large files, heavy indexing is performed in the background. "
        "Returns an acknowledgment with per-file processing results and a preview."
    ),
    responses={
        400: {"description": "Validation error or no files provided"},
        415: {"description": "Unsupported media type"},
    },
)
async def upload_chat_context(
    session_id: str = Form(..., description="Session ID to associate uploaded context with"),
    files: List[UploadFile] = File(..., description="One or more files (.docx, .xlsx, .pdf, .txt, .json)"),
    background_tasks: BackgroundTasks = None,
):
    """
    PUBLIC_INTERFACE
    Upload and process files to add user-provided context for a given chat session.

    Process:
        - Extract readable text (offloaded to a threadpool to avoid blocking the event loop).
        - Store a combined text preview.
        - Build a semantic index (chunk + embed + store) in a background task to avoid request timeouts.
        - For Excel files, also extract a row-wise JSON array and store it as the primary structured context for queries.

    Args:
        session_id (str): The chat session ID.
        files (List[UploadFile]): Uploaded files (multipart/form-data).
        background_tasks (BackgroundTasks): FastAPI background task handler used to run indexing.

    Returns:
        UploadContextResponse: Processing results and acknowledgment.
    """
    from .file_utils import (
        extract_text_from_bytes,
        summarize_text_preview,
        extract_xlsx_as_rowwise_json,
        extract_json_as_rowwise_records,
        generate_json_rag_chunks,
    )

    if not session_id or not isinstance(session_id, str):
        raise HTTPException(status_code=400, detail="session_id must be provided as a non-empty string.")
    if not files or len(files) == 0:
        raise HTTPException(status_code=400, detail="At least one file must be provided.")

    results: List[UploadedFileResult] = []
    combined_text_parts: List[str] = []
    total_chars = 0
    excel_rows_accumulator: List[Dict[str, Any]] = []
    json_docs_accumulator: List[Dict[str, Any]] = []

    # Helper to index text after response is returned
    def _background_indexer(sid: str, fname: str, txt: str):
        try:
            _index_text_for_session(sid, fname, txt)
        except Exception:
            # Swallow exceptions to prevent background task from crashing the server
            pass

    def _background_index_structured(sid: str, chunk_list: List[Dict[str, Any]]):
        try:
            _index_structured_chunks_for_session(sid, chunk_list or [])
        except Exception:
            pass

    for f in files:
        filename = f.filename or "unnamed"
        # Read file bytes asynchronously
        try:
            data = await f.read()
        except Exception as e:
            results.append(
                UploadedFileResult(
                    filename=filename, size=0, content_chars=0, preview="", error=f"Failed to read file: {e}"
                )
            )
            continue

        size = len(data or b"")
        # Extract in a worker thread (openpyxl/pdfminer/docx are CPU/IO heavy)
        try:
            text, err = await run_in_threadpool(extract_text_from_bytes, filename, data or b"")
        except Exception as e:
            text, err = "", f"Extraction failed: {e}"

        preview = summarize_text_preview(text, max_chars=500) if text else ""
        chars = len(text)

        # For structured files, also extract row-wise records as primary structured context
        xlsx_rows: List[Dict[str, Any]] = []
        if (filename or "").lower().endswith(".xlsx"):
            try:
                xlsx_rows, rows_err = await run_in_threadpool(
                    extract_xlsx_as_rowwise_json, filename, data or b"", True
                )
                if rows_err:
                    # Do not fail processing; just log in preview/error if needed
                    preview = (preview + f" [XLSX rows warn: {rows_err}]").strip()
                else:
                    # Accumulate for the session
                    excel_rows_accumulator.extend(xlsx_rows)
            except Exception as e:
                # Non-fatal: keep text extraction result
                preview = (preview + f" [XLSX rows extraction error: {e}]").strip()
        elif (filename or "").lower().endswith(".json"):
            # Capture raw JSON content (compact) for direct prompt injection
            try:
                raw_str = (data or b"").decode("utf-8", errors="ignore").strip()
                if raw_str:
                    parsed = json.loads(raw_str)
                    compact = json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
                    json_docs_accumulator.append(
                        {"filename": filename, "json": compact, "chars": len(compact)}
                    )
            except Exception as e:
                preview = (preview + f" [JSON raw capture error: {e}]").strip()
            # Also extract row-wise records for RAG fallback (existing behavior)
            try:
                json_rows, rows_err = await run_in_threadpool(
                    extract_json_as_rowwise_records, filename, data or b""
                )
                if rows_err:
                    preview = (preview + f" [JSON rows warn: {rows_err}]").strip()
                else:
                    excel_rows_accumulator.extend(json_rows)
            except Exception as e:
                preview = (preview + f" [JSON rows extraction error: {e}]").strip()
            # Generate schema-aware chunks for JSON and index them in the background
            try:
                json_chunks, json_chunks_err = await run_in_threadpool(
                    generate_json_rag_chunks, filename, data or b""
                )
                if json_chunks_err:
                    preview = (preview + f" [JSON chunking warn: {json_chunks_err}]").strip()
                elif json_chunks:
                    if background_tasks is not None:
                        background_tasks.add_task(_background_index_structured, session_id, json_chunks)
            except Exception as e:
                preview = (preview + f" [JSON chunking error: {e}]").strip()

        # Append to combined only if successful and non-empty
        if text and not err:
            combined_text_parts.append(f"[{filename}]\n{text}\n")
            total_chars += chars
            # Offload indexing and embeddings to a background task to avoid request timeouts
            if background_tasks is not None:
                background_tasks.add_task(_background_indexer, session_id, filename, text)

        results.append(
            UploadedFileResult(
                filename=filename,
                size=size,
                content_chars=chars,
                preview=preview,
                error=err,
            )
        )

    # Update the session context store with both combined textual context and structured JSON rows.
    prev_ctx = CONTEXT_STORE.get(session_id, {})
    prev_combined = prev_ctx.get("combined", "")
    prev_files = prev_ctx.get("files", [])
    prev_rows: List[Dict[str, Any]] = prev_ctx.get("xlsx_json_rows", []) or []
    prev_raw_docs: List[Dict[str, Any]] = prev_ctx.get("raw_json_docs", []) or []

    # Merge textual combined content if we have any extracted text
    if total_chars > 0:
        combined_text = "\n".join(combined_text_parts).strip()
        merged_combined = (prev_combined + "\n\n" + combined_text).strip() if prev_combined else combined_text
    else:
        merged_combined = prev_combined

    # Merge Excel JSON rows regardless of text extraction result (JSON may be primary)
    merged_rows = prev_rows + excel_rows_accumulator if excel_rows_accumulator else prev_rows
    merged_raw_docs = prev_raw_docs + json_docs_accumulator if json_docs_accumulator else prev_raw_docs

    CONTEXT_STORE[session_id] = {
        "files": prev_files + [r.model_dump() for r in results],
        "combined": merged_combined,
        "xlsx_json_rows": merged_rows,
        "raw_json_docs": merged_raw_docs,
    }

    message = (
        "Processed files successfully. Session context updated. Background indexing in progress."
        if total_chars > 0 or excel_rows_accumulator
        else "Processed files, but no readable content was extracted."
    )

    return UploadContextResponse(
        session_id=session_id,
        files_processed=results,
        total_chars=total_chars,
        message=message,
    )
