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
import re

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

# Per-session uploaded context store (legacy tracking for previews).
# Structure: { session_id: { "files": [ {filename, size, chars, preview, error?} ], "combined": str } }
CONTEXT_STORE: Dict[str, Dict[str, Any]] = {}

# Per-session schema summaries extracted from large Excel files for LLM context lookup.
# Structure:
#   EXCEL_SCHEMA_STORE[session_id] = {
#       "<filename>": {
#           "schema": { ...structured schema... },
#           "summary": "Sheet: ...; Columns: [...]"   # human-readable quick summary
#       },
#       ...
#   }
EXCEL_SCHEMA_STORE: Dict[str, Dict[str, Any]] = {}

# Per-session in-memory vector index for RAG.
# Structure:
#   RAG_INDEX_STORE[session_id] = {
#       "chunks": [ {"text": str, "filename": str} , ...],
#       "embeddings": [ [float, ...], ...],
#       "embedding_model": str
#   }
RAG_INDEX_STORE: Dict[str, Dict[str, Any]] = {}

# Per-session Excel raw bytes and parsed DataFrames for analytics-on-demand.
# Structure:
#   EXCEL_RAW_STORE[session_id] = {
#       "<filename>": {
#           "bytes": b"...",                              # raw uploaded bytes
#           "sheets": { "<sheet_name>": pandas.DataFrame }
#       }
#   }
EXCEL_RAW_STORE: Dict[str, Dict[str, Any]] = {}

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
    """
    _ensure_session_index(session_id)
    chunks = _split_into_chunks(text, chunk_size_words=180, overlap_words=40)
    if not chunks:
        return
    vectors = _embed_texts(chunks)

    store = RAG_INDEX_STORE[session_id]
    for chunk, vec in zip(chunks, vectors):
        store["chunks"].append({"text": chunk, "filename": filename})
        store["embeddings"].append(vec)  # vec could be None; retrieval handles fallback


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
        for item, vec in zip(index["chunks"], index["embeddings"]):
            if vec:
                score = _cosine_similarity(query_vec, vec)
            else:
                # If a particular chunk lacks embedding, degrade to lexical fallback for that item
                score = _simple_similarity(query, item["text"])
            scored.append((score, item["text"]))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [text for _, text in scored[:top_k]]

    # Fallback: lexical similarity if no query embedding
    scored_lex = [(_simple_similarity(query, item["text"]), item["text"]) for item in index["chunks"]]
    scored_lex.sort(key=lambda x: x[0], reverse=True)
    return [text for _, text in scored_lex[:top_k]]


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
    prompt = (
        f"You are an expert software assistant.\n"
        f"User's question: '{query}'\n"
        f"{f'Additional user-provided context (may be relevant):\n{trimmed_extra}\n' if trimmed_extra else ''}"
        f"Conversation history:\n{mem_str}\n"
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
def chat(request: ChatRequest):
    """
    PUBLIC_INTERFACE
    Handles user's chat request. Attempts to detect data analytics queries (sum, avg, count, unique, etc.)
    over uploaded Excel data and compute those directly using pandas. Otherwise, falls back to Gemini with RAG.
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

        # 1) Try analytics detection + computation on uploaded Excel data first
        analytics_answer = _maybe_answer_analytics(session_id, request.query)
        if analytics_answer is not None:
            final_ans = _clean_gemini_output(_safe_str_output(analytics_answer))
            try:
                memory.save_context({"input": request.query}, {"output": _safe_str_output(final_ans)})
            except Exception:
                pass
            return ChatAnswerResponse(answer=final_ans)

        # 2) Otherwise, use RAG + Gemini
        # Retrieve top-k relevant chunks from vector index
        top_chunks = _vector_search(session_id, request.query, top_k=3)
        # Include any Excel schema summaries available for this session to help the model understand data layout
        excel_schema_notes = []
        sess_schema = EXCEL_SCHEMA_STORE.get(session_id) or {}
        # Keep this lightweight: include only up to 2 schema summaries
        for fname, entry in list(sess_schema.items())[:2]:
            if isinstance(entry, dict) and "summary" in entry:
                excel_schema_notes.append(f"[{fname} schema]\n{entry['summary']}")
        retrieved_context = "\n---\n".join([*top_chunks, *excel_schema_notes]).strip()

        # Compose Gemini answer with retrieved context (if any)
        try:
            gemini_answer = get_gemini_response(request.query, memory, extra_context=retrieved_context)
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
def _maybe_answer_analytics(session_id: str, query: str) -> Optional[str]:
    """
    PUBLIC_INTERFACE
    Attempt to detect and compute simple analytics from uploaded Excel data.

    Supported intents (case-insensitive, simple heuristics):
    - sum/sum of <col>
    - average/avg/mean of <col>
    - min of <col>
    - max of <col>
    - median of <col>
    - count rows (with optional filters)
    - count unique/unique count/distinct of <col>
    - list unique values of <col> (limited)
    Optionally with 'in sheet <sheet>' and simple equality filter 'where <col> = <value>'.

    Returns:
        str answer if detected and computed, otherwise None to fall back to LLM.
    """
    import pandas as pd
    q = (query or "").strip()
    if not q:
        return None

    store = EXCEL_RAW_STORE.get(session_id)
    if not store:
        return None

    # Normalize whitespace
    q_lc = " ".join(q.lower().split())

    # Extract optional sheet hint: "in sheet <name>" or "from sheet <name>"
    sheet_hint = None
    m_sheet = re.search(r"(?:in|from)\s+sheet\s+([a-z0-9 _\-./]+)", q_lc)
    if m_sheet:
        sheet_hint = m_sheet.group(1).strip()

    # Extract optional simple equality filter: where <col> = <value>
    filter_col = None
    filter_val: Optional[str] = None
    m_where = re.search(r"where\s+([a-z0-9 _\-./]+)\s*=\s*['\"]?([^'\"\n\r]+?)['\"]?(?:\s|$)", q_lc)
    if m_where:
        filter_col = m_where.group(1).strip()
        filter_val = m_where.group(2).strip()

    # Operation and target column detection
    OPS = [
        ("sum", ["sum", "total"]),
        ("avg", ["average", "avg", "mean"]),
        ("min", ["min", "minimum", "lowest", "smallest"]),
        ("max", ["max", "maximum", "highest", "largest"]),
        ("median", ["median"]),
        ("count_unique", ["count unique", "unique count", "distinct count", "count distinct"]),
        ("list_unique", ["list unique", "unique values", "distinct values"]),
        ("count_rows", ["count rows", "row count", "number of rows", "how many rows", "how many records"]),
    ]

    op_detected: Optional[str] = None
    for op, keywords in OPS:
        if any(k in q_lc for k in keywords):
            op_detected = op
            break

    # Try to extract column name after 'of' or 'for' or 'in column'
    target_col = None
    m_col = re.search(r"(?:of|for|in\s+column)\s+([a-z0-9 _\-./]+)", q_lc)
    if m_col:
        target_col = m_col.group(1).strip()

    # If we have no operation and no strong column pattern and the question doesn't look analytical, bail.
    analytical_clues = ["sum", "average", "avg", "mean", "min", "max", "median", "count", "unique", "distinct"]
    if not op_detected and not any(c in q_lc for c in analytical_clues):
        return None

    # Iterate available dataframes and try to find a match
    def iter_session_frames():
        for fname, meta in store.items():
            if not isinstance(meta, dict):
                continue
            sheets = meta.get("sheets") or {}
            for sname, df in sheets.items():
                yield fname, sname, df

    # Helper to pick frames by sheet hint
    frames: List[Tuple[str, str, "pd.DataFrame"]] = []
    for fn, sh, df in iter_session_frames():
        if sheet_hint:
            if sh.lower().strip() == sheet_hint:
                frames.append((fn, sh, df))
        else:
            frames.append((fn, sh, df))

    if not frames:
        return None

    # If a target column is provided, resolve best matching column by case-insensitive equality or fuzzy containment.
    def resolve_column(df: "pd.DataFrame", col_hint: Optional[str]) -> Optional[str]:
        if col_hint is None:
            return None
        cols = list(df.columns)
        # exact case-insensitive match
        for c in cols:
            if str(c).lower().strip() == col_hint:
                return c
        # contains
        for c in cols:
            if col_hint in str(c).lower():
                return c
        return None

    # For count rows, we don't need a column necessarily.
    # For other ops except count_rows, we generally need a column.
    for fn, sh, df in frames:
        try:
            df_work = df.copy()

            # Apply simple equality filter if requested
            if filter_col and filter_val is not None:
                # Try to map filter_col to a real column
                resolved_filter_col = resolve_column(df_work, filter_col)
                if resolved_filter_col:
                    # Cast to str compare fallback if direct compare fails
                    try:
                        df_work = df_work[df_work[resolved_filter_col] == filter_val]
                    except Exception:
                        df_work = df_work[df_work[resolved_filter_col].astype(str) == str(filter_val)]

            # If operation is row count or question hints about counting rows
            if op_detected == "count_rows" or ("count" in q_lc and "unique" not in q_lc and target_col is None):
                return f"Row count in {fn} / {sh}: {len(df_work):,}"

            # Resolve column if needed
            col_name = resolve_column(df_work, target_col) if target_col else None

            # If it looks like we need a column but couldn't resolve, try to infer numeric column if only one numeric exists
            if not col_name and op_detected in {"sum", "avg", "min", "max", "median"}:
                numeric_cols = df_work.select_dtypes(include=["number"]).columns.tolist()
                if len(numeric_cols) == 1:
                    col_name = numeric_cols[0]

            if op_detected in {"sum", "avg", "min", "max", "median"} and not col_name:
                # Can't confidently proceed
                continue

            # Perform the operation
            if op_detected == "sum":
                val = pd.to_numeric(df_work[col_name], errors="coerce").sum(skipna=True)
                return f"Sum of '{col_name}' in {fn} / {sh}: {val:,.4f}"
            if op_detected == "avg":
                val = pd.to_numeric(df_work[col_name], errors="coerce").mean(skipna=True)
                return f"Average of '{col_name}' in {fn} / {sh}: {val:,.4f}"
            if op_detected == "min":
                val = pd.to_numeric(df_work[col_name], errors="coerce").min(skipna=True)
                return f"Minimum of '{col_name}' in {fn} / {sh}: {val:,.4f}"
            if op_detected == "max":
                val = pd.to_numeric(df_work[col_name], errors="coerce").max(skipna=True)
                return f"Maximum of '{col_name}' in {fn} / {sh}: {val:,.4f}"
            if op_detected == "median":
                val = pd.to_numeric(df_work[col_name], errors="coerce").median(skipna=True)
                return f"Median of '{col_name}' in {fn} / {sh}: {val:,.4f}"
            if op_detected == "count_unique" and col_name:
                n = df_work[col_name].nunique(dropna=True)
                return f"Unique values count for '{col_name}' in {fn} / {sh}: {n:,}"
            if op_detected == "list_unique" and col_name:
                vals = df_work[col_name].dropna().astype(str).unique().tolist()
                # Limit display
                preview = ", ".join(vals[:20])
                more = "" if len(vals) <= 20 else f" (+{len(vals)-20} more)"
                return f"Unique values for '{col_name}' in {fn} / {sh}: {preview}{more}"
            # Generic "count unique X" pattern without explicit op_detected mapping
            if "count" in q_lc and "unique" in q_lc and col_name:
                n = df_work[col_name].nunique(dropna=True)
                return f"Unique values count for '{col_name}' in {fn} / {sh}: {n:,}"

        except Exception:
            # Continue to next frame
            continue

    # No confident structured answer
    return None


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
        "(.docx, .xlsx, .pdf, .txt). The extracted content is stored per session and used as additional context "
        "when answering subsequent chat queries. Builds a vector index (Gemini embeddings) for semantic retrieval. "
        "Returns an acknowledgment with per-file processing results and a preview."
    ),
    responses={
        400: {"description": "Validation error or no files provided"},
        415: {"description": "Unsupported media type"},
    },
)
def upload_chat_context(
    session_id: str = Form(..., description="Session ID to associate uploaded context with"),
    files: List[UploadFile] = File(..., description="One or more files (.docx, .xlsx, .pdf, .txt)"),
):
    """
    PUBLIC_INTERFACE
    Upload and process files to add user-provided context for a given chat session.

    Process:
        - Extract readable text.
        - Split into overlapping chunks.
        - Embed each chunk using Gemini embeddings (if API key available).
        - Store chunks and embeddings in a per-session in-memory index for retrieval.

    Args:
        session_id (str): The chat session ID.
        files (List[UploadFile]): Uploaded files (multipart/form-data).

    Returns:
        UploadContextResponse: Processing results and acknowledgment.
    """
    from .file_utils import (
        extract_text_from_bytes,
        summarize_text_preview,
        extract_xlsx_schema_or_text,
        build_xlsx_column_chunks_from_schema,
        render_xlsx_column_chunk_text,
        iter_xlsx_row_slices,
    )

    if not session_id or not isinstance(session_id, str):
        raise HTTPException(status_code=400, detail="session_id must be provided as a non-empty string.")
    if not files or len(files) == 0:
        raise HTTPException(status_code=400, detail="At least one file must be provided.")

    results: List[UploadedFileResult] = []
    combined_text_parts: List[str] = []
    total_chars = 0

    for f in files:
        filename = f.filename or "unnamed"
        # Read file bytes
        try:
            data = f.file.read()
        except Exception as e:
            results.append(
                UploadedFileResult(
                    filename=filename, size=0, content_chars=0, preview="", error=f"Failed to read file: {e}"
                )
            )
            continue

        size = len(data or b"")
        # Extract with special handling for Excel large files
        text = ""
        err = None
        schema_summary = None
        is_xlsx = (filename or "").lower().endswith(".xlsx")
        if is_xlsx:
            # Keep raw bytes for analytics-on-demand; also parse to dataframes now
            try:
                import pandas as pd
                from io import BytesIO
                xls = pd.ExcelFile(BytesIO(data or b""))
                # Initialize session map if missing
                sess_excel_map = EXCEL_RAW_STORE.get(session_id, {})
                # Build per-sheet dataframes (lightweight read; pandas handles streaming reasonably)
                sheet_map: Dict[str, Any] = {}
                for sname in xls.sheet_names:
                    try:
                        df = xls.parse(sname)
                        # Normalize columns to string for consistent matching
                        df.columns = [str(c) for c in df.columns]
                        sheet_map[sname] = df
                    except Exception:
                        continue
                sess_excel_map[filename] = {"bytes": data or b"", "sheets": sheet_map}
                EXCEL_RAW_STORE[session_id] = sess_excel_map
            except Exception:
                # If pandas parse fails, analytics will be unavailable for this file
                pass

            summary_or_text, schema, excel_err = extract_xlsx_schema_or_text(filename, data or b"")
            if excel_err:
                text, err = "", excel_err
            else:
                if schema:
                    # Large file path: store schema and provide human-readable summary as preview
                    schema_summary = {"schema": schema, "summary": summary_or_text}
                    text = summary_or_text  # Use the concise schema summary as the extracted "text"

                    # Additionally, build column chunks and index each chunk separately
                    try:
                        col_chunks = build_xlsx_column_chunks_from_schema(schema, group_size=100, prefer_theme_groups=True)
                        for ch in col_chunks:
                            chunk_text = render_xlsx_column_chunk_text(
                                filename=filename,
                                sheet=ch.get("sheet", "Sheet"),
                                group_label=ch.get("group_label", ""),
                                column_names=ch.get("column_names", []),
                            )
                            _index_text_for_session(session_id, f"{filename}:{ch.get('sheet')}:{ch.get('group_label')}", chunk_text)
                    except Exception:
                        # Do not block upload if chunking fails
                        pass
                else:
                    # Small file path: we have full text
                    text = summary_or_text
                    # Additionally, if rows are very wide, build record slices and index them independently.
                    try:
                        # Use defaults: slice if >200 columns, groups of 100 columns, cap processed rows for safety.
                        row_slices = iter_xlsx_row_slices(
                            filename=filename,
                            content=data or b"",
                            per_row_max_cols=200,
                            group_size=100,
                            # Safety cap to avoid huge ingestion in extremely tall sheets:
                            max_rows=2000,
                        )
                        for rs in row_slices:
                            # Store each slice as independent minimal unit
                            _index_text_for_session(
                                session_id,
                                f"{filename}:{rs.get('sheet')}:{rs.get('row_index')}:{rs.get('meta',{}).get('group_label','')}",
                                rs.get("text", ""),
                            )
                    except Exception:
                        # Do not fail upload if row slicing fails
                        pass
        else:
            text, err = extract_text_from_bytes(filename, data or b"")

        preview = summarize_text_preview(text, max_chars=500) if text else ""
        chars = len(text)

        # Append to combined only if successful and non-empty
        if not err and text:
            combined_text_parts.append(f"[{filename}]\n{text}\n")
            total_chars += chars

            # Store Excel schema for large files for LLM context lookup
            if schema_summary:
                sess_map = EXCEL_SCHEMA_STORE.get(session_id, {})
                sess_map[filename] = schema_summary
                EXCEL_SCHEMA_STORE[session_id] = sess_map

            # Build semantic index: chunk + embed + store
            try:
                _index_text_for_session(session_id, filename, text)
            except Exception:
                # Do not fail upload on indexing failure; retrieval will fall back gracefully.
                pass

        results.append(
            UploadedFileResult(
                filename=filename,
                size=size,
                content_chars=chars,
                preview=preview,
                error=err,
            )
        )

    # If at least one file produced content, update the legacy session context store (for optional previews)
    if total_chars > 0:
        combined_text = "\n".join(combined_text_parts).strip()
        prev_ctx = CONTEXT_STORE.get(session_id, {})
        prev_combined = prev_ctx.get("combined", "")
        prev_files = prev_ctx.get("files", [])

        # Merge with previous context if any
        merged_combined = (prev_combined + "\n\n" + combined_text).strip() if prev_combined else combined_text
        CONTEXT_STORE[session_id] = {
            "files": prev_files + [r.model_dump() for r in results],
            "combined": merged_combined,
        }

    message = (
        "Processed files successfully. Session context updated and indexed."
        if total_chars > 0
        else "Processed files, but no readable content was extracted."
    )

    return UploadContextResponse(
        session_id=session_id,
        files_processed=results,
        total_chars=total_chars,
        message=message,
    )
