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
from typing import List, Optional, Dict, Any
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

# Per-session uploaded context store (legacy tracking for previews).
# Structure: { session_id: { "files": [ {filename, size, chars, preview, error?} ], "combined": str } }
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

# Register XLSX upload routes
try:
    from .xlsx_routes import router as xlsx_router
    app.include_router(xlsx_router)
except Exception:
    # non-fatal if optional dependencies missing
    pass

# Start background worker for embeddings on app import/startup
try:
    from .background_worker import start_worker
    start_worker()
except Exception:
    # non-fatal in case of import errors; app can still serve other endpoints
    pass

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
    """Schema for the chat response that returns the final answer and relevant data context."""
    answer: str = Field(..., description="Final natural language answer returned by Gemini.")
    data_context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Relevant data used during answering, e.g., matched rows and fields for frontend display."
    )


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
    job_id: Optional[str] = Field(default=None, description="Background job identifier for status tracking")
    catalog: Optional[Dict[str, Any]] = Field(default=None, description="Generated catalog/metadata for uploaded spreadsheets")


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
    Handles user queries with two modes:
      - Type 1 (structured XLSX retrieval): return ONLY pandas code operating on the uploaded dataframe.
      - Type 2 (general QA): return ONLY natural language using Gemini with RAG context.
    Never mix code and explanation in the same response.
    """
    import traceback
    import re

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

        # Save incoming turn
        try:
            memory.save_context({"input": request.query}, {"output": _safe_str_output("", "No answer available")})
        except Exception:
            raise HTTPException(status_code=500, detail="Failed to save user context.")

        query_text = (request.query or "").strip()

        # Classification heuristic:
        # If the session has XLSX loaded and the query appears to ask for data/table operations,
        # classify as Type 1. Indicators: mentions of columns, filters, sort, top/first/last, sum/avg/etc.
        from .xlsx_utils import XLSX_SESSIONS, get_default_dataframe, build_pandas_code

        def looks_type1(q: str) -> bool:
            if session_id not in XLSX_SESSIONS:
                return False
            patterns = [
                r"\b(top|first|last)\s+\d+",
                r"\b(sort|order)\s+by\b",
                r"\blimit\s+\d+",
                r"\b(sum|avg|average|mean|count|max|min)\s+of\b",
                r"[A-Za-z0-9_ ]+\s*:\s*[^\n,;]+",      # contains filter like status: open
                r"[A-Za-z0-9_ ]+\s*(>=|<=|>|<|==|=)\s*[0-9]",  # numeric comparison
                r"\bselect\b", r"\bfilter\b", r"\bgroup\b", r"\baggregate\b",
            ]
            for pat in patterns:
                if re.search(pat, q, flags=re.IGNORECASE):
                    return True
            # Mention of "table", "rows", "columns"
            if re.search(r"\b(table|rows?|columns?)\b", q, flags=re.IGNORECASE):
                return True
            return False

        if looks_type1(query_text):
            # Type 1: Generate pandas code using the default dataframe
            try:
                fname, sheet, df = get_default_dataframe(session_id)
            except Exception:
                # If no dataframe, fall back to Type 2
                fname, sheet, df = None, None, None

            if df is not None:
                df_var = "df"  # fixed reference name for generated code
                code = build_pandas_code(query_text, df_var, df)
                # For Type 1: must return ONLY code, no explanation.
                answer_text = code.strip()
                # Save final turn strictly as code
                try:
                    memory.save_context({"input": query_text}, {"output": _safe_str_output(answer_text)})
                except Exception:
                    pass
                return ChatAnswerResponse(answer=answer_text, data_context={"mode": "type1", "source": {"file": fname, "sheet": sheet}})
            # else fallback to Type 2 if dataframe not available

        # Type 2: General QA with RAG (legacy retrieval path preserved)
        # Reuse previous retrieval code for vector store + lexical fallback
        # Extract simple filters for keyword_filter (as legacy behavior)
        field_value_pairs: Dict[str, Any] = {}
        numeric_conditions: List[Dict[str, Any]] = []
        for m in re.finditer(r"([A-Za-z0-9_ ]+)\s*:\s*([^\n,;]+)", query_text):
            key = m.group(1).strip()
            val = m.group(2).strip()
            if key:
                field_value_pairs[key] = val
        for m in re.finditer(r"([A-Za-z0-9_ ]+)\s*(>=|<=|>|<|=)\s*([0-9]+(?:\.[0-9]+)?)", query_text):
            key = m.group(1).strip()
            op = m.group(2)
            num = float(m.group(3))
            numeric_conditions.append({"field": key, "op": op, "value": num})

        def _norm_key(k: str) -> str:
            return re.sub(r"[^a-z0-9_]", "", re.sub(r"\s+", "_", k.strip().lower()))

        normalized_where = { _norm_key(k): v for k, v in field_value_pairs.items() if k.strip() }

        from .vector_store import get_vector_store
        from .config_utils import get_gemini_api_key as _get_key

        data_context: Dict[str, Any] = {"retrieval": {"top_k": 5, "used": "vector_store"}, "hits": []}
        retrieved_context_text = ""

        query_vec = None
        try:
            key = _get_key()
            if key:
                genai.configure(api_key=key)
                result = genai.embed_content(model="models/text-embedding-004", content=query_text)
                qv = result.get("embedding", {}).get("values")
                if isinstance(qv, list) and qv:
                    query_vec = [float(x) for x in qv]
        except Exception:
            query_vec = None

        store = None
        try:
            store = get_vector_store()
        except Exception:
            store = None

        vector_hits: List[Dict[str, Any]] = []
        if store and query_vec:
            try:
                vector_hits = store.search(session_id=session_id, query_vector=query_vec, top_k=5)
            except Exception:
                vector_hits = []

        filtered_hits: List[Dict[str, Any]] = []
        if store and normalized_where:
            try:
                filtered_hits = store.keyword_filter(session_id=session_id, where=normalized_where, limit=100)
                def _passes_numeric(meta: Dict[str, Any]) -> bool:
                    for cond in numeric_conditions:
                        field = _norm_key(cond["field"])
                        val = meta.get(field)
                        try:
                            fval = float(val)
                        except Exception:
                            return False
                        op = cond["op"]
                        c = cond["value"]
                        if op == ">=" and not (fval >= c): return False
                        if op == "<=" and not (fval <= c): return False
                        if op == ">" and not (fval > c): return False
                        if op == "<" and not (fval < c): return False
                        if op == "=" and not (fval == c): return False
                    return True
                filtered_hits = [h for h in filtered_hits if _passes_numeric(h.get("metadata", {}))]
            except Exception:
                filtered_hits = []

        hits = filtered_hits if filtered_hits else vector_hits

        if not hits:
            top_chunks = _vector_search(session_id, query_text, top_k=3)
            retrieved_context_text = "\n---\n".join(top_chunks).strip()
            data_context["retrieval"]["used"] = "legacy_chunk_index"
            data_context["hits"] = [{"text": t} for t in top_chunks]
        else:
            contexts = []
            for h in hits:
                meta = h.get("metadata", {}) or {}
                parts = []
                for k, v in meta.items():
                    parts.append(f"{k}: {v}")
                row_text = " | ".join(parts)
                contexts.append(row_text)
            retrieved_context_text = "\n---\n".join(contexts[:5]).strip()
            data_context["hits"] = hits

        # Gemini natural language answer (Type 2)
        try:
            gemini_answer = get_gemini_response(query_text, memory, extra_context=retrieved_context_text)
        except Exception:
            gemini_answer = "[Gemini unavailable]"

        gemini_answer = _clean_gemini_output(_safe_str_output(gemini_answer))
        try:
            memory.save_context({"input": query_text}, {"output": _safe_str_output(gemini_answer)})
        except Exception:
            pass

        return ChatAnswerResponse(answer=gemini_answer, data_context=data_context)

    except HTTPException:
        raise
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


# PUBLIC_INTERFACE
@app.get(
    "/chat/context-status/{job_id}",
    tags=["Chat"],
    summary="Get context upload job status and metadata",
    description=(
        "Retrieve the status and spreadsheet catalog metadata for a previously started context upload job. "
        "Response includes current processing state, per-session embedding progress, row/column counts, "
        "dataset readiness, and any encountered embedding/indexing errors."
    ),
    responses={
        200: {
            "description": "Detailed job status including embedding progress and dataset readiness",
        },
        404: {"description": "Job not found"},
    },
)
def get_context_status(job_id: str):
    """
    PUBLIC_INTERFACE
    Get status for an upload job initiated by /chat/upload-context.

    Args:
        job_id (str): The job identifier returned by the upload endpoint.

    Returns:
        dict: Job status and metadata including per-file results and generated spreadsheet catalogs with:
            - status: current high-level job state (created|running|completed|completed_with_errors)
            - files: per-file processing details (size, content_chars, preview, error)
            - total_chars: total characters extracted across files
            - catalog: object containing:
                - embedding_progress: { queued, rows_enqueued, done, error }
                - dataset_ready: boolean flag indicating if dataset is ingest-ready
                - sheets_summary: list of { sheet_name, rows_scanned, columns_count }
                - index_errors: list of embedding/index errors encountered during processing
                - files: list of file-level catalog data (as generated by XLSX ingestion) or errors
    """
    from .job_tracker import get_job
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Enrich response with computed per-sheet summaries and consolidate errors from catalog content
    resp = job

    try:
        catalog = resp.setdefault("catalog", {})
        files_meta = catalog.get("files") or []
        sheets_summary = []
        index_errors = list(catalog.get("index_errors") or [])
        # Derive sheet summaries from each file's catalog if available
        for fmeta in files_meta:
            fname = fmeta.get("filename") or ""
            if "catalog_error" in fmeta:
                index_errors.append(f"Catalog error for {fname}: {fmeta.get('catalog_error')}")
                continue
            inner = fmeta.get("catalog") or {}
            for sheet in inner.get("sheets", []) or []:
                sheet_name = sheet.get("sheet_name") or ""
                rows_scanned = int(sheet.get("row_count_scanned") or 0)
                columns_count = len(sheet.get("columns") or [])
                sheets_summary.append(
                    {
                        "sheet_name": sheet_name,
                        "rows_scanned": rows_scanned,
                        "columns_count": columns_count,
                        "file": fname,
                    }
                )
            # If embedding prep error captured at file-level
            if "embedding_prep_error" in fmeta:
                index_errors.append(f"Embedding prep error for {fname}: {fmeta.get('embedding_prep_error')}")

        # Attach computed summaries
        catalog["sheets_summary"] = sheets_summary
        # Ensure embedding_progress keys exist and are integers
        ep = catalog.get("embedding_progress") or {}
        ep.setdefault("queued", int(ep.get("queued") or 0))
        ep.setdefault("rows_enqueued", int(ep.get("rows_enqueued") or 0))
        ep.setdefault("done", int(ep.get("done") or 0))
        ep.setdefault("error", int(ep.get("error") or 0))
        catalog["embedding_progress"] = ep

        # Dataset readiness heuristic: true if there is any sheet or total chars > 0 and not only errors
        dataset_ready = bool(sheets_summary) or int(resp.get("total_chars") or 0) > 0
        catalog["dataset_ready"] = bool(catalog.get("dataset_ready")) or dataset_ready

        # Deduplicate index errors
        if index_errors:
            # Remove falsy/duplicates
            clean_errs = []
            seen = set()
            for e in index_errors:
                if not e:
                    continue
                if e in seen:
                    continue
                seen.add(e)
                clean_errs.append(e)
            catalog["index_errors"] = clean_errs
        else:
            catalog.setdefault("index_errors", [])

        resp["catalog"] = catalog
    except Exception:
        # Best-effort enrichment; return base job if anything fails
        pass

    return resp


# --- FILE UPLOAD ENDPOINTS FOR CONTEXT ---

# PUBLIC_INTERFACE
@app.post(
    "/chat/upload-context",
    response_model=UploadContextResponse,
    tags=["Chat"],
    summary="Upload context files for a chat session",
    description=(
        "Accepts one or more files via multipart/form-data and extracts readable text from supported types "
        "(.docx, .pdf, .txt). The extracted content is stored per session and used as additional context "
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
    files: List[UploadFile] = File(..., description="One or more files (.docx, .pdf, .txt)"),
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
    from .file_utils import extract_text_from_bytes, summarize_text_preview
    from .job_tracker import start_job, finalize_job, JOBS  # type: ignore

    if not session_id or not isinstance(session_id, str):
        raise HTTPException(status_code=400, detail="session_id must be provided as a non-empty string.")
    if not files or len(files) == 0:
        raise HTTPException(status_code=400, detail="At least one file must be provided.")

    # Accepting .xlsx but not processed here; encourage dedicated endpoint for structured ingestion
    # .xlsx will be ignored in this endpoint and not counted toward text context.

    # Initialize job tracking
    filenames = [(f.filename or "unnamed") for f in files]
    job = start_job(session_id, filenames)

    results: List[UploadedFileResult] = []
    combined_text_parts: List[str] = []
    total_chars = 0

    for idx, f in enumerate(files):
        filename = f.filename or "unnamed"
        try:
            JOBS[job.job_id].files[idx].status = "processing"
        except Exception:
            pass

        # Read file bytes
        try:
            data = f.file.read()
            size = len(data or b"")
            if idx < len(JOBS[job.job_id].files):
                JOBS[job.job_id].files[idx].size = size
        except Exception as e:
            results.append(
                UploadedFileResult(
                    filename=filename, size=0, content_chars=0, preview="", error=f"Failed to read file: {e}"
                )
            )
            try:
                JOBS[job.job_id].files[idx].status = "error"
                JOBS[job.job_id].files[idx].error = f"Failed to read file: {e}"
            except Exception:
                pass
            continue

        # Extract text for supported types
        text, err = extract_text_from_bytes(filename, data or b"")
        preview = summarize_text_preview(text, max_chars=500) if text else ""
        chars = len(text)

        # store to job record
        try:
            rec = JOBS[job.job_id].files[idx]
            rec.preview = preview
            rec.content_chars = chars
            rec.message = "Processed"
            if err:
                rec.status = "error"
                rec.error = err
            else:
                rec.status = "done"
        except Exception:
            pass

        # Append to combined only if successful and non-empty
        if text and not err:
            combined_text_parts.append(f"[{filename}]\n{text}\n")
            total_chars += chars
            # Build semantic index: chunk + embed + store
            try:
                _index_text_for_session(session_id, filename, text)
            except Exception:
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

        merged_combined = (prev_combined + "\n\n" + combined_text).strip() if prev_combined else combined_text
        CONTEXT_STORE[session_id] = {
            "files": prev_files + [r.model_dump() for r in results],
            "combined": merged_combined,
        }

    # finalize job
    try:
        JOBS[job.job_id].total_chars = total_chars
        # No XLSX catalog; ensure catalog is empty or minimal
        JOBS[job.job_id].catalog = {}
        finalize_job(job.job_id)
    except Exception:
        pass

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
        job_id=job.job_id,
        catalog=None,
    )
