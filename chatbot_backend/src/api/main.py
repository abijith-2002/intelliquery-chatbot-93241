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
import json
import pandas as pd

from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory

# New: stdlib imports for timeouts/limits and background tasks
import os
from fastapi import BackgroundTasks

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

# ---- Upload/processing configuration (tunable via environment) ----
# Max upload size for a single file in bytes (default 50 MB). Reverse proxy may also enforce limits.
MAX_UPLOAD_BYTES = int(os.getenv("CHATBOT_MAX_UPLOAD_BYTES", str(50 * 1024 * 1024)))
# Max cumulative size across all files in one request (default 100 MB).
MAX_TOTAL_UPLOAD_BYTES = int(os.getenv("CHATBOT_MAX_TOTAL_UPLOAD_BYTES", str(100 * 1024 * 1024)))
# Max time (seconds) allowed to process an upload request before early return (default 25s).
UPLOAD_PROCESS_TIMEOUT_SECS = int(os.getenv("CHATBOT_UPLOAD_PROCESS_TIMEOUT_SECS", "25"))
# Whether to build embeddings on the upload endpoint (may be heavy); default True. Disable if causing timeouts.
BUILD_EMBEDDINGS_ON_UPLOAD = os.getenv("CHATBOT_BUILD_EMBEDDINGS_ON_UPLOAD", "true").lower() in ("1", "true", "yes")
# Whether to parse full Excel to DataFrames on upload; if False, defer to separate endpoint to avoid long blocking.
PARSE_EXCEL_ON_UPLOAD = os.getenv("CHATBOT_PARSE_EXCEL_ON_UPLOAD", "true").lower() in ("1", "true", "yes")
# For Excel schema, cap rows sampled per sheet to reduce heavy stats on very large files. Default 10_000 rows.
EXCEL_SCHEMA_MAX_SAMPLE_ROWS = int(os.getenv("CHATBOT_EXCEL_SCHEMA_MAX_SAMPLE_ROWS", "10000"))

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

# Per-session Excel data store
# Structure:
#   EXCEL_STORE[session_id] = {
#       "df": pandas.DataFrame from most recently uploaded sheet (default: first non-empty sheet),
#       "sheets": Dict[str, DataFrame],
#       "schema": Dict[str, Any]  # compact schema/stats for Gemini
#   }
EXCEL_STORE: Dict[str, Dict[str, Any]] = {}

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


class ExcelQueryRequest(BaseModel):
    """Schema for querying uploaded Excel data using Gemini-driven pandas expressions."""
    session_id: str = Field(..., description="Session ID that has an uploaded Excel sheet")
    sheet_name: Optional[str] = Field(default=None, description="Optional sheet name to select; defaults to first non-empty sheet uploaded")
    column_name: Optional[str] = Field(default=None, description="Optional column name to select; if provided, df will be narrowed before processing.")
    query: str = Field(..., description="User's natural language question/task about the Excel data")
    mode: Optional[str] = Field(
        default="auto",
        description=(
            "Data inclusion mode for Gemini prompt: "
            "'auto' (default), 'summary' (schema + sample), 'entire' (attempt full data for small files), "
            "'sample' (force sample), or 'sheet' (force a particular sheet/column view)."
        ),
    )


class ExcelQueryResponse(BaseModel):
    """Schema for response containing computed result and Gemini narrative."""
    expression: str = Field(..., description="The pandas expression used to compute the result")
    result: Any = Field(..., description="Computed result as JSON-serializable data")
    narrative: str = Field(..., description="Natural language explanation generated by Gemini")


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

        # Log user input with a temp placeholder output
        try:
            memory.save_context({"input": request.query}, {"output": _safe_str_output("", "No answer available")})
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to save user context: {e}")

        # Retrieve top-k relevant chunks from vector index
        top_chunks = _vector_search(session_id, request.query, top_k=3)
        retrieved_context = "\n---\n".join(top_chunks).strip()

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

def _log_upload_issue(filename: str, message: str) -> None:
    """
    Lightweight logger for upload issues. Keeps logs uniform without external deps.
    """
    try:
        print(f"[upload] file={filename} :: {message}")
    except Exception:
        pass


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
        422: {"description": "Validation error (ensure multipart/form-data with 'session_id' and 'files' fields)"},
    },
)
def upload_chat_context(
    session_id: str = Form(..., description="Session ID to associate uploaded context with"),
    files: List[UploadFile] = File(..., description="One or more files (.docx, .xlsx, .pdf, .txt)"),
    background_tasks: BackgroundTasks = None,
):
    """
    PUBLIC_INTERFACE
    Upload and process files to add user-provided context for a given chat session.

    Request:
        Content-Type: multipart/form-data
        Fields:
            - session_id (form field): string, required
            - files (one or more file fields): allowed types [.txt, .pdf, .docx, .xlsx]

    Process:
        - Extract readable text.
        - Split into overlapping chunks.
        - Embed each chunk using Gemini embeddings (if API key available).
        - Store chunks and embeddings in a per-session in-memory index for retrieval.

    Returns:
        UploadContextResponse: Processing results and acknowledgment.

    Error responses:
        400: Missing session_id or files; size limit exceeded; or other validation error.
        415: Unsupported file media type.
        422: Likely incorrect request format. Ensure multipart/form-data is used with 'session_id' and 'files'.
    """
    from .file_utils import extract_text_from_bytes, summarize_text_preview
    from .excel_utils import parse_xlsx_to_dataframe, build_schema_for_gemini

    if not session_id or not isinstance(session_id, str):
        raise HTTPException(status_code=400, detail="session_id must be provided as a non-empty string.")
    if not files or len(files) == 0:
        raise HTTPException(status_code=400, detail="At least one file must be provided.")

    results: List[UploadedFileResult] = []
    combined_text_parts: List[str] = []
    total_chars = 0
    total_bytes_accum = 0

    def _read_file_enforcing_limits_sync(upload: UploadFile) -> bytes:
        """
        Read file stream in chunks synchronously, enforcing per-file and total limits.
        UploadFile.read() is async, but in a sync path under most FastAPI route functions
        it's safer to access the underlying file to avoid event-loop misuse.
        """
        nonlocal total_bytes_accum
        chunk_size = 1024 * 1024  # 1 MB
        collected: List[bytes] = []
        read_so_far = 0
        # Prefer underlying file-like object if available (SpooledTemporaryFile)
        fileobj = getattr(upload, "file", None)
        if fileobj is None:
            # Fallback: small single read if .file isn't present
            data = upload.file.read() if hasattr(upload, "file") else b""
            if not isinstance(data, (bytes, bytearray)):
                data = b""
            if len(data) > 0:
                read_so_far += len(data)
                total_bytes_accum += len(data)
                if read_so_far > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=400, detail=f"File {upload.filename} exceeds max size limit.")
                if total_bytes_accum > MAX_TOTAL_UPLOAD_BYTES:
                    raise HTTPException(status_code=400, detail="Total upload size exceeds max limit.")
                collected.append(bytes(data))
            return b"".join(collected)

        # Ensure file pointer at start
        try:
            fileobj.seek(0)
        except Exception:
            pass

        while True:
            chunk = fileobj.read(chunk_size)
            if not chunk:
                break
            if not isinstance(chunk, (bytes, bytearray)):
                # Safety: coerce or stop
                chunk = bytes(str(chunk), "utf-8")
            read_so_far += len(chunk)
            total_bytes_accum += len(chunk)
            if read_so_far > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=400, detail=f"File {upload.filename} exceeds max size limit.")
            if total_bytes_accum > MAX_TOTAL_UPLOAD_BYTES:
                raise HTTPException(status_code=400, detail="Total upload size exceeds max limit.")
            collected.append(bytes(chunk))
        return b"".join(collected)

    # Use a monotonic time function for wall-clock checks
    try:
        import time
        start = time.monotonic()
        now_func = time.monotonic
    except Exception:
        start = 0.0
        now_func = (lambda: 0.0)

    for f in files:
        filename = f.filename or "unnamed"
        # Enforce wall time for the whole request
        now = now_func()
        if start and now - start > UPLOAD_PROCESS_TIMEOUT_SECS:
            partial_msg = "Partial processing due to time limit; larger files will be processed in background (if configured)."
            return UploadContextResponse(
                session_id=session_id,
                files_processed=results,
                total_chars=total_chars,
                message=partial_msg,
            )

        # Read file bytes (chunked) using sync pipeline to avoid event-loop misuse
        try:
            data = _read_file_enforcing_limits_sync(f)
        except HTTPException as he:
            results.append(
                UploadedFileResult(
                    filename=filename, size=0, content_chars=0, preview="", error=he.detail if isinstance(he.detail, str) else "Upload size limit exceeded"
                )
            )
            continue
        except Exception as e:
            results.append(
                UploadedFileResult(
                    filename=filename, size=0, content_chars=0, preview="", error=f"Failed to read file: {e}"
                )
            )
            continue

        size = len(data or b"")
        # Extract (fast path preview)
        text, err = extract_text_from_bytes(filename, data or b"")
        preview = summarize_text_preview(text, max_chars=500) if text else ""
        chars = len(text)

        # Excel parsing: optionally defer heavy DataFrame extraction
        if filename.lower().endswith(".xlsx") and not err and size > 0:
            if not PARSE_EXCEL_ON_UPLOAD:
                # We still want the UI to know there is Excel content even if deferring parse.
                preview = (preview + " [Excel parsing deferred by server configuration; schema will be built when querying.]").strip()
            else:
                try:
                    # Build a reduced-size schema by sampling, to avoid large memory/time usage
                    sheets = parse_xlsx_to_dataframe(data or b"")
                    # Downsample large sheets before schema to reduce heavy stats
                    sampled_sheets = {}
                    any_truncated = False
                    for sname, sdf in sheets.items():
                        if sdf is None:
                            sampled_sheets[sname] = sdf
                        else:
                            if EXCEL_SCHEMA_MAX_SAMPLE_ROWS > 0 and getattr(sdf, "shape", (0, 0))[0] > EXCEL_SCHEMA_MAX_SAMPLE_ROWS:
                                sampled_sheets[sname] = sdf.head(EXCEL_SCHEMA_MAX_SAMPLE_ROWS)
                                any_truncated = True
                            else:
                                sampled_sheets[sname] = sdf
                    # Choose default df: first non-empty sheet; otherwise first sheet
                    chosen_df = None
                    for sname, sdf in sheets.items():
                        try:
                            if sdf is not None and not sdf.empty:
                                chosen_df = sdf
                                break
                        except Exception:
                            continue
                    if chosen_df is None and sheets:
                        chosen_df = next(iter(sheets.values()))
                    # Build schema with truncation notes
                    schema = build_schema_for_gemini(sampled_sheets, max_examples_per_col=3, max_rows_per_sheet=EXCEL_SCHEMA_MAX_SAMPLE_ROWS)
                    if any_truncated:
                        # Add a user-facing note into the preview so large files clearly show up with a warning.
                        preview = (preview + f" [schema built from first {EXCEL_SCHEMA_MAX_SAMPLE_ROWS} rows per sheet; truncated for speed]").strip()
                    # Store regardless of size (always include schema). Merge with existing session Excel store if present.
                    existing = EXCEL_STORE.get(session_id, {})
                    merged_sheets = dict(existing.get("sheets", {}) or {})
                    # Later uploads with same sheet name overwrite previous; different files append
                    merged_sheets.update(sheets or {})
                    # Choose a default df: prefer a non-empty chosen_df; otherwise keep existing df if available
                    final_df = chosen_df or existing.get("df")
                    # Keep previous schema notes and rebuild later on query if needed
                    EXCEL_STORE[session_id] = {
                        "df": final_df,
                        "sheets": merged_sheets,
                        "schema": schema or existing.get("schema"),
                    }
                except MemoryError as me:
                    preview = (preview + f" [Excel parsing skipped due to memory limits; try reducing file size or sheets. Error: {me}]").strip()
                except Exception as e:
                    preview = (preview + f" [Excel parsing warning: {e}]").strip()

        # Append to combined only if successful and non-empty
        if text and not err:
            combined_text_parts.append(f"[{filename}]\n{text}\n")
            total_chars += chars

            # Build semantic index may be expensive; optionally defer to background
            def _index_job():
                try:
                    _index_text_for_session(session_id, filename, text)
                except Exception:
                    pass

            if BUILD_EMBEDDINGS_ON_UPLOAD:
                # If background_tasks provided, schedule to avoid blocking request
                if background_tasks is not None:
                    background_tasks.add_task(_index_job)
                else:
                    try:
                        _index_job()
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


# PUBLIC_INTERFACE
@app.post(
    "/chat/excel-query",
    response_model=ExcelQueryResponse,
    tags=["Chat"],
    summary="Query uploaded Excel data (safe pandas execution via Gemini)",
    description=(
        "Given a session_id with an uploaded Excel file, this endpoint intelligently prepares Gemini inputs:\n"
        "- For small/medium files, it may include full schema and data (or sampled data) so Gemini can compute directly.\n"
        "- For very large files and aggregate-style queries, it computes aggregates over the entire DataFrame in Python/pandas,\n"
        "  and sends only the results/schema for Gemini to interpret.\n"
        "It then evaluates a pandas expression in a restricted sandbox if needed and returns the computed JSON result with a narrative."
    ),
    responses={
        400: {"description": "Validation error or missing Excel data"},
        500: {"description": "Gemini error or evaluation error"},
    },
)
def excel_query(request: ExcelQueryRequest):
    """
    PUBLIC_INTERFACE
    Execute a user query against the uploaded Excel sheet(s) for a session.

    Strategy:
        - Small/medium data: include schema and data (per mode=auto/summary/entire) in the Gemini prompt.
        - Very large data with aggregation intent: compute aggregates across the full DataFrame in pandas and ask Gemini to narrate.
        - For extremely large files, use rolling/chunk aggregation to compute results over the entire dataset without loading all rows at once (when possible).
        - Otherwise, fallback to prompting Gemini for a pandas expression using schema metadata and evaluate it safely.

    Steps:
        1) Retrieve stored schema and DataFrame for the session.
        2) Optionally narrow to a sheet and/or a selected column per request.
        3) Choose strategy based on DataFrame size, request.mode, and query intent (aggregation or not).
        4) Either send schema+data to Gemini for a direct answer, or ask Gemini for a pandas expression to evaluate safely.
        5) Return the result and a narrative explanation, plus diagnostics about truncation or chunking.

    Args:
        request (ExcelQueryRequest): session_id, optional sheet_name/column_name, user query, and mode.

    Returns:
        ExcelQueryResponse: The pandas expression (or note), computed result, and a narrative explanation.
    """
    from .excel_utils import (
        get_gemini_pandas_prompt,
        safe_eval_pandas_expression,
        normalize_result_for_json,
        build_schema_for_gemini,
    )
    from .excel_query_strategy import (
        SizeThresholds,
        estimate_df_size,
        is_aggregate_query,
        compute_aggregate_answer,
        build_prompt_with_schema_and_optional_data,
    )

    session_id = (request.session_id or "").strip()
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required.")

    store = EXCEL_STORE.get(session_id)
    if not store or ("sheets" not in store and "df" not in store):
        raise HTTPException(status_code=400, detail="No Excel data found for this session. Upload an .xlsx first.")

    # Select df
    df = None
    sheets = store.get("sheets") or {}
    # If a specific sheet is requested
    if request.sheet_name:
        if request.sheet_name not in sheets:
            raise HTTPException(status_code=400, detail=f"Sheet '{request.sheet_name}' not found.")
        df = sheets[request.sheet_name]
    else:
        # prefer stored default
        df = store.get("df")
        if df is None:
            # fallback to first non-empty or first
            for sname, sdf in sheets.items():
                try:
                    if isinstance(sdf, type(df)) and getattr(sdf, "empty", True) is False:
                        df = sdf
                        break
                except Exception:
                    continue
            if df is None and len(sheets) > 0:
                df = next(iter(sheets.values()))

    if df is None:
        # Debug aid for large uploads: log available sheets to server console
        try:
            available = list((sheets or {}).keys())
            print(f"[excel_query] No DataFrame resolved for session={session_id}. Available sheets: {available}")
        except Exception:
            pass
        raise HTTPException(status_code=400, detail="No usable DataFrame found in uploaded Excel.")

    # Optional column narrowing to focus the context and reduce size
    if getattr(request, "column_name", None):
        col = request.column_name
        if col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{col}' not found in selected DataFrame.")
        try:
            df = df[[col]]  # keep as DataFrame to preserve consistent downstream behavior
        except Exception:
            # Fallback: Series to_frame
            df = df[col].to_frame()

    # Ensure schema exists and is populated (rebuild if needed)
    schema = store.get("schema")
    rebuild_reason = None
    if not schema or not isinstance(schema, dict) or not schema.get("sheets"):
        rebuild_reason = "missing_or_invalid"
    else:
        try:
            has_columns = any((len(s.get("columns") or []) > 0) for s in schema.get("sheets", []))
            if not has_columns:
                rebuild_reason = "empty_columns"
        except Exception:
            rebuild_reason = "schema_inspection_failed"

    if rebuild_reason is not None:
        try:
            build_source = sheets if sheets else {"Sheet1": df}
            max_rows = int(os.getenv("CHATBOT_EXCEL_SCHEMA_MAX_SAMPLE_ROWS", "10000"))
            schema = build_schema_for_gemini(build_source, max_examples_per_col=3, max_rows_per_sheet=max_rows)
            store["schema"] = schema
            print(f"[excel_query] Rebuilt schema for session={session_id} reason={rebuild_reason} sheets={list(build_source.keys())}")
        except Exception as e:
            try:
                schema = build_schema_for_gemini({"Sheet1": df}, max_examples_per_col=3, max_rows_per_sheet=1000)
                store["schema"] = schema
                print(f"[excel_query] Minimal schema built due to error: {e}")
            except Exception as e2:
                print(f"[excel_query] Failed to build any schema: {e2}")
                raise HTTPException(status_code=500, detail="Failed to build schema for Gemini prompt.")

    # Compose strict prompt
    user_query = (request.query or "").strip()
    if not user_query:
        raise HTTPException(status_code=400, detail="query is required.")

    # Decide strategy and handle modes
    mode = (request.mode or "auto").lower()
    # Accept additional aliases
    if mode not in ("auto", "summary", "entire", "sample", "sheet"):
        mode = "auto"

    thresholds = SizeThresholds()
    size_class = "unknown"
    try:
        size_class = estimate_df_size(df, thresholds)
    except Exception:
        pass

    wants_agg = False
    try:
        wants_agg = is_aggregate_query(user_query)
    except Exception:
        wants_agg = False

    # Diagnostics container
    diagnostics: Dict[str, Any] = {
        "mode": mode,
        "size_class": size_class,
        "sheet": request.sheet_name or "default",
        "column": getattr(request, "column_name", None),
        "notes": [],
    }

    gemini_api_key = get_gemini_api_key()
    if not gemini_api_key:
        raise HTTPException(status_code=500, detail="Gemini API key is not set in environment variables.")
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")

    # Strategy A: Large DF and aggregate intent -> compute in pandas, ask Gemini to narrate
    if size_class == "large" and wants_agg:
        try:
            agg_payload = compute_aggregate_answer(df, user_query)
        except Exception as e:
            agg_payload = None
            diagnostics["notes"].append(f"aggregate_compute_failed: {e}")
            print(f"[excel_query] Aggregate compute failed: {e}")
        if agg_payload is not None:
            # Ask Gemini to explain the aggregates with schema context
            try:
                narr_prompt = (
                    "You are provided the schema of an Excel dataset and aggregate results computed over the full dataset.\n"
                    "Explain concisely (2-4 sentences) how these aggregates answer the user's question. Do not include code.\n\n"
                    f"User request:\n{user_query}\n\n"
                    f"Schema:\n{json.dumps(schema, ensure_ascii=False)}\n\n"
                    f"Aggregates:\n{json.dumps(agg_payload, ensure_ascii=False)}\n"
                    "If applicable, note that results were computed across the entire dataset using server-side aggregation."
                )
                narr_resp = model.generate_content([{"role": "user", "parts": [narr_prompt]}])
                narrative = _clean_gemini_output((narr_resp.text or "").strip())
            except Exception:
                narrative = "Aggregates were computed across the entire dataset to answer your question."
            return ExcelQueryResponse(
                expression="[server] computed aggregates in pandas over the entire DataFrame",
                result={"aggregates": agg_payload, "diagnostics": diagnostics},
                narrative=narrative or "Aggregates computed over full dataset.",
            )
        # If we cannot compute aggregates, fall through to expression strategy.

    # Strategy A2: Extremely large and non-aggregate -> rolling stats as safe fallback
    # Provide top-level stats to Gemini for summarization if data is too big to include
    try:
        if size_class == "large" and not wants_agg:
            # Compute lightweight overall stats without heavy memory usage
            # Only numeric columns; include row/column counts and null summaries
            num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            overall = {
                "rows": int(df.shape[0]),
                "cols": int(df.shape[1]),
                "numeric_cols": num_cols,
                "nulls_per_col": {c: int(df[c].isna().sum()) for c in df.columns},
            }
            # Use describe on numeric columns but limit to safe try
            try:
                overall["numeric_describe"] = df[num_cols].describe().to_dict() if num_cols else {}
            except Exception:
                overall["numeric_describe"] = {}
            diagnostics["notes"].append("used_largefile_overview_stats")
            narr_prompt = (
                "Given the user's question and a summary of a very large dataset, provide a concise answer or guidance. "
                "If exact computation is infeasible from the summary, clearly explain limitations and suggest a targeted filter/aggregation to run.\n\n"
                f"User request:\n{user_query}\n\n"
                f"Schema (compact):\n{json.dumps(schema, ensure_ascii=False)}\n\n"
                f"Dataset overview stats:\n{json.dumps(overall, ensure_ascii=False)}\n"
            )
            try:
                resp = model.generate_content([{"role": "user", "parts": [narr_prompt]}])
                narrative = _clean_gemini_output((resp.text or "").strip())
            except Exception:
                narrative = "Provided a high-level answer based on summary statistics due to dataset size."
            return ExcelQueryResponse(
                expression="[server] high-level summary (no expression evaluation)",
                result={"summary": "Large dataset; returned overview stats", "diagnostics": diagnostics, "overview": overall},
                narrative=narrative,
            )
    except Exception as e:
        diagnostics["notes"].append(f"largefile_overview_failed: {e}")
        # continue to next strategies

    # Strategy B: Small/Medium DF or mode requests data -> include schema (+ data/sample) and ask Gemini to produce a direct answer
    try:
        # Support explicit modes
        eff_mode = mode
        if mode == "sample":
            eff_mode = "summary"
        if mode == "sheet":
            # Encourage Gemini to focus on selected sheet/column; still use auto inclusion strategy
            eff_mode = "auto"
            diagnostics["notes"].append("sheet_mode_active")
        prompt_with_data, ctx_meta = build_prompt_with_schema_and_optional_data(
            user_query=user_query, schema=schema, df=df, mode=eff_mode, thresholds=thresholds
        )
    except Exception as e:
        prompt_with_data, ctx_meta = "", {"included": {"schema": True, "data_rows": 0, "data_truncated": False}}
        diagnostics["notes"].append(f"schema_data_prompt_failed: {e}")
        print(f"[excel_query] Failed to build data-inclusive prompt: {e}")

    direct_answer_possible = bool(
        prompt_with_data and (size_class in ("small", "medium") or mode in ("entire", "summary", "sample"))
    )

    if direct_answer_possible:
        try:
            resp = model.generate_content(
                [{"role": "user", "parts": [prompt_with_data + "\n\nProvide the final answer succinctly."]}]
            )
            answer_text = _clean_gemini_output((resp.text or "").strip())
            return ExcelQueryResponse(
                expression="[server] direct answer via schema/data prompt (no expression evaluation)",
                result={"answer": answer_text, "context_meta": ctx_meta, "diagnostics": diagnostics},
                narrative="Answer generated by Gemini using provided schema and data context.",
            )
        except Exception as e:
            diagnostics["notes"].append(f"direct_answer_failed: {e}")
            print(f"[excel_query] Direct answer path failed: {e}")
            # Fallthrough to expression strategy

    # Strategy C: Expression generation based on schema-only (original pipeline)
    pandas_prompt = get_gemini_pandas_prompt(user_query, schema)

    # Logging prompt preview and schema summary
    try:
        preview_len = min(len(pandas_prompt), 1000)
        print(f"[excel_query] Prompt preview (first {preview_len} chars) for session={session_id}:\n{pandas_prompt[:preview_len]}")
        try:
            schema_sheets = len(schema.get("sheets", []))
            col_counts = [len(s.get("columns", [])) for s in schema.get("sheets", [])]
            print(f"[excel_query] Schema summary: sheets={schema_sheets} cols_per_sheet={col_counts}")
        except Exception:
            pass
    except Exception:
        pass

    try:
        response = model.generate_content([{"role": "user", "parts": [pandas_prompt]}])
        expression = (response.text or "").strip()
        if "```" in expression:
            expression = expression.replace("```python", "").replace("```py", "").replace("```", "").strip()
        if expression.lower().startswith("python"):
            expression = expression[6:].strip()
        if expression.endswith(";"):
            expression = expression[:-1].strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get pandas expression from Gemini: {e}")

    if not expression or "\n" in expression:
        expression = " ".join((expression or "").split())
    if not expression:
        raise HTTPException(status_code=500, detail="Gemini did not return a valid pandas expression.")

    try:
        computed = safe_eval_pandas_expression(df, expression)
        json_result = normalize_result_for_json(computed)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to evaluate pandas expression safely: {e}")

    try:
        narration_prompt = (
            "You are given:\n"
            f"- The user's request: {user_query}\n"
            f"- The pandas expression executed on DataFrame 'df': {expression}\n"
            "Provide a concise explanation (2-4 sentences) of what this code does and how it answers the user's request. "
            "Do not include code in the answer."
        )
        narr_resp = model.generate_content([{"role": "user", "parts": [narration_prompt]}])
        narrative = _clean_gemini_output((narr_resp.text or "").strip())
    except Exception:
        narrative = "Computed the result based on your request using a pandas expression."

    return ExcelQueryResponse(
        expression=expression,
        result={"value": json_result, "diagnostics": diagnostics},
        narrative=narrative,
    )
