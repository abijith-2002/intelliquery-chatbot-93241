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

# Per-session store for uploaded Excel DataFrames and metadata.
# Structure:
#   EXCEL_STORE[session_id] = [
#       {
#           "filename": str,
#           "sheets": {
#               sheet_name: {
#                   "columns": List[str],
#                   "dtypes": Dict[str, str],
#                   "rows": int,
#                   "sample": List[Dict[str, Any]],   # head(5)
#                   "parquet_path": Optional[str]     # if persisted as parquet
#               }, ...
#           }
#       }, ...
#   ]
EXCEL_STORE: Dict[str, List[Dict[str, Any]]] = {}

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

    Note: Excel files are not indexed into the RAG store by design. Excel content is handled
    through DataFrames and metadata for specialized Excel question answering.

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


# PUBLIC_INTERFACE
@app.get(
    "/chat/excel/columns",
    tags=["Chat"],
    summary="List indexed Excel columns for a session",
    description="Return a flattened list of (filename, sheet, column) entries that were indexed for the given session's Excel uploads."
)
def list_excel_columns(session_id: str):
    """
    PUBLIC_INTERFACE
    Retrieve the list of indexed Excel columns for the provided session.

    Args:
        session_id (str): The chat session identifier.

    Returns:
        dict: {
            "session_id": str,
            "columns": [
                {"filename": str, "sheet": str, "column": str},
                ...
            ]
        }
    """
    if not session_id or not isinstance(session_id, str):
        raise HTTPException(status_code=400, detail="session_id must be a non-empty string.")

    # Lazy import to avoid circulars at module import time
    from .excel_index import EXCEL_COLUMN_INDEX

    # If no index, return empty
    if session_id not in EXCEL_COLUMN_INDEX:
        return {"session_id": session_id, "columns": []}

    # Enumerate all columns quickly using a wildcard name query (iterate internally)
    out = []
    files_map = EXCEL_COLUMN_INDEX[session_id].get("files", {})
    for fname, sheets in files_map.items():
        for sname, meta in sheets.items():
            for col in meta.get("columns", []):
                out.append({"filename": fname, "sheet": sname, "column": col})
    return {"session_id": session_id, "columns": out}


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
        - For .txt/.pdf/.docx: extract readable text, chunk, embed, and index for retrieval.
        - For .xlsx: parse with pandas.read_excel, extract metadata (columns, dtypes, sample), and
          store per-session DataFrame metadata (optionally persisting Parquet), WITHOUT embedding/indexing raw cells.

    Args:
        session_id (str): The chat session ID.
        files (List[UploadFile]): Uploaded files (multipart/form-data).

    Returns:
        UploadContextResponse: Processing results and acknowledgment.
    """
    from .file_utils import extract_text_from_bytes, summarize_text_preview
    import io
    import os
    import pandas as pd

    if not session_id or not isinstance(session_id, str):
        raise HTTPException(status_code=400, detail="session_id must be provided as a non-empty string.")
    if not files or len(files) == 0:
        raise HTTPException(status_code=400, detail="At least one file must be provided.")

    results: List[UploadedFileResult] = []
    combined_text_parts: List[str] = []
    total_chars = 0

    for f in files:
        filename = f.filename or "unnamed"
        lower_name = filename.lower()
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

        if lower_name.endswith(".xlsx"):
            # New behavior for Excel: parse with pandas, extract metadata, store, and create a human-readable preview
            try:
                bio = io.BytesIO(data or b"")
                # Read all sheets into dict of DataFrames
                all_sheets = pd.read_excel(bio, sheet_name=None)
                if session_id not in EXCEL_STORE:
                    EXCEL_STORE[session_id] = []
                excel_entry = {
                    "filename": filename,
                    "sheets": {}
                }

                # Optional Parquet persistence per sheet (in-memory option retained by omitting path)
                parquet_dir = os.path.join("tmp_parquet", session_id)
                os.makedirs(parquet_dir, exist_ok=True)

                preview_chunks = []
                from .excel_index import upsert_excel_column_index  # integrate column index

                for sheet_name, df in all_sheets.items():
                    # Collect metadata
                    cols = list(df.columns)
                    dtypes = {str(c): str(t) for c, t in zip(cols, df.dtypes.values)}
                    rows = int(df.shape[0])
                    sample = df.head(5).to_dict(orient="records")

                    # Compute small per-column sample values for index preview
                    column_samples = {}
                    try:
                        for c in cols:
                            # take up to 5 non-null unique sample values
                            series = df[c].dropna().unique().tolist()
                            # convert to native python types where possible
                            preview_vals = []
                            for v in series[:5]:
                                try:
                                    preview_vals.append(v.item() if hasattr(v, "item") else v)
                                except Exception:
                                    preview_vals.append(str(v))
                            column_samples[str(c)] = preview_vals
                    except Exception:
                        column_samples = {}

                    # Persist Parquet for larger dataframes to avoid memory pressure
                    parquet_path = None
                    try:
                        if rows > 5000:
                            safe_sheet = str(sheet_name).replace("/", "_")
                            parquet_path = os.path.join(parquet_dir, f"{os.path.basename(filename)}__{safe_sheet}.parquet")
                            df.to_parquet(parquet_path, index=False)
                    except Exception:
                        parquet_path = None  # If pyarrow missing or write fails, ignore.

                    excel_entry["sheets"][str(sheet_name)] = {
                        "columns": cols,
                        "dtypes": dtypes,
                        "rows": rows,
                        "sample": sample,
                        "parquet_path": parquet_path,
                    }

                    # Update fast lookup (no embeddings by default; can be toggled later)
                    upsert_excel_column_index(
                        session_id=session_id,
                        filename=filename,
                        sheet_name=str(sheet_name),
                        columns=[str(c) for c in cols],
                        dtypes={str(k): str(v) for k, v in dtypes.items()},
                        sample_rows=sample,
                        column_samples=column_samples,
                        build_embeddings=False,  # set True in future to precompute per-column embeddings
                    )

                    # Build compact preview per sheet
                    preview_chunks.append(
                        f"[Sheet: {sheet_name}] Columns: {cols} | Rows: {rows} | Sample(2): {df.head(2).to_dict(orient='records')}"
                    )

                EXCEL_STORE[session_id].append(excel_entry)

                # For Excel, we do not index or embed; preview is metadata only
                excel_preview_text = " || ".join(preview_chunks)
                results.append(
                    UploadedFileResult(
                        filename=filename,
                        size=size,
                        content_chars=len(excel_preview_text),
                        preview=summarize_text_preview(excel_preview_text, max_chars=500),
                        error=None,
                    )
                )
                # We do NOT add to combined_text_parts or index for Excel
            except Exception as e:
                results.append(
                    UploadedFileResult(
                        filename=filename,
                        size=size,
                        content_chars=0,
                        preview="",
                        error=f"Failed to parse Excel: {e}",
                    )
                )
            continue  # proceed to next file

        # Non-Excel handling stays the same: extract text and index
        text, err = extract_text_from_bytes(filename, data or b"")
        preview = summarize_text_preview(text, max_chars=500) if text else ""
        chars = len(text)

        if text and not err:
            combined_text_parts.append(f"[{filename}]\n{text}\n")
            total_chars += chars
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

    # Update legacy context preview only for non-Excel extracted text
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
    else:
        # Still record files results even if no non-Excel text; keep files list
        prev_ctx = CONTEXT_STORE.get(session_id, {"files": [], "combined": ""})
        CONTEXT_STORE[session_id] = {
            "files": prev_ctx.get("files", []) + [r.model_dump() for r in results],
            "combined": prev_ctx.get("combined", ""),
        }

    message = "Processed files successfully. Session context updated and indexed for non-Excel files; Excel stored with metadata only."

    return UploadContextResponse(
        session_id=session_id,
        files_processed=results,
        total_chars=total_chars,
        message=message,
    )
