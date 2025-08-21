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
# Import new Excel processing utilities
from .file_utils import process_excel_for_session

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

# Per-session Excel metadata and Parquet file storage.
# Structure: { session_id: [metadata_dict1, metadata_dict2, ...] }
EXCEL_METADATA_STORE: Dict[str, List[Dict[str, Any]]] = {}

# PUBLIC_INTERFACE
def _load_session_metadata_from_disk(session_id: str) -> List[Dict[str, Any]]:
    """
    PUBLIC_INTERFACE
    Attempt to load Excel metadata for a session from disk when in-memory store is empty.

    Args:
        session_id (str): The session identifier.

    Returns:
        List[Dict[str, Any]]: List of Excel metadata dictionaries, or empty list if none found.
    """
    try:
        from .file_utils import get_session_excel_metadata
        metas = get_session_excel_metadata(session_id)
        if metas:
            EXCEL_METADATA_STORE[session_id] = metas
        return metas
    except Exception:
        return []

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


class ExcelQuestionRequest(BaseModel):
    """Schema for Excel-specific question requests."""
    session_id: str = Field(..., description="Session ID containing uploaded Excel files")
    question: str = Field(..., description="Question about the Excel data")


class ExcelQuestionResponse(BaseModel):
    """Schema for Excel question responses."""
    answer: str = Field(..., description="Answer based on Excel data analysis")
    data_context: Optional[str] = Field(default=None, description="Relevant data context used for the answer")
    metadata_used: List[str] = Field(default=[], description="List of Excel files/sheets used in the analysis")


def _clean_gemini_output(text: str) -> str:
    """
    Removes leading/trailing meta, KB source, or disclaimer information from Gemini output.
    Ensures only direct answers are delivered to the user.
    This implementation safely compiles regex patterns to avoid crashes from malformed patterns.
    """
    import re

    def _safe_compile(pattern: str, flags=0):
        try:
            return re.compile(pattern, flags)
        except re.error:
            # If a pattern is malformed, skip it rather than raising at runtime.
            return None

    # Define patterns as raw strings, carefully escaping parentheses when used literally.
    meta_pattern_strs = [
        r"^ *(?:based on (?:the )?(?:provided )?(?:knowledge ?base|context|sources)[^:]*:?)",
        r"^(?:as an?\s+[^\s:,]+ [^\s:,]+,?)[\s:.\-]*",
        r"^ *(?:this information[^:]*:?)",
        r"^ *(?:note:)[^\n\r]*",
        r"^ *(?:please note)[^\n\r]*",
        r"^ *(?:source[sd]?:)[^\n\r]*",
        r"^ *(?:from the knowledge base[^:]*:?)",
        r"^ *(?:provided context[^:]*:?)",
        # Leading parenthetical meta like "(based on ...)" or "(as an ai language model ...)"
        r"^\(?(?:based on|as an ai language model|this information|provided context)[^\)]*\)?",
    ]
    trail_pattern_strs = [
        r"\(? *(?:based on (?:the )?(?:provided )?(?:knowledge ?base|context|sources)[^)]*\)?[.!]? *$",
        r"\(? *(?:from the knowledge base)[^)]*\)?[.!]? *$",
        r"\(? *(?:provided context)[^)]*\)?[.!]? *$",
    ]

    meta_patterns = [
        p for p in ( _safe_compile(s, re.IGNORECASE | re.MULTILINE) for s in meta_pattern_strs ) if p is not None
    ]
    trail_patterns = [
        p for p in ( _safe_compile(s, re.IGNORECASE | re.MULTILINE) for s in trail_pattern_strs ) if p is not None
    ]

    clean_text = text
    for pat in meta_patterns:
        clean_text = pat.sub("", clean_text)
    for pat in trail_patterns:
        clean_text = pat.sub("", clean_text)
    return clean_text.strip()


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


def _create_excel_text_summary(metadata: Dict[str, Any]) -> str:
    """Create a text summary from Excel metadata for embedding and retrieval."""
    summary_parts = [f"Excel file: {metadata.get('filename', 'Unknown')}"]
    
    for sheet_name, sheet_info in metadata.get('sheets', {}).items():
        summary_parts.append(f"\nSheet: {sheet_name}")
        summary_parts.append(f"Rows: {sheet_info.get('row_count', 0)}, Columns: {sheet_info.get('column_count', 0)}")
        
        # Add column information
        columns_info = []
        for col_name, col_data in sheet_info.get('columns', {}).items():
            col_desc = f"{col_name} ({col_data.get('dtype', 'unknown')})"
            if col_data.get('sample_values'):
                sample_str = ', '.join(str(v) for v in col_data['sample_values'][:3])
                col_desc += f" - examples: {sample_str}"
            columns_info.append(col_desc)
        
        if columns_info:
            summary_parts.append(f"Columns: {'; '.join(columns_info[:10])}")  # Limit to first 10 columns
    
    return '\n'.join(summary_parts)


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
        "Excel files are processed with pandas and stored as Parquet for structured querying. "
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
        - For Excel files: process with pandas, store as Parquet, extract metadata.

    Args:
        session_id (str): The chat session ID.
        files (List[UploadFile]): Uploaded files (multipart/form-data).

    Returns:
        UploadContextResponse: Processing results and acknowledgment.
    """
    from .file_utils import extract_text_from_bytes, summarize_text_preview

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
        
        # Handle Excel files with structured processing
        if filename.lower().endswith('.xlsx'):
            try:
                excel_metadata, excel_err = process_excel_for_session(filename, data or b"", session_id)
                if excel_metadata and not excel_err:
                    # Store Excel metadata for the session
                    if session_id not in EXCEL_METADATA_STORE:
                        EXCEL_METADATA_STORE[session_id] = []
                    EXCEL_METADATA_STORE[session_id].append(excel_metadata)

                    # Create a summary for text indexing
                    text_summary = _create_excel_text_summary(excel_metadata)
                    combined_text_parts.append(f"[{filename} - Excel Data Summary]\n{text_summary}\n")
                    total_chars += len(text_summary)

                    # Index the summary for retrieval
                    try:
                        _index_text_for_session(session_id, filename, text_summary)
                    except Exception:
                        pass

                    preview = f"Excel file with {excel_metadata.get('total_rows', 0)} total rows across {len(excel_metadata.get('sheets', {}))} sheets"
                    chars = len(text_summary)
                    err = None
                else:
                    # Log details for debugging empty/failed Excel reads
                    try:
                        import logging as _logging
                        _logging.getLogger(__name__).warning(
                            "Excel processing yielded no data for '%s' (session %s). Error: %s | Metadata sheets: %s",
                            filename, session_id, excel_err, list((excel_metadata or {}).get('sheets', {}).keys())
                        )
                    except Exception:
                        pass
                    preview = ""
                    chars = 0
                    err = excel_err
            except Exception as e:
                preview = ""
                chars = 0
                err = f"Excel processing failed: {e}"
        else:
            # Regular text extraction for non-Excel files
            text, err = extract_text_from_bytes(filename, data or b"")
            preview = summarize_text_preview(text, max_chars=500) if text else ""
            chars = len(text)

            # Append to combined only if successful and non-empty
            if text and not err:
                combined_text_parts.append(f"[{filename}]\n{text}\n")
                total_chars += chars

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


# PUBLIC_INTERFACE
@app.post(
    "/chat/ask-excel-question",
    response_model=ExcelQuestionResponse,
    tags=["Chat"],
    summary="Ask questions about uploaded Excel data",
    description=(
        "Submit questions specifically about Excel data uploaded for the session. "
        "Uses structured metadata and Parquet data for precise analysis. "
        "Returns answers based on the actual Excel data structure and content."
    ),
    responses={
        400: {"description": "No Excel data found for session or invalid request"},
        500: {"description": "Error processing Excel question"},
    },
)
def ask_excel_question(request: ExcelQuestionRequest):
    """
    PUBLIC_INTERFACE
    Handle questions specifically about Excel data using structured metadata and Parquet files.
    
    This endpoint provides more precise answers for Excel-specific queries by leveraging
    the structured metadata extracted during upload and stored Parquet data.
    
    Args:
        request (ExcelQuestionRequest): The Excel question request with session_id and question.
    
    Returns:
        ExcelQuestionResponse: Answer with data context and metadata information.
    """
    session_id = request.session_id
    question = request.question
    
    # Get Excel metadata for the session
    excel_metadata_list = EXCEL_METADATA_STORE.get(session_id, [])
    # Lazy-load from disk if memory store is empty (e.g., after restart)
    if not excel_metadata_list:
        excel_metadata_list = _load_session_metadata_from_disk(session_id)
    if not excel_metadata_list:
        raise HTTPException(
            status_code=400, 
            detail="No Excel data found for this session. Please upload Excel files first."
        )
    
    try:
        # Prepare context from Excel metadata
        data_context_parts = []
        metadata_used = []
        
        for metadata in excel_metadata_list:
            filename = metadata.get('filename', 'Unknown')
            metadata_used.append(filename)
            
            # Add file summary
            data_context_parts.append(f"\n=== {filename} ===")
            data_context_parts.append(f"Total sheets: {len(metadata.get('sheets', {}))}")
            data_context_parts.append(f"Total rows: {metadata.get('total_rows', 0)}")
            data_context_parts.append(f"Total columns: {metadata.get('total_columns', 0)}")
            
            # Add detailed sheet information
            for sheet_name, sheet_info in metadata.get('sheets', {}).items():
                data_context_parts.append(f"\nSheet '{sheet_name}':")
                data_context_parts.append(f"  - {sheet_info.get('row_count', 0)} rows, {sheet_info.get('column_count', 0)} columns")
                
                # Add column details
                columns = sheet_info.get('columns', {})
                if columns:
                    data_context_parts.append("  - Columns:")
                    for col_name, col_data in list(columns.items())[:10]:  # Limit to first 10 columns
                        col_type = col_data.get('dtype', 'unknown')
                        sample_vals = col_data.get('sample_values', [])
                        sample_str = ', '.join(str(v) for v in sample_vals[:3]) if sample_vals else 'No samples'
                        data_context_parts.append(f"    * {col_name} ({col_type}): {sample_str}")
                # Include a compact view of first rows if available
                sample_rows = sheet_info.get('sample_data', {}).get('first_5_rows', [])
                if sample_rows:
                    data_context_parts.append("  - Sample rows (up to 3):")
                    for row in sample_rows[:3]:
                        data_context_parts.append(f"    • {row}")
        
        data_context = '\n'.join(data_context_parts)
        
        # Generate answer using Gemini with Excel-specific context
        excel_prompt = (
            f"You are a data analyst. Answer the following question about Excel data:\n"
            f"Question: {question}\n\n"
            f"Available Excel Data Context:\n{data_context}\n\n"
            f"Please provide a clear, specific answer based on the Excel data structure and content shown above. "
            f"If the question requires specific data values that aren't shown in the context, "
            f"explain what information is available and suggest how to get the specific data needed."
        )
        
        gemini_api_key = get_gemini_api_key()
        if not gemini_api_key:
            raise HTTPException(status_code=500, detail="Gemini API key not configured")
        
        try:
            import google.generativeai as genai
            genai.configure(api_key=gemini_api_key)
            model = genai.GenerativeModel("gemini-2.5-flash")
            response = model.generate_content([{"role": "user", "parts": [excel_prompt]}])
            answer = response.text.strip()
        except Exception as e:
            answer = f"Unable to generate answer using AI: {e}. Based on the uploaded Excel data, I can see {len(excel_metadata_list)} file(s) with a total of {sum(m.get('total_rows', 0) for m in excel_metadata_list)} rows across {sum(len(m.get('sheets', {})) for m in excel_metadata_list)} sheets."
        
        return ExcelQuestionResponse(
            answer=answer,
            data_context=data_context[:2000] if len(data_context) > 2000 else data_context,  # Truncate if too long
            metadata_used=metadata_used
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing Excel question: {e}")
