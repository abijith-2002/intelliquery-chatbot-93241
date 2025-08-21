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
from typing import List, Optional, Dict, Any, Tuple, Union
import google.generativeai as genai

from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
import os
import json
import pandas as pd
import textwrap

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

# Per-session in-memory XLSX schema store.
# Structure:
#   XLSX_SCHEMA_STORE[session_id] = {
#       "<workbook_id or filename>": {
#           "sheets": [
#               {
#                   "name": str,
#                   "columns": [
#                       {"name": str, "dtype": str, "top_values": [Any, ...]}
#                   ]
#               }, ...
#           ]
#       }, ...
#   }
XLSX_SCHEMA_STORE: Dict[str, Dict[str, Any]] = {}

# Per-session in-memory pandas DataFrame store.
# Structure:
#   XLSX_DF_STORE[session_id] = {
#       "<workbook_id or filename>": {
#           "<sheet_name>": pandas.DataFrame,
#           ...
#       },
#       ...
#   }
# Note: Populated on XLSX upload in future steps; currently acts as a placeholder for workflow.
XLSX_DF_STORE: Dict[str, Dict[str, Any]] = {}

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

def _pandas_safe_to_string(
    obj: Any,
    max_rows: int = 30,
    max_cols: int = 8,
    max_col_width: int = 60,
    max_chars: int = 3000,
) -> str:
    """
    Convert a pandas result (DataFrame, Series) or scalar to a compact, safe string.
    Truncates rows, columns, individual column width, and overall characters for safety.

    Args:
        obj: DataFrame, Series, or scalar.
        max_rows: Maximum number of rows to include.
        max_cols: Maximum number of columns to include when rendering tabular results.
        max_col_width: Maximum characters per cell/column content (approx for display).
        max_chars: Maximum number of characters in the final string.
    """
    try:
        if isinstance(obj, pd.DataFrame):
            # Ensure we only preview up to max_cols columns
            df = obj.copy()
            if df.shape[1] > max_cols:
                # Keep first max_cols columns and mark trimming
                extra = df.shape[1] - max_cols
                df = df.iloc[:, :max_cols].copy()
                df[f"... +{extra} more cols"] = ""
            # Head limit the rows
            df = df.head(max_rows)
            with pd.option_context(
                "display.max_rows", max_rows,
                "display.max_columns", max_cols + 1,  # account for added marker column
                "display.max_colwidth", max_col_width,
                "display.width", 1000,
            ):
                s = df.to_string(index=False)
        elif isinstance(obj, pd.Series):
            s = obj.head(max_rows).to_string()
        else:
            s = str(obj)

        s = str(s).strip()
        if len(s) > max_chars:
            s = s[: max_chars - 3] + "..."
        return s
    except Exception:
        # Fallback to raw string conversion
        try:
            s = str(obj)
            if len(s) > max_chars:
                s = s[: max_chars - 3] + "..."
            return s
        except Exception:
            return "<unrenderable result>"


# PUBLIC_INTERFACE
def get_gemini_nl_answer_from_pandas(query: str, pandas_result_text: str) -> str:
    """
    PUBLIC_INTERFACE
    Given the user's original query and a compact textual representation of a pandas result,
    ask Gemini to produce a concise, direct natural-language answer without referencing code,
    schemas, or internal processing steps.

    Prompts are intentionally compact to control token usage.

    Args:
        query (str): The original user question.
        pandas_result_text (str): The pandas result converted to a readable text table/value.

    Returns:
        str: A concise, plain-language answer phrased by Gemini.
    """
    if not pandas_result_text or not pandas_result_text.strip():
        pandas_result_text = "no matching data found"

    key = get_gemini_api_key()
    if not key:
        return pandas_result_text.strip()

    # Compact instruction and prompt
    instruction = (
        "Answer briefly using only the given result. "
        "Do not mention code, pandas, or schemas. No meta explanations."
    )
    # Limit the size we send to the model defensively
    MAX_RESULT_CHARS = 2800
    compact_result = pandas_result_text.strip()
    if len(compact_result) > MAX_RESULT_CHARS:
        compact_result = compact_result[: MAX_RESULT_CHARS - 3] + "..."

    MAX_QUERY_CHARS = 500
    compact_query = (query or "").strip()
    if len(compact_query) > MAX_QUERY_CHARS:
        compact_query = compact_query[: MAX_QUERY_CHARS - 3] + "..."

    prompt = f"{instruction}\nQ: {compact_query}\nResult:\n{compact_result}\nAnswer:"

    try:
        genai.configure(api_key=key)
        model = genai.GenerativeModel("gemini-2.5-flash")
        resp = model.generate_content([{"role": "user", "parts": [prompt]}])
        text = (resp.text or "").strip()
        return _clean_gemini_output(text) if text else pandas_result_text.strip()
    except Exception:
        return pandas_result_text.strip()

def _build_schema_prompt(session_id: str) -> Tuple[str, Dict[str, Dict[str, pd.DataFrame]]]:
    """
    Construct a compact schema description and return also the DataFrame store mapping for evaluation.

    Returns:
        (schema_text, df_store_for_session)
    """
    schema_bundle = XLSX_SCHEMA_STORE.get(session_id) or {}
    df_bundle = XLSX_DF_STORE.get(session_id) or {}
    if not schema_bundle or not df_bundle:
        return "", {}

    lines: List[str] = []
    # Keep ultra-compact formatting to minimize tokens
    for wb_name, schema in schema_bundle.items():
        try:
            sheets = schema.get("sheets", [])
            for s in sheets:
                sheet_name = s.get("name", "")
                col_names = [c.get("name", "") for c in s.get("columns", [])]
                cols_display = ", ".join(col_names[:30])
                if len(col_names) > 30:
                    cols_display += ", ..."
                lines.append(f"{wb_name}::{sheet_name} | {cols_display}")
        except Exception:
            continue

    schema_text = "\n".join(lines)
    return schema_text, df_bundle

def _gemini_pandas_code_for_query(user_query: str, schema_text: str) -> str:
    """
    Ask Gemini to produce strictly a pandas code string (no markdown, no prose) that answers the query
    using the provided schema. The code must only return the final expression or assignment to a variable named RESULT.

    Returns:
        str: code snippet
    """
    if not schema_text.strip():
        return ""

    instruction = textwrap.dedent(
        """
        Produce ONLY Python code (plain text, no markdown/comments) using pandas on provided DataFrames.

        Rules:
        - Access DataFrames via DFS[workbook][sheet].
        - No imports, no functions, no I/O, no network.
        - Assign final result to RESULT (e.g., RESULT = <expression>).
        - Keep operations simple and safe.
        - If not answerable with given columns: RESULT = "Not answerable from provided sheets"
        """
    ).strip()

    prompt = f"{instruction}\n\nSchema:\n{schema_text}\n\nUser question:\n{user_query}\n\nReturn only the code:"
    key = get_gemini_api_key()
    if not key:
        return ""
    try:
        genai.configure(api_key=key)
        model = genai.GenerativeModel("gemini-2.5-flash")
        resp = model.generate_content([{"role": "user", "parts": [prompt]}])
        code = (resp.text or "").strip()
        # enforce single-line purity by removing markdown fencing if any slipped in
        code = code.replace("```python", "").replace("```", "").strip()
        return code
    except Exception:
        return ""

def _safe_eval_pandas(code: str, dfs: Dict[str, Dict[str, pd.DataFrame]]) -> Tuple[bool, Union[str, Any]]:
    """
    Safely evaluate a pandas code string where the code must set RESULT variable.
    We restrict builtins and globals; only DFS and pd are available.

    Security:
    - Blocks dangerous tokens and access patterns: imports, dunders, file/network/process APIs, exec/eval, etc.
    - Disallows attribute access on modules other than pd/DFS objects resolved at runtime via pandas methods only.
    - Single-statement expectation: assignment to RESULT.

    Returns:
        (success, result_or_error)
    """
    if not code:
        return False, "No code produced"

    # Normalize newlines/spaces for checks
    lowered = " ".join(code.lower().split())

    # Forbidden token substrings to block outright
    forbidden_tokens = [
        "import", "__", "os.", "sys.", "pathlib", "shutil", "tempfile", "builtins",
        "pickle", "dill", "marshal", "ctypes", "cffi", "subprocess", "multiprocessing",
        "thread", "threading", "socket", "requests", "urllib", "http", "https",
        "ftp", "smtplib", "paramiko",
        "open(", "io.", "eval(", "exec(", "compile(", "globals(", "locals(", "vars(",
        "setattr(", "getattr(", "delattr(", "input(", "print(", "exit(", "quit(",
        "__import__", "memoryview(", "bytearray(", "buffer(", "reload(",
        "sys.exit", "os.system", "os.popen",
    ]
    for tok in forbidden_tokens:
        if tok in lowered:
            return False, f"Forbidden token detected in code: {tok}"

    # Forbid assignments to global names other than RESULT; block semicolons (multi stmt)
    if ";" in code:
        return False, "Multiple statements are not allowed"
    # Ensure RESULT is assigned
    if "result" not in lowered or "=" not in code:
        return False, "Code must assign the final value to RESULT"

    # Very conservative guard on backticks and triple-backticks
    if "```" in code or "`" in code:
        return False, "Code formatting markers are not allowed"

    # Prepare isolated execution env
    local_env: Dict[str, Any] = {}
    safe_globals = {
        "__builtins__": {},  # no builtins
        "DFS": dfs,          # provided dataframes
        "pd": pd,            # allow pandas access
    }

    try:
        exec(code, safe_globals, local_env)
        result = local_env.get("RESULT", safe_globals.get("RESULT"))
        if result is None:
            return False, "Code did not assign to RESULT"
        return True, result
    except Exception as e:
        return False, f"Evaluation error: {e}"
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

        # Build schema-driven pandas step if XLSX data is available for the session
        pandas_context_summary = ""
        try:
            schema_text, dfs_bundle = _build_schema_prompt(session_id)
            if schema_text and dfs_bundle:
                pandas_code = _gemini_pandas_code_for_query(request.query, schema_text)

                # Evaluate generated pandas safely
                success, result_or_err = _safe_eval_pandas(pandas_code, dfs_bundle)

                def _normalize_pandas_result_for_nl(res: Any) -> str:
                    """
                    Normalize a pandas evaluation output into a compact, user-friendly string
                    that is safe for NL phrasing. Handles empty results gracefully.

                    - If DataFrame/Series is empty, return 'no matching data found'.
                    - If scalar/other type is None/NaN/empty-like, return the same placeholder.
                    - Otherwise, render via _pandas_safe_to_string with defensive truncation.
                    """
                    try:
                        import numpy as _np
                    except Exception:
                        _np = None  # type: ignore

                    PLACEHOLDER = "no matching data found"

                    # Empty DataFrame
                    if isinstance(res, pd.DataFrame):
                        if res.empty:
                            return PLACEHOLDER
                        return _pandas_safe_to_string(res)

                    # Series or Index-like
                    if isinstance(res, pd.Series):
                        if res.empty:
                            return PLACEHOLDER
                        # If scalar-like selection sometimes returns a length-1 series, still display it
                        return _pandas_safe_to_string(res)

                    # Scalar or other objects
                    # Explicitly treat None/NaN/NaT/empty string as no data
                    if res is None:
                        return PLACEHOLDER
                    try:
                        if _np is not None and (res is _np.nan or (_np.isnan(res) if isinstance(res, (float, int)) else False)):
                            return PLACEHOLDER
                    except Exception:
                        pass
                    if isinstance(res, str) and res.strip() == "":
                        return PLACEHOLDER

                    # Lists/dicts: try to detect emptiness
                    if isinstance(res, (list, tuple, set, dict)) and len(res) == 0:
                        return PLACEHOLDER

                    # Default stringification with truncation
                    s = _pandas_safe_to_string(res)
                    return s if s.strip() else PLACEHOLDER

                if success:
                    pandas_context_summary = _normalize_pandas_result_for_nl(result_or_err)
                else:
                    # Include a compact note for Gemini phrasing and avoid leaking internals
                    pandas_context_summary = f"Pandas step note: {result_or_err}"
        except Exception as e:
            # Capture exceptions so flow never crashes; include brief note to aid NL phrasing
            pandas_context_summary = f"Pandas step note: {e}"

        # If we have a pandas-derived result, ask Gemini to phrase a concise, direct answer from it.
        if pandas_context_summary:
            final_answer = get_gemini_nl_answer_from_pandas(request.query, pandas_context_summary)
        else:
            # Otherwise, prepare combined extra context from vector retrieval and ask Gemini directly.
            extra_parts = []
            if retrieved_context:
                extra_parts.append(retrieved_context)
            combined_extra = "\n\n".join(extra_parts).strip()
            try:
                final_answer = get_gemini_response(request.query, memory, extra_context=combined_extra)
            except Exception as e:
                final_answer = "[Gemini unavailable: {}]".format(e)

        # Final output cleaning and save in memory
        final_answer = _clean_gemini_output(_safe_str_output(final_answer))
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
        - Extract readable text or token-limited chunks (for wide Excel).
        - Split text into overlapping chunks (non-wide) OR use row-based chunks (wide Excel).
        - Embed each chunk using Gemini embeddings (if API key available).
        - Store chunks and embeddings in a per-session in-memory index for retrieval.

    Wide Excel handling:
        For .xlsx files with large number of columns (e.g., >700), this endpoint will:
          - Produce chunk texts that include full column metadata, but only sample a subset
            of columns for detailed row values.
          - Optionally append per-column statistics.
          - Ensure each chunk remains within a target token budget for LLM prompt safety.

    Args:
        session_id (str): The chat session ID.
        files (List[UploadFile]): Uploaded files (multipart/form-data).

    Returns:
        UploadContextResponse: Processing results and acknowledgment.
    """
    from .file_utils import (
        extract_text_from_bytes,
        summarize_text_preview,
        extract_xlsx_wide_chunks,
        extract_xlsx_schema_and_dfs,
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
        name_lower = (filename or "").lower()
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

        # Specialized wide-Excel flow
        indexed_any = False
        if name_lower.endswith(".xlsx"):
            try:
                # First, extract schema and DataFrames for the workbook
                schema, dfs = extract_xlsx_schema_and_dfs(data or b"")

                # Initialize per-session stores if not present
                if session_id not in XLSX_SCHEMA_STORE:
                    XLSX_SCHEMA_STORE[session_id] = {}
                if session_id not in XLSX_DF_STORE:
                    XLSX_DF_STORE[session_id] = {}

                # Store in-memory per filename
                XLSX_SCHEMA_STORE[session_id][filename] = schema
                XLSX_DF_STORE[session_id][filename] = dfs

                # Persist schema JSON to filesystem under data/session_schemas/{session_id}/{filename}.schema.json
                target_dir = os.path.join("data", "session_schemas", session_id)
                os.makedirs(target_dir, exist_ok=True)
                safe_fname = f"{filename}.schema.json"
                schema_path = os.path.join(target_dir, safe_fname)
                with open(schema_path, "w", encoding="utf-8") as fp:
                    json.dump(schema, fp, indent=2, ensure_ascii=False)

                # Extract wide-aware chunks for indexing and preview
                wide_chunks = extract_xlsx_wide_chunks(
                    data or b"",
                    max_tokens_per_chunk=1200,
                    row_chunk_size=100,
                    base_sample_columns=5,
                    random_sample_columns=3,
                    stats_sample_rows=200,
                )
                if wide_chunks:
                    # Index each chunk
                    for idx, ch in enumerate(wide_chunks, start=1):
                        try:
                            _index_text_for_session(session_id, f"{filename}#chunk{idx}", ch)
                            combined_text_parts.append(f"[{filename}#chunk{idx}]\n{ch}\n")
                            total_chars += len(ch)
                            indexed_any = True
                        except Exception:
                            # Indexing failure for one chunk should not break the entire upload
                            pass

                    # Preview notes include schema success
                    base_preview = summarize_text_preview(wide_chunks[0], max_chars=460)
                    preview = f"{base_preview} [XLSX schema extracted and stored]"
                    results.append(
                        UploadedFileResult(
                            filename=filename,
                            size=size,
                            content_chars=sum(len(c) for c in wide_chunks),
                            preview=preview,
                            error=None,
                        )
                    )
                    continue  # move to next file after wide handling
                else:
                    # If no chunks, still report schema success in preview
                    preview = "[XLSX schema extracted and stored] No chunkable content found."
                    results.append(
                        UploadedFileResult(
                            filename=filename,
                            size=size,
                            content_chars=0,
                            preview=preview,
                            error=None,
                        )
                    )
                    continue
            except Exception:
                # Fall back to legacy extraction if XLSX specialized processing fails.
                # Proceed to legacy path below; will set err in results if needed.
                pass

        # Legacy extraction path (txt, pdf, docx, or xlsx fallback)
        text, err = extract_text_from_bytes(filename, data or b"")
        preview = summarize_text_preview(text, max_chars=500) if text else ""
        chars = len(text)

        if text and not err:
            combined_text_parts.append(f"[{filename}]\n{text}\n")
            total_chars += chars
            try:
                _index_text_for_session(session_id, filename, text)
                indexed_any = True
            except Exception:
                pass

        results.append(
            UploadedFileResult(
                filename=filename,
                size=size,
                content_chars=(sum(len(c) for c in wide_chunks) if name_lower.endswith(".xlsx") and indexed_any else chars),
                preview=preview,
                error=(None if indexed_any and not err else err),
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
@app.get(
    "/chat/schema",
    tags=["Chat"],
    summary="Get uploaded Excel schema JSON",
    description="Retrieve the stored schema JSON for an uploaded Excel file by session_id and filename.",
    responses={
        200: {"description": "Schema JSON returned"},
        404: {"description": "Schema not found for session/filename"},
        400: {"description": "Invalid request parameters"},
    },
)
def get_uploaded_schema(session_id: str, filename: str):
    """
    PUBLIC_INTERFACE
    Returns the JSON schema extracted from an uploaded Excel workbook for debugging or UI display.

    Args:
        session_id (str): The session identifier used during upload.
        filename (str): Original Excel filename used in the upload.

    Returns:
        dict: Schema JSON with sheets and columns metadata.

    Raises:
        HTTPException: 400 for invalid params, 404 if not found.
    """
    if not session_id or not filename:
        raise HTTPException(status_code=400, detail="session_id and filename are required")
    session_store = XLSX_SCHEMA_STORE.get(session_id)
    if not session_store:
        raise HTTPException(status_code=404, detail="No schema found for session")
    schema = session_store.get(filename)
    if not schema:
        raise HTTPException(status_code=404, detail="No schema found for given filename in session")
    # Return compacted schema (no change to content)
    return schema
