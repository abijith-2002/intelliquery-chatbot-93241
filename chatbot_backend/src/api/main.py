import os
import uuid
import logging
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config_utils import get_settings
from .file_utils import extract_text_from_file, chunk_text
from .vector_store_pinecone import PineconeVectorStore, VectorRecord

# ---------------------------------------------------------------------------------------
# Application setup
# ---------------------------------------------------------------------------------------
app = FastAPI(
    title="IntelliQuery Chatbot API",
    description="FastAPI backend for the IntelliQuery chatbot, providing chat endpoints using user-uploaded context files and Google Gemini for answer generation.",
    version="1.0.0",
)

# CORS configuration (adjust as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production: restrict to your frontend origin(s)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger("uvicorn.error")

# ---------------------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------------------
class UploadedFileResult(BaseModel):
    filename: str
    size: int
    content_chars: int
    preview: str
    error: Optional[str] = None


class UploadContextResponse(BaseModel):
    session_id: str
    files_processed: List[UploadedFileResult]
    total_chars: int
    message: str


class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Unique session identifier for the conversation context.")
    query: str = Field(..., description="User's query (question or message).")


class ChatAnswerResponse(BaseModel):
    answer: str = Field(..., description="Final natural language answer returned by Gemini.")


class TitleRequest(BaseModel):
    prompt: str


# ---------------------------------------------------------------------------------------
# Global services
# ---------------------------------------------------------------------------------------
settings = get_settings()
vector_store = PineconeVectorStore(
    api_key=settings.PINECONE_API_KEY,
    index_name=settings.PINECONE_INDEX_NAME,
    namespace=settings.PINECONE_NAMESPACE,
    host=settings.PINECONE_HOST,
    environment=settings.PINECONE_ENVIRONMENT,
    top_k=settings.PINECONE_TOP_K,
)

# ---------------------------------------------------------------------------------------
# Embedding utilities - HOOKS
# ---------------------------------------------------------------------------------------
def embed_texts_with_gemini(texts: List[str]) -> List[List[float]]:
    """
    HOOK: Gemini embedding logic
    - Integrate with Google's text-embedding-004 (dimension 768 recommended)
    - Ensure the Pinecone index was created with matching dimension and metric (cosine)

    Pseudocode (to implement in real environment):
        import google.generativeai as genai
        genai.configure(api_key=settings.GEMINI_API_KEY)
        model = "text-embedding-004"
        embeddings = []
        for t in texts:
            resp = genai.embed_content(model=model, content=t)
            embeddings.append(resp["embedding"])
        return embeddings

    For now, return placeholder random/sparse vectors if GEMINI_API_KEY is not configured.
    """
    # Fallback stub: deterministic pseudo-embedding (NOT for production)
    import hashlib
    import math
    dim = settings.EMBEDDING_DIM
    out: List[List[float]] = []
    for t in texts:
        h = hashlib.sha256(t.encode("utf-8")).digest()
        # Simple repeat to reach dim length
        vals = []
        while len(vals) < dim:
            for b in h:
                vals.append((b / 255.0) - 0.5)
                if len(vals) >= dim:
                    break
        # L2 normalize (approximate cosine behavior)
        norm = math.sqrt(sum(v * v for v in vals)) or 1.0
        out.append([v / norm for v in vals])
    return out


# ---------------------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------------------
@app.get("/", tags=["Health"], summary="Health Check", description="PUBLIC_INTERFACE\nAPI health check endpoint.\n\nReturns:\n    dict: Status message.")
def health_check():
    return {"status": "ok"}


# ---------------------------------------------------------------------------------------
# Chat title (Gemini-based)
# ---------------------------------------------------------------------------------------
@app.post("/chat/title", tags=["Chat"], summary="Generate a chat title from first prompt")
def generate_title(req: TitleRequest) -> str:
    prompt = (req.prompt or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Blank or invalid input")
    # HOOK: Optionally call Gemini to produce a concise title. For now, simple heuristic.
    title = prompt
    if len(title) > 60:
        title = title[:60] + "…"
    # Trim trailing punctuation
    if title.endswith("?") or title.endswith("."):
        title = title[:-1]
    # Capitalize
    if title:
        title = title[0].upper() + title[1:]
    return title or "Chat"


# ---------------------------------------------------------------------------------------
# Upload context: parse, chunk, embed, upsert to Pinecone
# ---------------------------------------------------------------------------------------
@app.post(
    "/chat/upload-context",
    tags=["Chat"],
    summary="Upload context files for a chat session",
    response_model=UploadContextResponse,
)
async def upload_chat_context(
    session_id: str = Form(..., description="Session ID to associate uploaded context with"),
    files: List[UploadFile] = File(..., description="One or more files (.docx, .xlsx, .pdf, .txt)"),
):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    processed: List[UploadedFileResult] = []
    total_chars = 0
    all_chunks: List[str] = []
    chunk_metadatas: List[Dict[str, Any]] = []

    for f in files:
        try:
            content = await f.read()
            text = extract_text_from_file(filename=f.filename, data=content)
            chars = len(text)
            total_chars += chars

            # Chunking
            chunks = chunk_text(text, chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP)
            all_chunks.extend(chunks)

            # Prepare metadata per chunk
            for idx, chunk in enumerate(chunks):
                meta = {
                    "session_id": session_id,
                    "filename": f.filename,
                    "chunk_index": idx,
                    "total_chunks": len(chunks),
                }
                chunk_metadatas.append(meta)

            # File-level preview (first 300 chars)
            preview = (text[:300] + "…") if len(text) > 300 else text
            processed.append(
                UploadedFileResult(
                    filename=f.filename,
                    size=len(content),
                    content_chars=chars,
                    preview=preview,
                )
            )
        except Exception as e:
            logger.exception("Failed to process file: %s", f.filename)
            processed.append(
                UploadedFileResult(
                    filename=f.filename,
                    size=0,
                    content_chars=0,
                    preview="",
                    error=str(e),
                )
            )

    # Filter out failed files' chunks by matching filenames in processed with no error
    # In this simplified flow, we already included chunks only for successfully processed files.
    if all_chunks:
        embeddings = embed_texts_with_gemini(all_chunks)

        # Build vector records for Pinecone upsert
        records: List[VectorRecord] = []
        for i, (chunk, meta) in enumerate(zip(all_chunks, chunk_metadatas)):
            vid = f"{session_id}::{meta.get('filename','unknown')}::{meta.get('chunk_index', i)}::{uuid.uuid4().hex}"
            records.append(
                VectorRecord(
                    id=vid,
                    values=embeddings[i],
                    metadata={**meta, "text": chunk},
                )
            )

        try:
            vector_store.upsert(records)
        except Exception as e:
            logger.exception("Upsert to Pinecone failed")
            # Soft-fail: mark all as error if desired, or bubble up
            raise HTTPException(status_code=500, detail=f"Pinecone upsert failed: {e}")

    return UploadContextResponse(
        session_id=session_id,
        files_processed=processed,
        total_chars=total_chars,
        message="Context uploaded and indexed into Pinecone" if all_chunks else "No valid content to index",
    )


# ---------------------------------------------------------------------------------------
# Chat: retrieve context via Pinecone search, then call Gemini to answer
# ---------------------------------------------------------------------------------------
@app.post(
    "/chat",
    tags=["Chat"],
    summary="Chat with bot",
    response_model=ChatAnswerResponse,
)
def chat(req: ChatRequest):
    session_id = (req.session_id or "").strip()
    query = (req.query or "").strip()
    if not session_id or not query:
        raise HTTPException(status_code=422, detail="session_id and query are required")

    # Embed the query
    query_embedding = embed_texts_with_gemini([query])[0]

    # Search Pinecone constrained by session_id
    try:
        results = vector_store.search(
            vector=query_embedding,
            top_k=settings.PINECONE_TOP_K,
            filter={"session_id": {"$eq": session_id}},
            include_metadata=True,
        )
    except Exception as e:
        logger.exception("Pinecone search failed")
        raise HTTPException(status_code=500, detail=f"Pinecone search failed: {e}")

    # Build context string from top results
    context_segments: List[str] = []
    for match in results.matches:
        md = match.metadata or {}
        text = md.get("text", "")
        fname = md.get("filename", "file")
        idx = md.get("chunk_index", 0)
        context_segments.append(f"[{fname}#chunk-{idx}] {text}")

    context_text = "\n\n".join(context_segments)

    # HOOK: Call Gemini with query + context_text to get final answer.
    # For review purposes, we implement a simple heuristic fallback answer.
    # Replace the following with a real Gemini call:
    #
    # pseudo:
    #   genai.configure(api_key=settings.GEMINI_API_KEY)
    #   prompt = f"Context:\n{context_text}\n\nQuestion:\n{query}\n\nAnswer comprehensively using only the context when possible."
    #   model = genai.GenerativeModel("gemini-pro")
    #   resp = model.generate_content(prompt)
    #   answer_text = resp.text
    #
    # TODO: Implement and handle errors, rate limits, etc.
    if context_text.strip():
        answer_text = f"(Using uploaded context) Answer to: {query}\n\nKey references:\n{context_text[:1000]}"
    else:
        answer_text = f"(No uploaded context found for this session) Answer to: {query}\n\nPlease upload files for better results."

    return ChatAnswerResponse(answer=answer_text)
