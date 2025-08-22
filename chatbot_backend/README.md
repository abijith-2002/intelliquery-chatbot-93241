# IntelliQuery Chatbot Backend

FastAPI backend integrating Pinecone for vector storage and search, supporting:
- Uploading large DOCX/PDF/TXT/XLSX files
- Chunking and embedding (Gemini hook with fallback)
- Pinecone vector upsert and semantic retrieval
- Chat endpoint that uses retrieved context to produce answers

## Endpoints
- GET `/` health check
- POST `/chat/upload-context` — upload files, extract text, chunk, embed, and index in Pinecone
- POST `/chat` — query Pinecone using the session_id to retrieve relevant context and produce an answer
- POST `/chat/title` — generate a short title from the first prompt

## Setup
1. Create a Pinecone index with dimension matching your embedding model (e.g., 768 for Gemini text-embedding-004), metric cosine, serverless preferred.
2. Copy `.env.example` to `.env` and fill in:
   - PINECONE_API_KEY
   - PINECONE_HOST (serverless) or PINECONE_ENVIRONMENT + PINECONE_INDEX_NAME
   - EMBEDDING_DIM=768
   - Optionally GEMINI_API_KEY
3. Install dependencies from `requirements.txt`
4. Run the app with:
   uvicorn src.api.main:app --reload --port 8000

## Notes
- `embed_texts_with_gemini()` contains the hook where Gemini embedding should be implemented.
- The chat answer generation currently uses a placeholder; integrate Gemini generation for production.
