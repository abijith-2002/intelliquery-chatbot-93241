# Pinecone Integration (Backend)

This backend uses Pinecone as the vector database for storing and querying embeddings from uploaded documents.

## Key Files
- src/api/main.py
  - Endpoints:
    - POST /chat/upload-context: accepts files, extracts text, chunks content, generates embeddings (Gemini hook), and upserts vectors into Pinecone.
    - POST /chat: embeds the query, searches Pinecone scoped by session_id, and composes a prompt for Gemini (hook) to produce the final answer.
    - POST /chat/title: simple title generation (replace with Gemini call as needed).
  - Embedding hooks:
    - `embed_texts_with_gemini()` is where Gemini embedding should be integrated. Currently a deterministic fallback is provided for local review.

- src/api/vector_store_pinecone.py
  - `PineconeVectorStore` provides upsert, delete, and search methods.
  - Requires the index to exist beforehand. Dimensions must match the embedding model.

- src/api/file_utils.py
  - File parsing for .txt, .pdf, .docx, .xlsx
  - Chunking logic with overlap

- src/api/config_utils.py
  - Loads settings from environment

- .env.example
  - Template showing all necessary environment variables

## Environment Variables
See `.env.example`. At minimum:
- PINECONE_API_KEY
- (Serverless) PINECONE_HOST
  - or (Legacy/provisioned) PINECONE_ENVIRONMENT + PINECONE_INDEX_NAME
- EMBEDDING_DIM (must match your Pinecone index dimension)
- Optional: GEMINI_API_KEY for real embeddings and generation

## Index Creation (Serverless)
Use Pinecone serverless to create an index:

```python
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key="YOUR_PINECONE_API_KEY")
pc.create_index(
    name="your-index-name",
    dimension=768,   # text-embedding-004
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
)
# Obtain the index host from the Pinecone console or API and set PINECONE_HOST in .env.
```

## Embedding Model
- Recommended: Google Gemini `text-embedding-004` (dimension 768)
- Make sure your index `dimension` matches `EMBEDDING_DIM` in .env and the actual model output.

## Flow
1. Upload files to `/chat/upload-context`
   - Extract text
   - Chunk and embed
   - Upsert to Pinecone (with per-chunk metadata including session_id, filename, chunk indices, and text)
2. Ask a question via `/chat`
   - Embed query
   - Query Pinecone filtered by `session_id`
   - Compose context from top matches and call Gemini to produce the final answer (hook)

## Notes
- Current code includes a deterministic fallback embedding to allow local review without a Gemini key. Replace with real Gemini calls for production.
- Handle batching for large uploads and implement retry/backoff as needed.
- Consider metadata size limits—storing the full chunk text in metadata is convenient but increases storage; alternatively store references and retrieve text elsewhere.
