# Pinecone Integration Plan (Replacing Supabase/pgvector)

This document outlines the changes required to use Pinecone as the vector database for semantic search, replacing any Supabase/pgvector usage. It covers backend integration points in the FastAPI service, required environment variables, client library requirements, index configuration, and operational considerations.

1) Summary of Changes
- Remove Supabase/pgvector references. No Postgres vector table will be used for retrieval.
- Use Pinecone as the centralized vector store for storing and querying text chunk embeddings.
- The backend will continue to use Google Gemini embeddings (models/text-embedding-004) to generate vectors, but it will store/query these vectors in Pinecone.
- Keep an in-memory fallback (existing behavior) if Pinecone or embeddings are unavailable.
- No changes to the public API contract are required. Retrieval behavior is internal.

2) Backend Client Requirements
- Library: pinecone-client (Python)
- Recommended helper: backoff for retrying idempotent Pinecone calls during rate limits and transient errors.

3) Environment Variables
Set the following variables in the backend environment (.env). Do not hardcode:
- PINECONE_API_KEY: Pinecone API key (required).
- PINECONE_ENVIRONMENT or PINECONE_HOST: Depending on Pinecone account/region setup. For serverless projects, use PINECONE_HOST. If using legacy indexes, PINECONE_ENVIRONMENT may be required. Only set what your account requires.
- PINECONE_INDEX_NAME: The index to use (e.g., intelliquery-chatbot).
- PINECONE_NAMESPACE: Optional namespace to scope data per environment or tenant (default: default).
- PINECONE_TOP_K: Optional top-k override for retrieval (default: 3).
- Optional Gemini key variables remain as implemented: GEMINI_API_KEY, REACT_APP_GEMINI_API_KEY, GOOGLE_API_KEY, GOOGLE_GEMINI_API_KEY.

Note: The frontend's existing variables (REACT_APP_...) are not used by the backend for Pinecone; they should not be relied upon server-side.

4) Index Naming and Dimensions
- Index Name: Use PINECONE_INDEX_NAME.
- Dimension: Must match the embedding model dimension. For Gemini text-embedding-004 as of this writing, the dimensionality is 768.
- Metric: cosine (recommended with Gemini embeddings).
- Pods/Serverless: Choose according to your Pinecone plan. For serverless, use the host parameter provided by Pinecone. For provisioned pods, specify environment/region.

5) Data Model and IDs
- Vector ID format: session_id:filename:chunk_index or a UUID. Include metadata for:
  - session_id (string)
  - filename (string)
  - chunk_index (int)
  - text (string, optional - can be stored in metadata to allow retrieval of the chunk without a second store)
- Namespace: Use PINECONE_NAMESPACE or derive from environment (e.g., dev, staging, prod).

6) Integration Points in Chatbot Backend (FastAPI)
Files affected: src/api/main.py, and a new client helper module src/api/vector_store_pinecone.py.

A. Upload context flow (/chat/upload-context)
- Current behavior: chunk -> embed (Gemini) -> store in in-memory store.
- New behavior:
  1. After successful text extraction and chunking, compute embeddings (existing code path).
  2. Upsert to Pinecone with ids and metadata as described.
  3. Keep existing in-memory index as a fallback only (no removal required).
  4. If Pinecone is not configured (missing API key/index), skip upsert and fallback to in-memory only.

B. Retrieval flow (/chat)
- Current behavior: _vector_search does cosine vs. in-memory embeddings, with lexical fallback.
- New behavior:
  1. If Pinecone is configured, generate query embedding and perform a top-k query to Pinecone within the target namespace filtered by session_id.
  2. Construct retrieved_context from Pinecone results’ metadata["text"] (or fetch/source using stored metadata).
  3. If Pinecone is not configured or query fails, fallback to the existing in-memory vector search, then lexical fallback.

C. Configuration and Utilities
- Add a Pinecone client wrapper in src/api/vector_store_pinecone.py:
  - initialization: create client from env, ensure index exists (optionally create if desired, or document that it must exist).
  - functions:
    - upsert_chunks(session_id, filename, chunks, embeddings)
    - query_top_k(session_id, query_embedding, top_k)
  - Safe to no-op if env is missing.

7) FastAPI/Code Changes
- Add a "vector backend configuration" check at startup or lazy-init on first use.
- Use env variables via python-dotenv (already included).
- Add logic in upload and chat handlers to route to Pinecone when available.

8) Security and Configuration
- Do not log API keys.
- Do not expose embeddings or vectors in responses.
- Keep API keys in .env; create a .env.example with the keys below.

9) Rate Limiting and Reliability
- Gemini embedding generation can rate limit. Consider simple retry/backoff around embedding and Pinecone upsert/query calls.
- Pinecone imposes rate limits by plan; handle HTTP 429s with exponential backoff (using backoff library).
- Batch upsert vectors (e.g., batches of 50-100) to reduce requests.

10) Costs and Operational Considerations
- Pinecone charges for read/write operations and storage. Monitor usage.
- Index dimension (768) and top_k affect latency and cost.
- For serverless, index creation and scaling are managed automatically; for provisioned, ensure pods and replicas are adequate.
- Clean-up strategy: If sessions are ephemeral, consider periodic deletion of old vectors (by filtering on session_id and time).

11) API Setup Steps
- Create a Pinecone project and obtain an API key.
- Create an index named PINECONE_INDEX_NAME with:
  - dimension: 768 (for Gemini text-embedding-004)
  - metric: cosine
  - serverless or provisioned per your plan
- Obtain the environment or host parameters.
- Populate backend .env with keys listed above.

12) Environment Variables Summary (.env.example)
- GEMINI_API_KEY=
- PINECONE_API_KEY=
- PINECONE_ENVIRONMENT=            # or leave empty if using serverless host
- PINECONE_HOST=                   # serverless endpoint if applicable
- PINECONE_INDEX_NAME=intelliquery-chatbot
- PINECONE_NAMESPACE=default
- PINECONE_TOP_K=3
- CHATBOT_SQLALCHEMY_DATABASE_URL=sqlite:///./chatbot_users.db

13) Removal/Revisions of Supabase/pgvector
- Remove any references to Supabase URL/Keys for vector search in docs and scripts.
- No Supabase/pgvector client libraries are required for vector search.

14) Next Steps (code)
- Implement src/api/vector_store_pinecone.py.
- Wire into upload and chat flows in src/api/main.py.
- Keep in-memory fallback in place.

```diff
High-level code diffs (conceptual):
+ from .vector_store_pinecone import PineconeVectorStore
+ pinecone_store = PineconeVectorStore.from_env()
...
def upload_chat_context(...):
    ...
    vectors = _embed_texts(chunks)
    if pinecone_store.is_configured:
        pinecone_store.upsert_chunks(session_id, filename, chunks, vectors)
    # existing in-memory index remains

def chat(...):
    ...
    if pinecone_store.is_configured:
        qvec = _embed_one(request.query)
        if qvec:
            top = pinecone_store.query_top_k(session_id, qvec, top_k=env_or_default)
            retrieved_context = "\n---\n".join([m["text"] for m in top])
    if not retrieved_context:
        retrieved_context = "\n---\n".join(_vector_search(session_id, request.query, top_k=3))
```

This plan keeps all public interfaces intact while transitioning the vector storage and retrieval to Pinecone.
