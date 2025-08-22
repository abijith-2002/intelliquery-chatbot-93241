# Project Repository

This is the initial README file for the project.

## Vector Search Migration: Supabase/pgvector → Pinecone

We are migrating the vector database from Supabase/pgvector to Pinecone.

Key changes:
- Backend (FastAPI) will store and query embeddings in Pinecone.
- New dependencies: `pinecone-client` and optional `backoff`.
- New environment variables:
  - PINECONE_API_KEY
  - PINECONE_INDEX_NAME
  - PINECONE_NAMESPACE (optional, default: default)
  - PINECONE_TOP_K (optional, default: 3)
  - One of:
    - PINECONE_HOST (serverless)
    - PINECONE_ENVIRONMENT (legacy/provisioned)

Backend integration points (to be wired):
- On upload (/chat/upload-context): after chunking and embedding, upsert vectors to Pinecone.
- On chat (/chat): query Pinecone using the query embedding filtered by session_id; fallback to in-memory index if Pinecone not configured/available.

Operational considerations:
- Index dimension must match Gemini embeddings (text-embedding-004: 768).
- Metric: cosine recommended.
- Handle rate limits with retries; batching upserts improves performance and cost.

See chatbot_backend/PINECONE_INTEGRATION.md for the detailed plan and env example in chatbot_backend/.env.example.
