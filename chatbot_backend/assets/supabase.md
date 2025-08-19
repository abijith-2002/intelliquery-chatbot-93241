# Supabase Integration (Vector Storage for JSON Facts)

This backend can optionally store JSON fact embeddings in Supabase (Postgres + pgvector). If Supabase is not configured, facts are still indexed in-memory and used for retrieval.

## Environment Variables

Set the following environment variables in the chatbot_backend container:

- REACT_APP_SUPABASE_URL
- REACT_APP_SUPABASE_SERVICE_ROLE_KEY (preferred for server-side writes)
- REACT_APP_SUPABASE_ANON_KEY (fallback if service role key is not available)

Note: Do not commit .env files; these are provided by the deployment environment.

## Python Client

The backend uses the `supabase` Python client. See `requirements.txt` for the pinned version.

## Expected Table Schema

Create a table named `json_embeddings` with columns:

- id: uuid, primary key, default gen_random_uuid() or uuid_generate_v4()
- session_id: text (indexed)
- filename: text
- path: text (dot-notation key)
- value: text (stringified JSON leaf value)
- embedding: vector (e.g., vector(768) for Gemini text-embedding-004; adjust dimension based on model)

Example SQL (adjust dimension as needed):
```sql
create extension if not exists vector;

create table if not exists public.json_embeddings (
  id uuid primary key default gen_random_uuid(),
  session_id text not null,
  filename text not null,
  path text not null,
  value text not null,
  embedding vector(768)
);

create index if not exists idx_json_embeddings_session on public.json_embeddings(session_id);
create index if not exists idx_json_embeddings_path on public.json_embeddings(path);
```

## How It's Used

- Endpoint: POST /chat/upload-json
  - Parses and flattens uploaded JSON files.
  - Builds 'path = value' lines, embeds them, and:
    - Inserts into Supabase `json_embeddings` if configured.
    - Always indexes in-memory for retrieval.

- Retrieval:
  - During `/chat`, the system performs semantic search over JSON facts first (preferred), then over document chunks.

If you change the embedding model, update the vector dimension accordingly and rotate the stored data if necessary.
