Background embeddings pipeline

Overview
- The backend now includes a background worker (src/api/background_worker.py) that batches requests for Gemini embeddings and persists vectors to a configured vector store implementation (FAISS in-memory or pgvector).
- Spreadsheet (.xlsx) ingestion has been removed. Only .txt, .pdf, and .docx uploads contribute to context and embeddings.
- Progress is tracked inside the UploadJob.catalog under the key "embedding_progress" and surfaced via /chat/context-status/{job_id}.

Configuration (.env)
- GEMINI_API_KEY: API key for Google Gemini (required).
- VECTOR_STORE: faiss (default) or pgvector.
- CHATBOT_PGVECTOR_URL: Required if VECTOR_STORE=pgvector. Example:
  postgresql+psycopg2://user:password@hostname:5432/dbname
- EMBEDDING_BATCH_SIZE: Optional, default 64.
- EMBEDDING_MAX_RETRIES: Optional, default 3.
- EMBEDDING_BACKOFF_BASE: Optional backoff base seconds, default 0.5.

Vector stores
- FAISS: In-memory index stored inside process. Suitable for demos and tests. Requires faiss-cpu.
- pgvector: Uses SQLAlchemy to upsert into a table named embeddings with vector column.

APIs
- Enqueue: upload_context will automatically enqueue per-row embeddings for XLSX files after cataloging. It creates a namespace "xlsx:{filename}:{sheet}" for grouping.
- Status: GET /chat/context-status/{job_id} returns job details including embedding_progress:
  {
    "embedding_progress": {
        "queued": 1200,
        "rows_enqueued": 1200,
        "done": 1184,
        "error": 16
    }
  }

Notes
- The chat retrieval path still uses in-memory RAG for previously extracted full texts. The background vector store path is designed for scalable per-row retrieval in future endpoints or integrations.
- If GEMINI_API_KEY is missing, vectors will be None and counted as errors; the job remains otherwise successful for ingestion.
- The worker starts on app import; if import fails, the rest of the API remains usable.
