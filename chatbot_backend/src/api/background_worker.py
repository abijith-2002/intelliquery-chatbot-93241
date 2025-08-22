import os
import threading
import time
from queue import Queue, Empty
from typing import Any, Dict, List, Optional

import google.generativeai as genai
from pydantic import BaseModel, Field

from .config_utils import get_gemini_api_key
from .vector_store import IVectorStore, VectorRecord, get_vector_store, build_vector_id
from .job_tracker import JOBS


class _EmbeddingTask(BaseModel):
    """Task representing a batch embedding request."""
    job_id: str = Field(..., description="Upload job id for progress updates")
    session_id: str = Field(..., description="Session id")
    namespace: str = Field(..., description="Namespace like 'xlsx:filename:sheet'")
    payloads: List[Dict[str, Any]] = Field(..., description="List of rows with 'text' and 'metadata'")


class _WorkerState:
    def __init__(self) -> None:
        self.queue: "Queue[_EmbeddingTask]" = Queue()
        self.thread: Optional[threading.Thread] = None
        self.running = False
        self.store: Optional[IVectorStore] = None


_state = _WorkerState()


def _embed_batch(texts: List[str]) -> List[Optional[List[float]]]:
    """Call Gemini embeddings for a batch of texts. Serially with per-item try/except."""
    vectors: List[Optional[List[float]]] = []
    api_key = get_gemini_api_key()
    if not api_key:
        return [None] * len(texts)
    try:
        genai.configure(api_key=api_key)
        model = "models/text-embedding-004"
        for t in texts:
            try:
                result = genai.embed_content(model=model, content=t or "")
                vec = result.get("embedding", {}).get("values")
                if isinstance(vec, list) and vec and isinstance(vec[0], (int, float)):
                    vectors.append([float(v) for v in vec])
                else:
                    vectors.append(None)
            except Exception:
                vectors.append(None)
    except Exception:
        return [None] * len(texts)
    return vectors


def _update_progress(job_id: str, inc_done: int = 0, inc_error: int = 0) -> None:
    """Update in-memory job extra progress counters."""
    job = JOBS.get(job_id)
    if not job:
        return
    # attach transient progress counters in catalog root
    if "embedding_progress" not in job.catalog:
        job.catalog["embedding_progress"] = {"done": 0, "error": 0}
    job.catalog["embedding_progress"]["done"] += inc_done
    job.catalog["embedding_progress"]["error"] += inc_error


def _worker_loop() -> None:
    _state.running = True
    # init vector store lazily
    if _state.store is None:
        _state.store = get_vector_store()

    max_retries = int(os.getenv("EMBEDDING_MAX_RETRIES", "3"))
    backoff_base = float(os.getenv("EMBEDDING_BACKOFF_BASE", "0.5"))

    while _state.running:
        try:
            task = _state.queue.get(timeout=0.5)
        except Empty:
            continue

        payloads = task.payloads
        # form text list
        texts = [str(p.get("text") or "") for p in payloads]
        # retry policy around _embed_batch
        vectors: List[Optional[List[float]]] = []
        attempt = 0
        while attempt <= max_retries:
            vectors = _embed_batch(texts)
            if any(v is None for v in vectors):
                # If all failed, retry; otherwise proceed and mark per-item errors.
                if all(v is None for v in vectors) and attempt < max_retries:
                    time.sleep(backoff_base * (2 ** attempt))
                    attempt += 1
                    continue
            break

        # collect records to upsert for successful vectors
        ok_records: List[VectorRecord] = []
        errors = 0
        for vec, payload in zip(vectors, payloads):
            if vec is None:
                errors += 1
                continue
            rec = VectorRecord(
                id=payload.get("id") or build_vector_id("row"),
                session_id=task.session_id,
                namespace=task.namespace,
                metadata=payload.get("metadata") or {},
                vector=vec,
            )
            ok_records.append(rec)

        done = 0
        if ok_records:
            try:
                done = _state.store.upsert(ok_records)  # type: ignore
            except Exception:
                # if vector persistence fails, consider all failed
                errors += len(ok_records)
                done = 0
        _update_progress(task.job_id, inc_done=done, inc_error=errors)
        try:
            _state.queue.task_done()
        except Exception:
            pass

    # cleanup
    try:
        if _state.store:
            _state.store.close()
    except Exception:
        pass


# PUBLIC_INTERFACE
def start_worker() -> None:
    """PUBLIC_INTERFACE
    Start background worker thread if not running.
    """
    if _state.thread and _state.thread.is_alive():
        return
    _state.thread = threading.Thread(target=_worker_loop, name="embedding-worker", daemon=True)
    _state.thread.start()


# PUBLIC_INTERFACE
def stop_worker() -> None:
    """PUBLIC_INTERFACE
    Stop background worker."""
    _state.running = False
    t = _state.thread
    if t and t.is_alive():
        t.join(timeout=2.0)


# PUBLIC_INTERFACE
def enqueue_embedding_job(job_id: str, session_id: str, namespace: str, rows: List[Dict[str, Any]]) -> int:
    """PUBLIC_INTERFACE
    Enqueue embedding of given rows.

    Each row dict should have:
      - id (optional): string id, auto-generated if missing
      - text: str
      - metadata: dict

    Args:
        job_id (str): Job id used for progress tracking.
        session_id (str): Session id.
        namespace (str): Namespace label (e.g., 'xlsx:filename:sheet').
        rows (List[Dict[str, Any]]): Rows to embed.

    Returns:
        int: Number of rows enqueued.
    """
    if not rows:
        return 0
    batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))
    # split into batches
    total = 0
    for i in range(0, len(rows), batch_size):
        payloads = rows[i:i + batch_size]
        task = _EmbeddingTask(job_id=job_id, session_id=session_id, namespace=namespace, payloads=payloads)
        _state.queue.put(task)
        total += len(payloads)
    # seed progress counters
    job = JOBS.get(job_id)
    if job:
        if "embedding_progress" not in job.catalog:
            job.catalog["embedding_progress"] = {"done": 0, "error": 0, "queued": total}
        else:
            job.catalog["embedding_progress"]["queued"] = job.catalog["embedding_progress"].get("queued", 0) + total
    return total
