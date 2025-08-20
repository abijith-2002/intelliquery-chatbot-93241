"""
Background job manager to process heavy tasks (e.g., Excel parsing, embedding, and vector store writes)
off the request thread to prevent gateway timeouts.

Provides a singleton JobManager with:
- submit_upload_job(session_id, files_data): returns job_id
- get_job(job_id): returns job info dict with status, progress, message, result if available
- cancel_job(job_id): best effort cancellation (if not started)

Jobs store progress and final UploadContextResponse-like results.

Note: This module is intentionally not imported at app module import time in main.py to avoid
import cycles. Endpoints import/get the singleton on demand.
"""
from __future__ import annotations

import uuid
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List, Tuple, Optional

# We import lazily inside runner to avoid circular import:
# from .processing import process_files


class JobStatus:
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELED = "canceled"


class JobManager:
    """
    Simple in-memory job manager for background processing.

    Thread-safe updates with a lock. Uses a ThreadPoolExecutor to run jobs in background.
    """

    def __init__(self, max_workers: int = 2) -> None:
        self._lock = threading.Lock()
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="bg-job")

    def _set_job(self, job_id: str, **updates: Any) -> None:
        with self._lock:
            info = self._jobs.get(job_id, {})
            info.update(updates)
            self._jobs[job_id] = info

    def _get_job(self, job_id: str) -> Dict[str, Any]:
        with self._lock:
            return dict(self._jobs.get(job_id, {}))

    def _run_upload_job(self, job_id: str, session_id: str, files_data: List[Tuple[str, bytes]]) -> None:
        # Avoid top-level import to prevent circular import at module import time
        from .processing import process_files

        self._set_job(job_id, status=JobStatus.RUNNING, progress=5, message="Starting processing...")
        try:
            # The process_files function returns a dict with keys:
            #  session_id, files_processed (list), total_chars, message
            def progress_cb(percentage: int, message: str) -> None:
                # Clamp and update
                pct = max(0, min(100, int(percentage)))
                self._set_job(job_id, progress=pct, message=message)

            result = process_files(session_id=session_id, files_data=files_data, progress_callback=progress_cb)
            # Mark success
            self._set_job(job_id, status=JobStatus.SUCCEEDED, progress=100, message="Completed", result=result)
        except Exception as e:
            self._set_job(job_id, status=JobStatus.FAILED, message=f"Job failed: {e}")

    # PUBLIC_INTERFACE
    def submit_upload_job(self, session_id: str, files_data: List[Tuple[str, bytes]]) -> str:
        """
        PUBLIC_INTERFACE
        Submit an upload processing task to run in background.

        Args:
            session_id (str): session ID to associate with.
            files_data (List[Tuple[str, bytes]]): List of tuples (filename, raw_bytes).

        Returns:
            str: job_id to poll status.
        """
        job_id = uuid.uuid4().hex
        self._set_job(job_id, status=JobStatus.PENDING, progress=0, message="Queued", result=None, session_id=session_id)
        # Enqueue
        self._executor.submit(self._run_upload_job, job_id, session_id, files_data)
        return job_id

    # PUBLIC_INTERFACE
    def get_job(self, job_id: str) -> Dict[str, Any]:
        """
        PUBLIC_INTERFACE
        Retrieve job status and details.

        Returns:
            dict: {status, progress, message, result?}
        """
        return self._get_job(job_id)

    # PUBLIC_INTERFACE
    def cancel_job(self, job_id: str) -> bool:
        """
        PUBLIC_INTERFACE
        Best-effort cancellation: if job is still pending (not started), mark canceled.

        Note: Running jobs are not forcibly terminated here.
        """
        with self._lock:
            info = self._jobs.get(job_id)
            if not info:
                return False
            if info.get("status") == JobStatus.PENDING:
                info["status"] = JobStatus.CANCELED
                info["message"] = "Canceled before start"
                info["progress"] = 0
                self._jobs[job_id] = info
                return True
            return False


# Singleton accessor
_singleton: Optional[JobManager] = None


# PUBLIC_INTERFACE
def get_job_manager() -> JobManager:
    """
    PUBLIC_INTERFACE
    Get the singleton JobManager instance. Max workers can be tuned with
    CHATBOT_BG_MAX_WORKERS env var.
    """
    import os
    global _singleton
    if _singleton is None:
        try:
            mw = int(os.getenv("CHATBOT_BG_MAX_WORKERS", "2"))
        except Exception:
            mw = 2
        _singleton = JobManager(max_workers=max(1, mw))
    return _singleton
