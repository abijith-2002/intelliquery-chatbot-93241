from typing import Any, Dict, List
from dataclasses import dataclass, field, asdict
from datetime import datetime
import uuid


@dataclass
class FileProcessRecord:
    filename: str
    size: int = 0
    status: str = "pending"  # pending | processing | done | error
    message: str = ""
    preview: str = ""
    content_chars: int = 0
    error: str = ""


@dataclass
class UploadJob:
    job_id: str
    session_id: str
    status: str = "created"  # created | running | completed | completed_with_errors
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    completed_at: str = ""
    files: List[FileProcessRecord] = field(default_factory=list)
    total_chars: int = 0
    # Catalog now can include richer metadata (embedding_progress, dataset readiness, row/col counts, errors, etc.)
    catalog: Dict[str, Any] = field(default_factory=dict)
    # Aggregate error list for quick surfacing at status endpoint (in addition to per-file errors)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["files"] = [asdict(f) for f in self.files]
        # Backwards-compatible: ensure embedding_progress keys exist if progress is being tracked
        if "embedding_progress" in d.get("catalog", {}):
            ep = d["catalog"]["embedding_progress"] or {}
            ep.setdefault("queued", ep.get("queued", 0))
            ep.setdefault("rows_enqueued", ep.get("rows_enqueued", 0))
            ep.setdefault("done", ep.get("done", 0))
            ep.setdefault("error", ep.get("error", 0))
            d["catalog"]["embedding_progress"] = ep
        return d


# Simple in-memory store
JOBS: Dict[str, UploadJob] = {}


# PUBLIC_INTERFACE
def start_job(session_id: str, filenames: List[str]) -> UploadJob:
    """
    PUBLIC_INTERFACE
    Start a new upload job for a session. Initializes per-file records.
    """
    job = UploadJob(job_id=str(uuid.uuid4()), session_id=session_id, status="running")
    for name in filenames:
        job.files.append(FileProcessRecord(filename=name))
    # Initialize standard fields that /chat/context-status expects
    job.catalog.setdefault("embedding_progress", {"queued": 0, "rows_enqueued": 0, "done": 0, "error": 0})
    job.catalog.setdefault("dataset_ready", False)
    job.catalog.setdefault("sheets_summary", [])  # list of {sheet_name, rows_scanned, columns_count}
    job.catalog.setdefault("index_errors", [])     # embedding/index errors encountered
    JOBS[job.job_id] = job
    return job


# PUBLIC_INTERFACE
def get_job(job_id: str) -> Dict[str, Any]:
    """
    PUBLIC_INTERFACE
    Retrieve job info by job_id. Returns empty dict if not found.
    """
    job = JOBS.get(job_id)
    return job.to_dict() if job else {}


# PUBLIC_INTERFACE
def finalize_job(job_id: str) -> None:
    """
    PUBLIC_INTERFACE
    Mark job as completed or completed_with_errors based on file statuses.
    Sets dataset readiness if any catalog content was generated.
    """
    job = JOBS.get(job_id)
    if not job:
        return
    has_error = any(f.status == "error" for f in job.files)
    job.status = "completed_with_errors" if has_error else "completed"
    # Determine dataset readiness:
    # Ready if there is at least one file entry in catalog OR total_chars > 0 (some content extracted)
    try:
        files_meta = (job.catalog or {}).get("files") or []
        job.catalog["dataset_ready"] = bool(files_meta) or job.total_chars > 0
    except Exception:
        job.catalog["dataset_ready"] = job.total_chars > 0
    job.completed_at = datetime.utcnow().isoformat() + "Z"
