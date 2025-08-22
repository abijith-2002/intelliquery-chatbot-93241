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
    catalog: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["files"] = [asdict(f) for f in self.files]
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
    """
    job = JOBS.get(job_id)
    if not job:
        return
    has_error = any(f.status == "error" for f in job.files)
    job.status = "completed_with_errors" if has_error else "completed"
    job.completed_at = datetime.utcnow().isoformat() + "Z"
