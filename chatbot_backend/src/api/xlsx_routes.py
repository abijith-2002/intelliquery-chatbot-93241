from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field

from .xlsx_utils import ingest_xlsx_for_session
from .job_tracker import start_job, finalize_job, JOBS
from .background_worker import enqueue_embedding_job
from .vector_store import build_vector_id

import pandas as pd

router = APIRouter()


class UploadXLSXResponse(BaseModel):
    """Response schema for XLSX uploads."""
    session_id: str = Field(..., description="Session ID associated with the upload")
    job_id: str = Field(..., description="Job id for progress tracking")
    message: str = Field(..., description="Status message")
    catalog: Dict[str, Any] = Field(default_factory=dict, description="Per-file sheet/column metadata")


# PUBLIC_INTERFACE
@router.post(
    "/chat/upload-xlsx",
    response_model=UploadXLSXResponse,
    tags=["Chat"],
    summary="Upload XLSX files for a chat session",
    description="Accepts one or more .xlsx files, loads sheets into memory per session, catalogs columns, and enqueues background embeddings per row.",
    responses={
        400: {"description": "Validation error or no files"},
        415: {"description": "Unsupported media type"},
    },
)
def upload_xlsx(
    session_id: str = Form(..., description="Session ID"),
    files: List[UploadFile] = File(..., description="One or more .xlsx files"),
):
    """
    PUBLIC_INTERFACE
    Upload and register XLSX spreadsheets for a session.
    - Loads each sheet into pandas DataFrame (capped at 50k rows).
    - Catalogs sheet names and columns.
    - Enqueues background embeddings per-row using Gemini, if configured.
    """
    if not session_id or not isinstance(session_id, str):
        raise HTTPException(status_code=400, detail="session_id must be provided as a non-empty string.")
    if not files:
        raise HTTPException(status_code=400, detail="At least one .xlsx file must be provided.")

    # validate types
    for f in files:
        if not (f.filename or "").lower().endswith(".xlsx"):
            raise HTTPException(status_code=415, detail=f"Unsupported media type for {f.filename}. Only .xlsx allowed.")

    job = start_job(session_id, [(f.filename or "unnamed.xlsx") for f in files])
    catalog_files: List[Dict[str, Any]] = []
    total_rows_enqueued = 0

    for idx, f in enumerate(files):
        filename = f.filename or "unnamed.xlsx"
        try:
            # Read the entire content once
            data = f.file.read()
            if not data or len(data) == 0:
                raise ValueError("Uploaded file is empty.")
        except Exception as e:
            # mark error
            try:
                JOBS[job.job_id].files[idx].status = "error"
                JOBS[job.job_id].files[idx].error = f"Failed to read file: {e}"
            except Exception:
                pass
            catalog_files.append({
                "filename": filename,
                "catalog_error": f"Failed to read file: {e}",
            })
            continue

        try:
            # ingest and catalog
            res = ingest_xlsx_for_session(session_id, filename, data)
            try:
                # Update job file record with a meaningful preview
                sheet_summaries = []
                total_rows = 0
                for s in res.sheets:
                    # compute rows from in-memory df to ensure accurate cap consideration
                    rows = 0
                    try:
                        from . import xlsx_utils as _xlsx
                        rows = len(_xlsx.XLSX_SESSIONS[session_id]["files"][filename]["sheets"][s]["df"])
                    except Exception:
                        rows = 0
                    cols = res.columns_per_sheet.get(s, [])
                    total_rows += int(rows)
                    sheet_summaries.append(f"{s} ({rows} rows, {len(cols)} cols)")
                preview_msg = f"{len(res.sheets)} sheet(s): " + ", ".join(sheet_summaries)
                # set record fields
                JOBS[job.job_id].files[idx].status = "done"
                JOBS[job.job_id].files[idx].size = len(data or b"")
                JOBS[job.job_id].files[idx].message = "XLSX ingested"
                JOBS[job.job_id].files[idx].preview = preview_msg
                JOBS[job.job_id].files[idx].content_chars = 0  # not text-based; keep 0
            except Exception:
                pass

            # Build per-row payloads for embeddings: namespace = xlsx:{filename}:{sheet}
            rows_enqueued = 0
            from . import xlsx_utils as _xlsx
            sess = _xlsx.XLSX_SESSIONS.get(session_id, {})
            filemeta = sess.get("files", {}).get(filename, {})
            for sheet_name, sh in (filemeta.get("sheets") or {}).items():
                df: pd.DataFrame = sh["df"]
                # Skip empty dataframes
                if df is None or getattr(df, "empty", False):
                    continue
                namespace = f"xlsx:{filename}:{sheet_name}"
                payloads = []
                # Create a concatenated row text and normalized metadata
                for i, row in df.iterrows():
                    meta: Dict[str, Any] = {}
                    parts = []
                    for col in df.columns:
                        val = row[col]
                        # pd.isna expects scalar; fallback defensively
                        try:
                            is_na = pd.isna(val)
                        except Exception:
                            is_na = False
                        sval = "" if is_na else str(val)
                        norm_key = re_norm(col)
                        meta[norm_key] = sval
                        parts.append(f"{norm_key}: {sval}")
                    text = " | ".join(parts)
                    payloads.append({"id": build_vector_id("xlsxrow"), "text": text, "metadata": meta})
                if payloads:
                    rows_enqueued += enqueue_embedding_job(job.job_id, session_id, namespace, payloads)

            total_rows_enqueued += rows_enqueued

            catalog_files.append({
                "filename": filename,
                "catalog": {
                    "sheets": [
                        {
                            "sheet_name": s,
                            "columns": res.columns_per_sheet.get(s, []),
                            "row_count_scanned": len((_xlsx.XLSX_SESSIONS[session_id]["files"][filename]["sheets"][s]["df"]))
                        } for s in res.sheets
                    ]
                }
            })

        except Exception as e:
            try:
                JOBS[job.job_id].files[idx].status = "error"
                JOBS[job.job_id].files[idx].error = f"Failed to process XLSX: {e}"
            except Exception:
                pass
            catalog_files.append({
                "filename": filename,
                "catalog_error": f"Failed to process XLSX: {e}",
            })

    # finalize job and attach catalog
    try:
        JOBS[job.job_id].catalog["files"] = catalog_files
        JOBS[job.job_id].catalog.setdefault("embedding_progress", {}).setdefault("rows_enqueued", 0)
        JOBS[job.job_id].catalog["embedding_progress"]["rows_enqueued"] += int(total_rows_enqueued)
        finalize_job(job.job_id)
    except Exception:
        pass

    return UploadXLSXResponse(
        session_id=session_id,
        job_id=job.job_id,
        message="XLSX uploaded and cataloged.",
        catalog={"files": catalog_files},
    )


def re_norm(s: str) -> str:
    import re
    return re.sub(r"[^a-z0-9_]", "", re.sub(r"\s+", "_", (s or "").strip().lower()))
