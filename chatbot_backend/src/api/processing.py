"""
Shared processing utilities for uploaded files:
- Extract/parse content from supported types (.txt, .pdf, .docx, .xlsx, .json)
- Build previews and character counts
- Index chunks into in-memory store and ChromaDB via functions defined in main.py

This module is imported at runtime by background_jobs and sync endpoints to avoid code duplication.
"""
from __future__ import annotations

from typing import List, Tuple, Dict, Any, Optional, Callable

from fastapi import HTTPException

# We'll import file parsers from file_utils
from .file_utils import (
    extract_text_from_bytes,
    summarize_text_preview,
    parse_and_flatten_json,
    format_kv_pairs_as_text,
    parse_xlsx_to_row_chunks,
)

# Note: To avoid circular import at module import time, we do local imports
# of functions from main.py inside the functions that need them.


def _index_json(session_id: str, filename: str, data: bytes) -> Tuple[Dict[str, Any], Optional[str]]:
    from .main import _index_json_kv_pairs_for_session
    pairs, parse_err = parse_and_flatten_json(data or b"")
    text = format_kv_pairs_as_text(pairs) if pairs else ""
    preview = summarize_text_preview(text, max_chars=500) if text else ""
    chars = len(text)
    if pairs and not parse_err:
        try:
            _index_json_kv_pairs_for_session(session_id, filename, pairs)
        except Exception:
            # Non-fatal
            pass
    result = {
        "filename": filename,
        "size": len(data or b""),
        "content_chars": chars,
        "preview": preview,
        "error": parse_err,
    }
    return result, parse_err


def _index_xlsx(session_id: str, filename: str, data: bytes) -> Tuple[Dict[str, Any], Optional[str], str]:
    from .main import _index_xlsx_row_chunks_for_session
    row_chunks, parse_err = parse_xlsx_to_row_chunks(data or b"", cols_per_chunk=30)
    joined = "\n".join(rc["text"] for rc in row_chunks) if row_chunks else ""
    preview = summarize_text_preview(joined, max_chars=500) if joined else ""
    chars = len(joined)
    if row_chunks and not parse_err:
        try:
            _index_xlsx_row_chunks_for_session(session_id, filename, row_chunks)
        except Exception:
            pass
    result = {
        "filename": filename,
        "size": len(data or b""),
        "content_chars": chars,
        "preview": preview,
        "error": parse_err,
    }
    combined_text = f"[{filename}]\n{joined}\n" if joined else ""
    return result, parse_err, combined_text


def _index_general_file(session_id: str, filename: str, data: bytes) -> Tuple[Dict[str, Any], Optional[str], str]:
    from .main import _index_text_for_session
    text, err = extract_text_from_bytes(filename, data or b"")
    preview = summarize_text_preview(text, max_chars=500) if text else ""
    chars = len(text)
    if text and not err:
        try:
            _index_text_for_session(session_id, filename, text)
        except Exception:
            pass
    result = {
        "filename": filename,
        "size": len(data or b""),
        "content_chars": chars,
        "preview": preview,
        "error": err,
    }
    combined_text = f"[{filename}]\n{text}\n" if text else ""
    return result, err, combined_text


# PUBLIC_INTERFACE
def process_files(
    session_id: str,
    files_data: List[Tuple[str, bytes]],
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> Dict[str, Any]:
    """
    PUBLIC_INTERFACE
    Process uploaded files: parse, preview, and index asynchronously/synchronously.

    Args:
        session_id (str): session identifier.
        files_data (List[Tuple[str, bytes]]): list of (filename, bytes) tuples.
        progress_callback (Callable[[int, str], None], optional): callback to report progress percentage and message.

    Returns:
        dict: {
            "session_id": str,
            "files_processed": List[UploadedFileResult-like dicts],
            "total_chars": int,
            "message": str
        }
    """
    if not session_id or not isinstance(session_id, str):
        raise HTTPException(status_code=400, detail="session_id must be provided as a non-empty string.")
    if not files_data:
        raise HTTPException(status_code=400, detail="At least one file must be provided.")

    results: List[Dict[str, Any]] = []
    combined_text_parts: List[str] = []
    total_chars = 0

    n = len(files_data)
    for idx, (filename, data) in enumerate(files_data, start=1):
        name_lower = (filename or "unnamed").lower()
        if progress_callback:
            pct = int(((idx - 1) / max(1, n)) * 80)  # up to 80% while processing files
            progress_callback(pct, f"Processing {idx}/{n}: {filename}")

        if name_lower.endswith(".json"):
            res, _err = _index_json(session_id, filename, data)
            results.append(res)
            if not res.get("error") and res.get("content_chars", 0) > 0:
                total_chars += int(res["content_chars"])
                # For combined text, rebuild from pairs text
                pairs, _ = parse_and_flatten_json(data or b"")
                text = format_kv_pairs_as_text(pairs) if pairs else ""
                if text:
                    combined_text_parts.append(f"[{filename}]\n{text}\n")
            continue

        if name_lower.endswith(".xlsx"):
            res, _err, joined = _index_xlsx(session_id, filename, data)
            results.append(res)
            if not res.get("error") and res.get("content_chars", 0) > 0:
                total_chars += int(res["content_chars"])
                if joined:
                    combined_text_parts.append(joined)
            continue

        # General (txt, pdf, docx)
        res, _err, combined = _index_general_file(session_id, filename, data)
        results.append(res)
        if not res.get("error") and res.get("content_chars", 0) > 0:
            total_chars += int(res["content_chars"])
            if combined:
                combined_text_parts.append(combined)

    # Update legacy context store in main.py
    from .main import CONTEXT_STORE
    if total_chars > 0:
        combined_text = "\n".join(combined_text_parts).strip()
        prev_ctx = CONTEXT_STORE.get(session_id, {})
        prev_combined = prev_ctx.get("combined", "")
        prev_files = prev_ctx.get("files", [])
        merged_combined = (prev_combined + "\n\n" + combined_text).strip() if prev_combined else combined_text
        CONTEXT_STORE[session_id] = {
            "files": prev_files + results,
            "combined": merged_combined,
        }

    if progress_callback:
        progress_callback(90, "Finalizing...")

    message = (
        "Processed files successfully. Session context updated and indexed."
        if total_chars > 0
        else "Processed files, but no readable content was extracted."
    )

    if progress_callback:
        progress_callback(95, "Almost done")

    return {
        "session_id": session_id,
        "files_processed": results,
        "total_chars": total_chars,
        "message": message,
    }
