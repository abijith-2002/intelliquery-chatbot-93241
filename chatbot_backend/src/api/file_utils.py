import io
import math
import random
from typing import List, Tuple, Optional, Dict, Any

# Libraries for file parsing
# - TXT: native decode
# - PDF: pdfminer.six
# - DOCX: python-docx
# - XLSX: openpyxl
from pdfminer.high_level import extract_text as pdf_extract_text
from docx import Document as DocxDocument
from openpyxl import load_workbook


# PUBLIC_INTERFACE
def extract_text_from_bytes(filename: str, content: bytes) -> Tuple[str, Optional[str]]:
    """
    PUBLIC_INTERFACE
    Extract readable text from a file given its filename and raw bytes.

    Supports:
        - .txt  : UTF-8 decode with errors ignored
        - .pdf  : pdfminer.six text extraction
        - .docx : python-docx extraction (paragraphs and table cells)
        - .xlsx : openpyxl extraction (sheet name and cells, tab-separated rows)

    Args:
        filename (str): Original filename (used for type detection).
        content (bytes): Raw file content.

    Returns:
        Tuple[str, Optional[str]]: (text, error)
            - text: extracted text content (empty if error)
            - error: error message if extraction failed, otherwise None

    Notes:
        This function returns a single string, intended for legacy/simple ingestion.
        For wide Excel files (>700 columns), use extract_xlsx_wide_chunks to produce
        token-bounded, metadata-rich chunks suitable for RAG ingestion.
    """
    name_lower = (filename or "").lower()

    try:
        if name_lower.endswith(".txt"):
            return _extract_txt(content), None
        if name_lower.endswith(".pdf"):
            return _extract_pdf(content), None
        if name_lower.endswith(".docx"):
            return _extract_docx(content), None
        if name_lower.endswith(".xlsx"):
            return _extract_xlsx(content), None
        return "", f"Unsupported file type for '{filename}'. Allowed: .txt, .pdf, .docx, .xlsx"
    except Exception as e:
        return "", f"Failed to extract '{filename}': {e}"


def _extract_txt(content: bytes) -> str:
    """Decode as utf-8 ignoring errors."""
    return content.decode("utf-8", errors="ignore")


def _extract_pdf(content: bytes) -> str:
    """Extract text from PDF using pdfminer.six."""
    bio = io.BytesIO(content)
    text = pdf_extract_text(bio) or ""
    return text


def _extract_docx(content: bytes) -> str:
    """Extract text from DOCX using python-docx (paragraphs and table cells)."""
    bio = io.BytesIO(content)
    doc = DocxDocument(bio)
    parts: List[str] = []
    # Paragraphs
    for p in doc.paragraphs:
        if p.text:
            parts.append(p.text)
    # Tables
    for tbl in doc.tables:
        for row in tbl.rows:
            row_vals = []
            for cell in row.cells:
                row_vals.append(cell.text.strip())
            if any(v for v in row_vals):
                parts.append("\t".join(row_vals))
    return "\n".join(parts).strip()


def _extract_xlsx(content: bytes) -> str:
    """Extract text from XLSX using openpyxl (sheet by sheet, TSV rows)."""
    bio = io.BytesIO(content)
    wb = load_workbook(bio, data_only=True, read_only=True)
    parts: List[str] = []
    for ws in wb.worksheets:
        parts.append(f"[Sheet: {ws.title}]")
        for row in ws.iter_rows(values_only=True):
            vals = []
            for cell in row:
                if cell is None:
                    vals.append("")
                else:
                    vals.append(str(cell))
            # Skip completely empty rows
            if any(v.strip() for v in vals):
                parts.append("\t".join(vals))
        parts.append("")  # blank line between sheets
    return "\n".join(parts).strip()


def _estimate_tokens(s: str) -> int:
    """
    Rough token estimator: assumes ~4 chars per token as heuristic.
    Keeps us within safety bounds for LLM prompts.
    """
    if not s:
        return 0
    # count words as a backup; tokens ~ max(len(s)/4, word_count)
    words = len(s.split())
    chars = len(s)
    return max(words, math.ceil(chars / 4))


def _stringify(cell: Any) -> str:
    """Convert cell to user-friendly string."""
    if cell is None:
        return ""
    return str(cell)


def _compute_column_stats(values: List[Any]) -> Dict[str, Any]:
    """
    Compute simple stats for a column based on sampled values.
    - For numeric: min, max, mean, count
    - For non-numeric: unique count and top samples
    """
    cleaned = [v for v in values if v is not None and _stringify(v).strip() != ""]
    if not cleaned:
        return {"count": 0}

    # detect numeric
    numeric_vals: List[float] = []
    for v in cleaned:
        try:
            numeric_vals.append(float(v))
        except Exception:
            pass

    if len(numeric_vals) >= max(3, len(cleaned) // 2):
        mn = min(numeric_vals)
        mx = max(numeric_vals)
        mean = sum(numeric_vals) / len(numeric_vals)
        return {"count": len(cleaned), "min": mn, "max": mx, "mean": mean}
    else:
        unique_vals = {}
        for v in cleaned:
            sv = _stringify(v)
            unique_vals[sv] = unique_vals.get(sv, 0) + 1
        top = sorted(unique_vals.items(), key=lambda x: x[1], reverse=True)[:5]
        return {"count": len(cleaned), "unique": len(unique_vals), "top": [k for k, _ in top]}


# PUBLIC_INTERFACE
def extract_xlsx_wide_chunks(
    content: bytes,
    max_tokens_per_chunk: int = 1200,
    row_chunk_size: int = 100,
    base_sample_columns: int = 5,
    random_sample_columns: int = 3,
    stats_sample_rows: int = 200,
) -> List[str]:
    """
    PUBLIC_INTERFACE
    Specialized extractor for very wide Excel files. Produces a list of chunk strings.
    Each chunk:
      - Includes sheet name and row range
      - Includes complete column metadata (list of all column names)
      - Contains detailed row data only for a sampled subset of columns
      - Optionally includes lightweight per-column statistics
      - Enforces an approximate token limit per chunk

    Args:
        content (bytes): XLSX file bytes.
        max_tokens_per_chunk (int): Target token cap per chunk.
        row_chunk_size (int): Number of rows per chunk window before token trimming.
        base_sample_columns (int): Always include first N columns.
        random_sample_columns (int): Add up to M random additional columns, budget permitting.
        stats_sample_rows (int): Rows to sample for stats per sheet.

    Returns:
        List[str]: Chunk strings ready for embedding/indexing.
    """
    bio = io.BytesIO(content)
    wb = load_workbook(bio, data_only=True, read_only=True)

    chunks: List[str] = []

    for ws in wb.worksheets:
        # Prepare column headers from first non-empty row, else generate generic names
        header_row_iter = ws.iter_rows(min_row=1, max_row=1, values_only=True)
        headers: List[str] = []
        try:
            first_row = next(header_row_iter)
            headers = [(_stringify(c) or f"col_{i+1}") for i, c in enumerate(first_row or [])]
        except StopIteration:
            headers = []

        # If header seems empty, attempt to infer width by scanning next row
        if not headers or all(h.strip() == "" for h in headers):
            row_iter = ws.iter_rows(min_row=1, max_row=2, values_only=True)
            inferred_width = 0
            for r in row_iter:
                inferred_width = max(inferred_width, len(list(r or [])))
            headers = [f"col_{i+1}" for i in range(inferred_width or 1)]

        num_cols = len(headers)

        # Determine sampling for columns
        sampled_cols_idx = list(range(min(base_sample_columns, num_cols)))
        remaining = [i for i in range(num_cols) if i not in sampled_cols_idx]
        random.shuffle(remaining)
        sampled_cols_idx += remaining[: max(0, min(random_sample_columns, len(remaining)))]

        sampled_headers = [headers[i] for i in sampled_cols_idx]

        # Precompute basic stats using a limited row sample to keep costs down
        stats_rows = []
        for i, row in enumerate(ws.iter_rows(min_row=2, values_only=True), start=2):
            stats_rows.append(row)
            if len(stats_rows) >= stats_sample_rows:
                break

        col_stats: Dict[str, Dict[str, Any]] = {}
        for ci, h in enumerate(headers):
            values = []
            for r in stats_rows:
                if ci < len(r or ()):
                    values.append(r[ci])
            col_stats[h] = _compute_column_stats(values)

        # Construct chunks by iterating rows in windows
        current_rows: List[List[Any]] = []
        start_row_idx = 2  # assuming row 1 is header

        def flush_chunk(rows: List[List[Any]], start_idx: int, end_idx: int):
            """
            Flushes the current buffer to a chunk with accurate metadata.
            Ensures inclusive [Rows: start-end] and correct rX labels.
            """
            if not rows:
                return
            # Core metadata
            meta_lines = []
            meta_lines.append(f"[Sheet: {ws.title}]")
            meta_lines.append(f"[Rows: {start_idx}-{end_idx}]")
            meta_lines.append(f"[Columns ({num_cols}): {', '.join(headers)}]")

            # Build body with sampled columns only
            body_lines = []
            body_lines.append(f"[Sampled Columns: {', '.join(sampled_headers)}]")
            body_lines.append("Rows:")
            for ridx, r in enumerate(rows):
                row_vals = []
                for ci in sampled_cols_idx:
                    # Ensure we do not index beyond row width (normalize blank if short)
                    val = _stringify(r[ci] if ci < len(r) else "")
                    row_vals.append(f"{headers[ci]}={val}")
                # Row label is the real Excel row number (1-based), e.g., r2..rN
                body_lines.append(f"- r{start_idx + ridx}: " + "; ".join(row_vals))

            # Append condensed per-column stats (metadata-level)
            stats_lines = []
            stats_lines.append("[Column Stats]")
            # Only include compact stats to preserve tokens
            for h in headers[: min(50, len(headers))]:  # cap verbose stats to first 50 columns
                st = col_stats.get(h, {})
                stats_desc_parts = []
                for k in ["count", "min", "max", "mean", "unique"]:
                    if k in st:
                        stats_desc_parts.append(f"{k}={st[k]}")
                if "top" in st and st["top"]:
                    stats_desc_parts.append(f"top={st['top']}")
                if stats_desc_parts:
                    stats_lines.append(f"- {h}: " + ", ".join(_stringify(x) for x in stats_desc_parts))

            # Combine and enforce token budget by trimming body if needed
            assembled = "\n".join(meta_lines + [""] + body_lines + [""] + stats_lines)
            tokens = _estimate_tokens(assembled)
            if tokens > max_tokens_per_chunk:
                # Trim rows gradually but never produce an empty body
                prunable = list(rows)
                # Keep proportionally based on token budget; minimum of 5 rows to preserve tail visibility
                keep = max(5, int(len(prunable) * max_tokens_per_chunk / max(tokens, 1)))
                prunable = prunable[:keep]

                body_lines_trim = []
                body_lines_trim.append(f"[Sampled Columns: {', '.join(sampled_headers)}]")
                body_lines_trim.append("Rows:")
                for ridx, r in enumerate(prunable):
                    row_vals = []
                    for ci in sampled_cols_idx:
                        val = _stringify(r[ci] if ci < len(r) else "")
                        row_vals.append(f"{headers[ci]}={val}")
                    body_lines_trim.append(f"- r{start_idx + ridx}: " + "; ".join(row_vals))
                body_lines_trim.append(f"... (trimmed; total rows in window: {len(rows)})")

                assembled = "\n".join(meta_lines + [""] + body_lines_trim + [""] + stats_lines)

                # final safety check: if still too large, drop stats lines
                if _estimate_tokens(assembled) > max_tokens_per_chunk:
                    assembled = "\n".join(meta_lines + [""] + body_lines_trim)

            chunks.append(assembled)

        # Iterate through all data rows
        for row_number, row in enumerate(ws.iter_rows(min_row=2, values_only=True), start=2):
            # Normalize row to header width (pad or trim to consistent column count)
            normalized = list(row or [])
            if len(normalized) < num_cols:
                normalized.extend([""] * (num_cols - len(normalized)))
            elif len(normalized) > num_cols:
                normalized = normalized[:num_cols]
            current_rows.append(normalized)

            # When buffer hits the configured window size, flush with inclusive end row = row_number
            if len(current_rows) >= row_chunk_size:
                flush_chunk(current_rows, start_row_idx, row_number)
                current_rows = []
                start_row_idx = row_number + 1  # Next chunk starts on the very next row

        # Flush remaining rows buffer.
        # Compute the true inclusive end index from start_row_idx and buffered length.
        if current_rows:
            final_end_row_number = start_row_idx + len(current_rows) - 1
            flush_chunk(current_rows, start_row_idx, final_end_row_number)

    return chunks


# PUBLIC_INTERFACE
def summarize_text_preview(text: str, max_chars: int = 500) -> str:
    """
    PUBLIC_INTERFACE
    Produce a compact preview of extracted content for UI confirmation.

    Args:
        text (str): Full extracted text.
        max_chars (int): Maximum number of characters to include.

    Returns:
        str: A trimmed single-line preview (with newlines collapsed).
    """
    if not text:
        return ""
    collapsed = " ".join(text.split())
    if len(collapsed) <= max_chars:
        return collapsed
    return collapsed[: max_chars - 3] + "..."
