import io
import os
from typing import List, Tuple, Optional

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
    """Extract text from XLSX using openpyxl (sheet by sheet, TSV rows).

    Performance safeguards:
    - Uses read_only=True to stream rows.
    - Respects environment-configurable limits to prevent timeouts on very large files:
        FILE_EXTRACT_MAX_XLSX_CELLS (int): total maximum cells to process across the workbook (default: 200000)
        FILE_EXTRACT_MAX_XLSX_ROWS_PER_SHEET (int): per-sheet row cap (default: 20000)
    When limits are hit, extraction stops gracefully for speed and reliability.
    """
    # Read limits from environment with sensible defaults
    try:
        max_cells_total = int(os.getenv("FILE_EXTRACT_MAX_XLSX_CELLS", "200000"))
    except Exception:
        max_cells_total = 200000
    try:
        max_rows_per_sheet = int(os.getenv("FILE_EXTRACT_MAX_XLSX_ROWS_PER_SHEET", "20000"))
    except Exception:
        max_rows_per_sheet = 20000

    bio = io.BytesIO(content)
    wb = load_workbook(bio, data_only=True, read_only=True)
    parts: List[str] = []
    processed_cells = 0

    for ws in wb.worksheets:
        parts.append(f"[Sheet: {ws.title}]")
        row_count = 0
        for row in ws.iter_rows(values_only=True):
            row_count += 1
            if row_count > max_rows_per_sheet:
                parts.append("[...]")
                break

            vals = []
            for cell in row:
                processed_cells += 1
                if processed_cells > max_cells_total:
                    # Stop processing further to avoid excessive CPU time
                    vals.append("...")
                    parts.append("\t".join(vals))
                    parts.append("[...]")
                    break
                if cell is None:
                    vals.append("")
                else:
                    vals.append(str(cell))

            # If we exceeded the global cap during cell iteration, stop entire workbook processing
            if processed_cells > max_cells_total:
                break

            # Skip completely empty rows
            if any((v.strip() if isinstance(v, str) else str(v).strip()) for v in vals):
                parts.append("\t".join(vals))

        parts.append("")  # blank line between sheets
        if processed_cells > max_cells_total:
            break

    return "\n".join(parts).strip()


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
