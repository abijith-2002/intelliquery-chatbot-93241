import io
from typing import List, Tuple, Optional

# Libraries for file parsing
# - TXT: native decode
# - PDF: pdfminer.six
# - DOCX: python-docx
# - XLSX: pandas.read_excel (refactored; no cell flattening)
from pdfminer.high_level import extract_text as pdf_extract_text
from docx import Document as DocxDocument
import pandas as pd


# PUBLIC_INTERFACE
def extract_text_from_bytes(filename: str, content: bytes) -> Tuple[str, Optional[str]]:
    """
    PUBLIC_INTERFACE
    Extract readable text from a file given its filename and raw bytes.

    Supports:
        - .txt  : UTF-8 decode with errors ignored
        - .pdf  : pdfminer.six text extraction
        - .docx : python-docx extraction (paragraphs and table cells)
        - .xlsx : preview text constructed from DataFrame metadata (no cell flattening)

    Args:
        filename (str): Original filename (used for type detection).
        content (bytes): Raw file content.

    Returns:
        Tuple[str, Optional[str]]: (text, error)
            - text: extracted text content or a metadata preview (empty if error)
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
            # For xlsx, do not flatten the entire sheet contents anymore.
            # Instead, return a compact preview constructed from DataFrame metadata.
            preview, err = _xlsx_preview_from_metadata(content)
            return (preview or ""), err
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


def _xlsx_preview_from_metadata(content: bytes) -> Tuple[str, Optional[str]]:
    """
    Build a lightweight human-readable preview based on DataFrame metadata
    using pandas.read_excel. Returns a string suitable for UI previews.

    We intentionally avoid returning flattened cell values to prevent large-context
    ingestion; actual DataFrame storage is handled by the upload endpoint logic.

    Returns:
        Tuple[str, Optional[str]]: (preview_text, error)
    """
    try:
        bio = io.BytesIO(content)
        # Attempt reading first sheet only for a concise preview
        df = pd.read_excel(bio, sheet_name=0, nrows=5)
        cols = list(df.columns)
        dtypes = [str(t) for t in df.dtypes.values]
        sample_rows = df.head(3).to_dict(orient="records")
        preview_lines = [
            f"Columns: {cols}",
            f"Dtypes: {dtypes}",
            f"Sample (first 3 rows): {sample_rows}",
        ]
        return " | ".join(preview_lines), None
    except Exception as e:
        return "", f"Failed to generate xlsx preview: {e}"


# PUBLIC_INTERFACE
def summarize_text_preview(text: str, max_chars: int = 500) -> str:
    """
    PUBLIC_INTERFACE
    Produce a compact preview of extracted content for UI confirmation.

    Args:
        text (str): Full extracted text or a metadata preview string.
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
