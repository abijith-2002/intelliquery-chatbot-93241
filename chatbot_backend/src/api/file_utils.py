import io
import re
from typing import List

from pdfminer.high_level import extract_text as pdf_extract_text
from docx import Document
from openpyxl import load_workbook


SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx", ".xlsx"}


def _ext_of(filename: str) -> str:
    filename = filename or ""
    idx = filename.rfind(".")
    if idx == -1:
        return ""
    return filename[idx:].lower()


def extract_text_from_file(filename: str, data: bytes) -> str:
    """
    Extract text content from supported files:
      - .txt
      - .pdf
      - .docx
      - .xlsx (reads all cells row-wise as TSV-like text)
    """
    ext = _ext_of(filename)
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {ext or '(none)'}")

    if ext == ".txt":
        return data.decode("utf-8", errors="ignore")

    if ext == ".pdf":
        with io.BytesIO(data) as bio:
            return pdf_extract_text(bio) or ""

    if ext == ".docx":
        with io.BytesIO(data) as bio:
            doc = Document(bio)
            parts = []
            for p in doc.paragraphs:
                txt = p.text.strip()
                if txt:
                    parts.append(txt)
            return "\n".join(parts)

    if ext == ".xlsx":
        with io.BytesIO(data) as bio:
            wb = load_workbook(bio, data_only=True, read_only=True)
            parts = []
            for ws in wb.worksheets:
                for row in ws.iter_rows(values_only=True):
                    cells = [(str(c) if c is not None else "").strip() for c in row]
                    line = "\t".join(cells).strip()
                    if line:
                        parts.append(line)
            return "\n".join(parts)

    # Fallback (should not happen)
    return ""


def chunk_text(text: str, chunk_size: int = 1200, chunk_overlap: int = 200) -> List[str]:
    """
    Split text into overlapping chunks. Attempts to split on paragraph/sentence boundaries,
    then enforces chunk size with overlap.

    Strategy:
      1) Normalize whitespace.
      2) Prefer splitting by double newlines or periods.
      3) Build chunks up to chunk_size (characters) with chunk_overlap between consecutive chunks.

    Note: For PDFs or spreadsheets, text may not have clear sentence boundaries; we still chunk by length.
    """
    text = (text or "").strip()
    if not text:
        return []

    # Normalize whitespace
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    # Try to split by double newlines (paragraphs)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paragraphs) <= 1:
        # Fallback: split by periods
        paragraphs = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

    chunks: List[str] = []
    buffer = ""

    def flush_buffer():
        nonlocal buffer
        if buffer.strip():
            chunks.append(buffer.strip())
        buffer = ""

    for part in paragraphs:
        if len(buffer) + len(part) + 1 <= chunk_size:
            buffer = (buffer + " " + part).strip() if buffer else part
        else:
            # If current part is too long itself, hard-split it
            if not buffer:
                start = 0
                while start < len(part):
                    end = min(start + chunk_size, len(part))
                    chunks.append(part[start:end].strip())
                    start = end - chunk_overlap if end < len(part) else end
                continue
            else:
                flush_buffer()
                # Now add the current part
                if len(part) <= chunk_size:
                    buffer = part
                else:
                    start = 0
                    while start < len(part):
                        end = min(start + chunk_size, len(part))
                        chunks.append(part[start:end].strip())
                        start = end - chunk_overlap if end < len(part) else end

    flush_buffer()

    # Apply overlap between chunks post-hoc if needed (already handled above for hard splits)
    if chunk_overlap > 0 and chunks:
        overlapped: List[str] = []
        for i, ch in enumerate(chunks):
            if i == 0:
                overlapped.append(ch)
            else:
                prev = overlapped[-1]
                # Take suffix of prev and prefix of current to maintain continuity
                suffix = prev[-chunk_overlap:] if len(prev) > chunk_overlap else prev
                merged = (suffix + " " + ch).strip()
                if len(merged) > chunk_size:
                    overlapped.append(ch)
                else:
                    overlapped[-1] = merged
        chunks = overlapped

    # Final cleanup
    chunks = [c.strip() for c in chunks if c.strip()]
    return chunks
