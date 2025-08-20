import io
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
            # For .xlsx, we keep behavior of returning textual content for small files.
            # Large files are detected and summarized as schema only by extract_xlsx_schema_or_text().
            return _extract_xlsx_text(content), None
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


def _extract_xlsx_text(content: bytes) -> str:
    """
    Extract text from XLSX using openpyxl (sheet by sheet, TSV rows).
    Note: This function reads through all cells; for large files, prefer using
    extract_xlsx_schema_or_text which can summarize schema instead.
    """
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
            if any((v or "").strip() for v in vals):
                parts.append("\t".join(vals))
        parts.append("")  # blank line between sheets
    return "\n".join(parts).strip()


def _infer_column_type(samples: List[Any]) -> str:
    """
    Infer a column type from a list of sample values.
    Priority:
        - if all null -> "null"
        - if all bool -> "boolean"
        - if all numbers -> "number"
        - if all datetime/date/time -> "datetime"
        - else -> "string"
    """
    non_null = [s for s in samples if s not in (None, "")]
    if not non_null:
        return "null"
    if all(isinstance(s, bool) for s in non_null):
        return "boolean"
    if all(isinstance(s, (int, float)) for s in non_null):
        return "number"
    import datetime as _dt
    if all(isinstance(s, (_dt.date, _dt.datetime, _dt.time)) for s in non_null):
        return "datetime"
    return "string"


# PUBLIC_INTERFACE
def extract_xlsx_schema_or_text(
    filename: str,
    content: bytes,
    row_threshold: int = 100,
    col_threshold: int = 100,
) -> Tuple[str, Optional[Dict[str, Any]], Optional[str]]:
    """
    PUBLIC_INTERFACE
    Smart extractor for .xlsx files:
    - If workbook is small (<= row_threshold rows and <= col_threshold columns per sheet),
      return full text and no schema (schema=None).
    - If large (any sheet exceeds thresholds), avoid full scan and instead return a schema summary
      describing sheet names, column headers, inferred data types, and sample values per column.

    Args:
        filename: Name of the file for logging/identification.
        content: Raw xlsx bytes.
        row_threshold: Threshold for number of data rows (excluding header) to be considered large.
        col_threshold: Threshold for number of columns to be considered large.

    Returns:
        Tuple[str, Optional[Dict[str, Any]], Optional[str]]:
            - text: If small, the textual TSV-style content; if large, a formatted schema summary string.
            - schema: Structured JSON schema when large; otherwise None for small files.
            - error: Error string if any failure occurred; otherwise None.
    """
    try:
        bio = io.BytesIO(content)
        wb = load_workbook(bio, data_only=True, read_only=True)
    except Exception as e:
        return "", None, f"Failed to open Excel file '{filename}': {e}"

    # Detect large using worksheet metadata
    is_large = False
    for ws in wb.worksheets:
        try:
            rows = ws.max_row or 0
            cols = ws.max_column or 0
            if rows > row_threshold or cols > col_threshold:
                is_large = True
                break
        except Exception:
            # If metadata fetch fails, treat as large for safety
            is_large = True
            break

    if not is_large:
        # Small workbook: return full text (TSV style), schema None
        try:
            text = _extract_xlsx_text(content)
            return text, None, None
        except Exception as e:
            return "", None, f"Failed to extract Excel text from '{filename}': {e}"

    # Large workbook: schema only
    schema: Dict[str, Any] = {
        "filename": filename,
        "sheets": [],
        "notes": f"Detected as large workbook (> {row_threshold} rows or > {col_threshold} columns); returning schema only.",
    }

    lines_for_human_summary: List[str] = []
    try:
        for ws in wb.worksheets:
            sheet_info: Dict[str, Any] = {"name": ws.title, "columns": []}

            # Header (first row)
            header_cells = next(ws.iter_rows(min_row=1, max_row=1, values_only=True), None)
            headers: List[str] = []
            if header_cells:
                headers = [str(c) if c is not None else f"Column_{idx+1}" for idx, c in enumerate(header_cells)]

            # Sample a few rows after header for type inference
            sample_rows: List[tuple] = []
            max_samples = 5
            if (ws.max_row or 0) >= 2:
                for row in ws.iter_rows(min_row=2, max_row=min(1 + max_samples, ws.max_row), values_only=True):
                    sample_rows.append(row)

            # Decide column count
            num_cols = ws.max_column or (len(headers) if headers else 0)
            if headers and num_cols < len(headers):
                num_cols = len(headers)

            # Build columns metadata
            for col_idx in range(1, (num_cols or 0) + 1):
                col_name = headers[col_idx - 1] if col_idx - 1 < len(headers) else f"Column_{col_idx}"
                # Collect sample values for this column
                col_samples: List[Any] = []
                for r in sample_rows:
                    if r is None:
                        continue
                    if col_idx - 1 < len(r):
                        col_samples.append(r[col_idx - 1])

                col_type = _infer_column_type(col_samples)
                # Prepare up to 3 printable sample values
                sample_strs: List[str] = []
                for v in col_samples:
                    if v in (None, ""):
                        continue
                    try:
                        sample_strs.append(str(v))
                    except Exception:
                        sample_strs.append("<unprintable>")
                    if len(sample_strs) >= 3:
                        break

                sheet_info["columns"].append(
                    {
                        "name": col_name,
                        "type": col_type,
                        "sample_values": sample_strs,
                    }
                )

            schema["sheets"].append(sheet_info)

            # Human summary for this sheet
            cols_fmt = ", ".join([f"{c['name']} ({c['type']})" for c in sheet_info["columns"][:50]])
            lines_for_human_summary.append(f"Sheet: {ws.title}; Columns: [{cols_fmt}]")

        human_summary = "\n".join(lines_for_human_summary).strip()
        return human_summary, schema, None
    except Exception as e:
        return "", None, f"Failed to extract Excel schema from '{filename}': {e}"


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
