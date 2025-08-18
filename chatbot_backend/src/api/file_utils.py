import io
import os
import json
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
        - .xlsx : openpyxl extraction (sheet-by-sheet parsing, table detection with headers,
                  JSON sample, and TSV preview for robust, structured context)

    For .xlsx, this function:
        - Detects the header row heuristically (first row with >=2 non-empty cells, not mostly numeric)
        - Sanitizes header names (non-empty, unique)
        - Maps each subsequent row to a dict of {column_name: value}
        - Generates a JSON sample (first N rows) and a TSV preview for human readability
        - Returns a consolidated, annotated text summary of all sheets

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


def _coerce_cell_value(val: Any) -> Any:
    """
    Normalize cell values for JSON serialization.
    - Datetimes -> ISO string (openpyxl may already convert with data_only)
    - Numbers, bools -> unchanged
    - None -> None
    - Others -> string
    """
    # Avoid importing datetime to keep runtime lean; str() is adequate for Gemini context
    if val is None:
        return None
    if isinstance(val, (int, float, bool)):
        return val
    # For anything else (dates, decimals, formulas result strings), stringify
    return str(val)


def _is_potential_header(row_vals: List[Any]) -> bool:
    """
    Heuristic to determine whether a row looks like a header:
    - At least 2 non-empty cells
    - Not mostly numeric (header cells typically text-like)
    """
    non_empty = [v for v in row_vals if (v is not None and str(v).strip() != "")]
    if len(non_empty) < 2:
        return False
    numeric_like = 0
    for v in non_empty:
        s = str(v).strip()
        # Count numeric-like tokens (allows floats, ints, percentages)
        if s.replace(".", "", 1).replace("%", "", 1).isdigit():
            numeric_like += 1
    # If more than half of the non-empty are numeric-like, this is likely not a header row
    return numeric_like <= (len(non_empty) // 2)


def _sanitize_headers(raw_headers: List[Any]) -> List[str]:
    """
    Sanitize header names to be non-empty, trimmed, and unique.
    Empty or None header names become 'column_{i}' (1-indexed).
    """
    headers: List[str] = []
    seen: Dict[str, int] = {}
    for idx, h in enumerate(raw_headers, start=1):
        name = str(h).strip() if h is not None and str(h).strip() != "" else f"column_{idx}"
        # Enforce uniqueness
        base = name
        count = seen.get(base, 0)
        if count > 0:
            name = f"{base}_{count+1}"
        seen[base] = count + 1
        headers.append(name)
    return headers


def _safe_int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _extract_xlsx(content: bytes) -> str:
    """
    Extract text from XLSX using openpyxl with robust, structured context.

    Enhancements vs. basic TSV:
    - Detect header row and map subsequent rows to column names.
    - Produce a JSON sample (first N rows per sheet) with clean types.
    - Include a TSV preview for readability.
    - Annotate with sheet name, columns, and approximate row counts.

    Performance safeguards:
    - Uses read_only=True and data_only=True to stream rows.
    - Respects environment-configurable limits to prevent timeouts on very large files:
        FILE_EXTRACT_MAX_XLSX_CELLS (int): total maximum cells to process across the workbook (default: 200000)
        FILE_EXTRACT_MAX_XLSX_ROWS_PER_SHEET (int): per-sheet row cap (default: 20000)
        FILE_EXTRACT_XLSX_JSON_SAMPLE_ROWS (int): number of records to include in JSON sample per sheet (default: 50)
        FILE_EXTRACT_XLSX_TSV_PREVIEW_ROWS (int): number of TSV rows to preview per sheet (default: 20)
    When limits are hit, extraction stops gracefully for speed and reliability.
    """
    # Read limits from environment with sensible defaults
    max_cells_total = _safe_int_env("FILE_EXTRACT_MAX_XLSX_CELLS", 200000)
    max_rows_per_sheet = _safe_int_env("FILE_EXTRACT_MAX_XLSX_ROWS_PER_SHEET", 20000)
    json_sample_rows = _safe_int_env("FILE_EXTRACT_XLSX_JSON_SAMPLE_ROWS", 50)
    tsv_preview_rows = _safe_int_env("FILE_EXTRACT_XLSX_TSV_PREVIEW_ROWS", 20)

    bio = io.BytesIO(content)
    wb = load_workbook(bio, data_only=True, read_only=True)
    parts: List[str] = []
    processed_cells = 0

    for ws in wb.worksheets:
        try:
            parts.append(f"[Sheet: {ws.title}]")
            header_found = False
            headers: List[str] = []
            row_index = 0
            # To process potential header + preceding buffer elegantly
            buffered_rows: List[List[Any]] = []
            sample_records: List[Dict[str, Any]] = []
            tsv_lines: List[str] = []
            parsed_data_rows = 0
            truncated_notice_added = False

            for row in ws.iter_rows(values_only=True):
                row_index += 1
                if row_index > max_rows_per_sheet:
                    parts.append("[...] (Row limit reached for this sheet)")
                    break

                # Convert the row to a standard list
                row_vals = list(row) if row is not None else []
                # Update global cell counter
                processed_cells += len(row_vals)
                if processed_cells > max_cells_total and not truncated_notice_added:
                    parts.append("[...] (Global cell limit reached; workbook parsing truncated)")
                    truncated_notice_added = True
                    # Exit entire workbook parsing
                    break

                # Buffer rows until header is found
                buffered_rows.append(row_vals)

                if not header_found:
                    if _is_potential_header(row_vals):
                        # Treat current buffered row as header
                        headers = _sanitize_headers(row_vals)
                        header_found = True

                        # TSV header line for preview
                        tsv_lines.append("\t".join(headers))

                        # Process buffered rows after header row as data
                        for data_row in buffered_rows[buffered_rows.index(row_vals) + 1 :]:
                            # Skip entirely empty rows
                            if not any(v is not None and str(v).strip() != "" for v in data_row):
                                continue
                            record = _row_to_record(headers, data_row)
                            parsed_data_rows += 1

                            # JSON sample
                            if len(sample_records) < json_sample_rows:
                                sample_records.append(record)
                            # TSV preview
                            if len(tsv_lines) < tsv_preview_rows + 1:  # +1 for header
                                tsv_lines.append(_record_to_tsv(headers, record))

                        # clear buffer to conserve memory
                        buffered_rows = []
                    else:
                        # Continue scanning for header
                        continue
                else:
                    # Header already found; process current row as data
                    if not any(v is not None and str(v).strip() != "" for v in row_vals):
                        continue
                    record = _row_to_record(headers, row_vals)
                    parsed_data_rows += 1

                    if len(sample_records) < json_sample_rows:
                        sample_records.append(record)
                    if len(tsv_lines) < tsv_preview_rows + 1:  # +1 for header
                        tsv_lines.append(_record_to_tsv(headers, record))

                if processed_cells > max_cells_total:
                    # Global cap reached; stop processing this sheet
                    break

            # If we never found a header but we have some non-empty row, fallback: use first non-empty row as header
            if not header_found:
                # Find first non-empty buffered row
                first_non_empty_idx = None
                for i, r in enumerate(buffered_rows):
                    if any(v is not None and str(v).strip() != "" for v in r):
                        first_non_empty_idx = i
                        break
                if first_non_empty_idx is not None:
                    raw_headers = buffered_rows[first_non_empty_idx]
                    headers = _sanitize_headers(raw_headers)
                    header_found = True
                    tsv_lines.append("\t".join(headers))
                    # Process remaining as data
                    for data_row in buffered_rows[first_non_empty_idx + 1 :]:
                        if not any(v is not None and str(v).strip() != "" for v in data_row):
                            continue
                        record = _row_to_record(headers, data_row)
                        parsed_data_rows += 1
                        if len(sample_records) < json_sample_rows:
                            sample_records.append(record)
                        if len(tsv_lines) < tsv_preview_rows + 1:
                            tsv_lines.append(_record_to_tsv(headers, record))

            # Build sheet summary
            if header_found:
                parts.append(f"Columns ({len(headers)}): {json.dumps(headers, ensure_ascii=False)}")
                parts.append(f"Parsed data rows (approx): {parsed_data_rows}")
                # JSON sample
                parts.append("JSON sample (first rows):")
                parts.append(json.dumps(sample_records, ensure_ascii=False, indent=2))
                # TSV preview
                parts.append("TSV preview:")
                parts.append("\n".join(tsv_lines))
            else:
                parts.append("No tabular data detected (sheet appears empty or formatting not recognized).")

            parts.append("")  # blank line between sheets

            if processed_cells > max_cells_total:
                break

        except Exception as sheet_error:
            # Do not fail the entire extraction if one sheet has issues
            parts.append(f"[Sheet: {ws.title}]")
            parts.append(f"Error parsing sheet: {sheet_error}")
            parts.append("")

    return "\n".join(parts).strip()


def _row_to_record(headers: List[str], row_vals: List[Any]) -> Dict[str, Any]:
    """
    Map a row list to a dict using the provided headers. Pads/truncates safely.
    Values are coerced for JSON friendliness.
    """
    record: Dict[str, Any] = {}
    max_len = len(headers)
    for i in range(max_len):
        key = headers[i]
        val = row_vals[i] if i < len(row_vals) else None
        record[key] = _coerce_cell_value(val)
    return record


def _record_to_tsv(headers: List[str], record: Dict[str, Any]) -> str:
    """
    Convert a record dict to a TSV line in header order.
    """
    vals: List[str] = []
    for h in headers:
        v = record.get(h, "")
        if v is None:
            vals.append("")
        else:
            s = str(v)
            # Avoid newlines/tabs in TSV cell
            s = s.replace("\n", " ").replace("\r", " ").replace("\t", " ")
            vals.append(s)
    return "\t".join(vals)


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
