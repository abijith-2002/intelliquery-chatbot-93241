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
from itertools import islice


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


def _group_headers_by_theme(headers: List[str]) -> Dict[str, List[int]]:
    """
    Group column indices by header 'theme' using simple heuristics.
    Theme is inferred by common prefixes before delimiters like ':', '-', or by shared keywords.

    Returns:
        dict: {theme: [zero_based_col_indices]}
    """
    import re
    themes: Dict[str, List[int]] = {}
    for idx, h in enumerate(headers):
        base = (h or "").strip()
        # Normalize
        key = base.lower()
        # Try to extract thematic prefix
        m = re.match(r"([a-z0-9 ]+)[ _:-]+.*", key)
        theme = m.group(1).strip() if m else key
        # Guard: if theme too short, use full header
        if len(theme) < 3:
            theme = key
        themes.setdefault(theme, []).append(idx)
    return themes


def _fixed_size_column_groups(num_cols: int, group_size: int = 100) -> List[List[int]]:
    """
    Build equal-sized groups of zero-based column indices.
    """
    groups: List[List[int]] = []
    start = 0
    while start < num_cols:
        end = min(num_cols, start + group_size)
        groups.append(list(range(start, end)))
        start = end
    return groups


# PUBLIC_INTERFACE
def build_xlsx_column_chunks_from_schema(
    schema: Dict[str, Any],
    group_size: int = 100,
    prefer_theme_groups: bool = True,
) -> List[Dict[str, Any]]:
    """
    PUBLIC_INTERFACE
    Given a large-workbook schema (as returned by extract_xlsx_schema_or_text),
    compute column chunk groups for each sheet.

    If prefer_theme_groups is True, attempt to group columns by header theme; if that
    yields too many tiny groups or is ineffective, fall back to fixed-size groups.

    Returns:
        List[Dict[str, Any]]:
            [
              {
                "sheet": "Sheet1",
                "group_type": "theme" | "fixed",
                "group_label": "orders" | "columns_1_100",
                "column_indices": [0,1,2,...],   # zero-based indices
                "column_names": ["Order ID", "Order Date", ...]
              },
              ...
            ]
    """
    chunks: List[Dict[str, Any]] = []
    if not schema or "sheets" not in schema:
        return chunks

    for sheet in schema.get("sheets", []):
        headers = [c.get("name", f"Column_{i+1}") for i, c in enumerate(sheet.get("columns", []))]
        num_cols = len(headers)

        used_groups: List[List[int]] = []
        group_type = "fixed"
        labels: List[str] = []

        if prefer_theme_groups and headers:
            theme_map = _group_headers_by_theme(headers)
            # Consider theme grouping effective if average group size >= 3 or max group size >= group_size/2
            group_lists = list(theme_map.values())
            if group_lists:
                avg = sum(len(g) for g in group_lists) / len(group_lists)
                mx = max(len(g) for g in group_lists)
                if avg >= 3 or mx >= max(10, group_size // 2):
                    used_groups = [sorted(g) for g in group_lists]
                    group_type = "theme"
                    labels = list(theme_map.keys())

        if not used_groups:
            used_groups = _fixed_size_column_groups(num_cols, group_size=group_size)
            group_type = "fixed"
            labels = [f"columns_{g[0]+1}_{g[-1]+1}" for g in used_groups if g]

        # Build chunk descriptors
        for i, g in enumerate(used_groups):
            if not g:
                continue
            column_names = [headers[j] for j in g]
            label = labels[i] if i < len(labels) else (f"{group_type}_{i+1}")
            chunks.append(
                {
                    "sheet": sheet.get("name", "Sheet"),
                    "group_type": group_type,
                    "group_label": label,
                    "column_indices": g,
                    "column_names": column_names,
                }
            )
    return chunks


# PUBLIC_INTERFACE
def iter_xlsx_row_slices(
    filename: str,
    content: bytes,
    per_row_max_cols: int = 200,
    group_size: int = 100,
    max_rows: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    PUBLIC_INTERFACE
    Iterate an .xlsx and produce minimal 'record slices' for very wide rows.

    Behavior:
      - For each worksheet, read the header (first row) and subsequent data rows.
      - If the number of columns > per_row_max_cols, split each row into multiple
        column groups (slices) using fixed-size groups of length `group_size`.
      - For narrower rows, a single slice containing all non-empty columns is produced.
      - Each slice is a minimal text unit suitable for indexing/embedding.

    Args:
        filename: Original filename for traceability.
        content: Raw xlsx bytes.
        per_row_max_cols: Threshold of columns beyond which a row is sliced.
        group_size: Number of columns per slice when slicing is needed.
        max_rows: Optional cap on number of data rows to process per sheet (for safety).

    Returns:
        List[Dict[str, Any]]: A list of slice descriptors:
            {
              "sheet": str,
              "row_index": int,  # 1-based Excel row number
              "slice_index": int,  # 1-based slice index within the row
              "column_indices": List[int],  # zero-based indices included in this slice
              "text": str,  # rendered minimal text for the slice
              "meta": {
                  "filename": str,
                  "group_label": str,  # e.g., columns_1_100
              }
            }
    """
    bio = io.BytesIO(content)
    wb = load_workbook(bio, data_only=True, read_only=True)

    slices: List[Dict[str, Any]] = []
    for ws in wb.worksheets:
        # Header row (names)
        header_cells = next(ws.iter_rows(min_row=1, max_row=1, values_only=True), None)
        headers: List[str] = []
        if header_cells:
            headers = [str(c) if c is not None else f"Column_{idx+1}" for idx, c in enumerate(header_cells)]
        num_cols = ws.max_column or len(headers)
        if headers and num_cols < len(headers):
            num_cols = len(headers)

        # Decide if row slicing is needed based on width
        needs_slicing = (num_cols or 0) > per_row_max_cols
        column_groups: List[List[int]]
        if needs_slicing:
            column_groups = _fixed_size_column_groups(num_cols, group_size=group_size)
        else:
            column_groups = [list(range(0, num_cols))] if num_cols else []

        # Iterate data rows
        start_row = 2
        end_row = ws.max_row or 1
        row_iter = ws.iter_rows(min_row=start_row, max_row=end_row, values_only=True)
        if max_rows is not None and max_rows > 0:
            row_iter = islice(row_iter, 0, max_rows)

        for idx_offset, row_vals in enumerate(row_iter):
            excel_row_num = start_row + idx_offset
            # Normalize row values list to num_cols length
            row_list = list(row_vals or [])
            if len(row_list) < num_cols:
                row_list = row_list + [None] * (num_cols - len(row_list))

            # Skip fully empty rows
            if not any((str(v).strip() if v is not None else "") for v in row_list):
                continue

            # Build text slices per group
            for s_idx, g in enumerate(column_groups, start=1):
                if not g:
                    continue
                cols_text_parts: List[str] = []
                for j in g:
                    col_name = headers[j] if j < len(headers) else f"Column_{j+1}"
                    val = row_list[j]
                    val_str = "" if val is None else str(val)
                    if val_str.strip() == "":
                        continue
                    cols_text_parts.append(f"{col_name}: {val_str}")
                # If nothing non-empty in this slice, skip to avoid noisy empty chunks
                if not cols_text_parts:
                    continue
                group_label = f"columns_{g[0]+1}_{g[-1]+1}"
                text = f"[{filename} | {ws.title} | Row {excel_row_num} | {group_label}]\n" + "; ".join(cols_text_parts)
                slices.append(
                    {
                        "sheet": ws.title,
                        "row_index": excel_row_num,
                        "slice_index": s_idx,
                        "column_indices": g,
                        "text": text,
                        "meta": {
                            "filename": filename,
                            "group_label": group_label,
                        },
                    }
                )
    return slices


# PUBLIC_INTERFACE
def render_xlsx_column_chunk_text(
    filename: str,
    sheet: str,
    group_label: str,
    column_names: List[str],
) -> str:
    """
    PUBLIC_INTERFACE
    Render a lightweight textual representation for a column chunk to be embedded/indexed.

    This avoids scanning all rows of the Excel file; it relies on column names and types in schema,
    focusing the embedding on the 'theme' of the columns.

    Returns:
        str: A concise description suitable for RAG indexing.
    """
    names_preview = ", ".join(column_names[:50])
    return f"[{filename} | {sheet} | Columns: {group_label}]\n{names_preview}"


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
