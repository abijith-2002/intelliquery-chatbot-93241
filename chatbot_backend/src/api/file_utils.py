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
        - .xlsx : openpyxl parsing into a JSON array of row objects (keys are column headers). The returned text for .xlsx is the JSON array string.
        - .json : JSON parsing with structure discovery, array-of-objects tabular summary,
                  unique values per key, JSON sample, and TSV preview

    For .xlsx, this function:
        - Detects the header row heuristically (first row with >=2 non-empty cells, not mostly numeric)
        - Sanitizes header names (non-empty, unique)
        - Maps each subsequent row to a dict of {column_name: value}
        - Generates:
            * a JSON sample (first N rows)
            * a TSV preview for human readability
            * per-column Unique values (first M distinct values per column)
        - Returns a consolidated, annotated text summary of all sheets

    For .json, this function:
        - Accepts both a single object and an array of objects
        - If array of objects, treats as a table and produces:
            * Columns (union of keys; dot-notation for nested up to depth 2)
            * Unique values per column (first M)
            * JSON sample (first N records)
            * TSV preview (first K rows)
        - If a single object, enumerates keys/types and, for any array-of-objects fields,
          produces a similar tabular summary per field.

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
        if name_lower.endswith(".json"):
            return _extract_json(content), None
        return "", f"Unsupported file type for '{filename}'. Allowed: .txt, .pdf, .docx, .xlsx, .json"
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
    Extract XLSX content as a JSON array of row objects.

    This implementation converts every data row into a dict keyed by column headers
    and returns a single JSON array string. Headers are detected heuristically and
    sanitized for uniqueness. Sheet name is included per row via the "sheet" key.

    Limits and safeguards are controlled by:
      - FILE_EXTRACT_MAX_XLSX_CELLS
      - FILE_EXTRACT_MAX_XLSX_ROWS_PER_SHEET
      - FILE_EXTRACT_XLSX_JSON_ROWS_MAX

    On failure, an exception is raised so the caller can surface the error.
    """
    # Reuse the dedicated row-wise JSON converter
    rows, err = extract_xlsx_as_rowwise_json("uploaded.xlsx", content, include_sheet_name=True)
    if err:
        raise Exception(err)
    # Serialize to JSON string; the row-wise converter already enforces safe caps
    return json.dumps(rows, ensure_ascii=False)


def _safe_str_len(s: str, max_len: int) -> str:
    """Trim a string to max_len characters with ellipsis."""
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def _flatten_keys(d: Dict[str, Any], parent: str = "", sep: str = ".", depth: int = 0, max_depth: int = 2) -> List[str]:
    """
    Collect flattened keys (dot notation) up to max_depth levels.
    Values that are dicts/lists beyond max_depth are not expanded further.
    """
    keys: List[str] = []
    if not isinstance(d, dict):
        return keys
    for k, v in d.items():
        if not isinstance(k, str):
            k = str(k)
        nk = f"{parent}{sep}{k}" if parent else k
        if isinstance(v, dict) and depth < max_depth:
            keys.extend(_flatten_keys(v, nk, sep, depth + 1, max_depth))
        else:
            keys.append(nk)
    return keys


def _flatten_to_record(obj: Dict[str, Any], max_depth: int = 2, max_str_len: int = 300) -> Dict[str, Any]:
    """
    Flatten a JSON object (dict) to a single level using dot notation for keys up to max_depth.
    Values are coerced to JSON-friendly scalars/strings with trimming.
    """
    record: Dict[str, Any] = {}

    def _walk(o: Any, parent: str = "", depth: int = 0):
        if isinstance(o, dict) and depth < max_depth:
            for kk, vv in o.items():
                key = f"{parent}.{kk}" if parent else str(kk)
                _walk(vv, key, depth + 1)
        else:
            # Coerce value
            v = _coerce_cell_value(o)
            if isinstance(v, str):
                v = _safe_str_len(v, max_str_len)
            record[parent] = v

    if isinstance(obj, dict):
        _walk(obj, "", 0)
    return record


def _json_array_of_objects_summary(
    arr: List[Any],
    title: str = "[JSON Array of objects]",
    json_sample_rows_env: str = "FILE_EXTRACT_JSON_SAMPLE_ROWS",
    tsv_preview_rows_env: str = "FILE_EXTRACT_JSON_TSV_PREVIEW_ROWS",
    unique_limit_env: str = "FILE_EXTRACT_JSON_UNIQUE_LIMIT",
    scan_rows_env: str = "FILE_EXTRACT_JSON_SCAN_ROWS",
) -> str:
    """
    Produce a structured summary for an array of objects:
    - Columns (union of flattened keys up to depth 2)
    - Approx record count
    - Unique values by column (first M string values)
    - JSON sample (first N rows)
    - TSV preview (first K rows)
    """
    json_sample_rows = _safe_int_env(json_sample_rows_env, 50)
    tsv_preview_rows = _safe_int_env(tsv_preview_rows_env, 20)
    unique_limit = _safe_int_env(unique_limit_env, 50)
    scan_rows = _safe_int_env(scan_rows_env, 1000)

    # Determine union of columns from first scan_rows items
    cols_set: set = set()
    count = 0
    for item in arr[:scan_rows]:
        if isinstance(item, dict):
            for k in _flatten_keys(item, max_depth=2):
                cols_set.add(k)
        count += 1
    columns = sorted(list(cols_set))

    # Initialize uniques map
    uniques: Dict[str, set] = {c: set() for c in columns}

    # Build samples and TSV
    sample_records: List[Dict[str, Any]] = []
    tsv_lines: List[str] = []
    if columns:
        tsv_lines.append("\t".join(columns))
    parsed_rows = 0
    for item in arr:
        if not isinstance(item, dict):
            # skip non-dict entries in mixed arrays
            continue
        rec_full = _flatten_to_record(item, max_depth=2)
        # Normalize to target columns
        rec: Dict[str, Any] = {c: rec_full.get(c) for c in columns}
        parsed_rows += 1

        # Update uniques for string-like non-empty values
        _update_uniques(uniques, columns, rec, unique_limit)

        if len(sample_records) < json_sample_rows:
            sample_records.append(rec)
        if len(tsv_lines) < tsv_preview_rows + 1:  # +1 header
            tsv_lines.append(_record_to_tsv(columns, rec))

    parts: List[str] = []
    parts.append(title)
    parts.append(f"Columns ({len(columns)}): {json.dumps(columns, ensure_ascii=False)}")
    parts.append(f"Parsed data rows (approx): {parsed_rows}")
    # Unique values map
    unique_map = {
        col: sorted([v for v in vals if isinstance(v, str) and v.strip() != ""])[: unique_limit]
        for col, vals in uniques.items()
        if any((isinstance(v, str) and v.strip() != "") for v in vals)
    } if uniques else {}
    if unique_map:
        parts.append(f"Unique values by column (first {unique_limit}):")
        parts.append(json.dumps(unique_map, ensure_ascii=False, indent=2))
    # JSON sample
    parts.append("JSON sample (first rows):")
    parts.append(json.dumps(sample_records, ensure_ascii=False, indent=2))
    # TSV preview
    parts.append("TSV preview:")
    parts.append("\n".join(tsv_lines))
    parts.append("")  # spacer
    return "\n".join(parts).strip()


def _extract_json(content: bytes) -> str:
    """
    Extract structured context from a JSON file.

    Behavior:
      - If the root is an array:
          * If elements are objects: treat as a table and output Columns/Unique/JSON sample/TSV preview
          * Else: show element type summary and sample values
      - If the root is an object:
          * List top-level keys and value types
          * For any key whose value is an array of objects, produce a dataset block similar to the table output
          * Include a compact JSON sample of the root object (trimmed)
    """
    raw = content.decode("utf-8", errors="ignore").strip()
    if raw == "":
        return "Empty JSON content."

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        # Raise an exception so the caller wraps it into an error message
        raise Exception(f"Invalid JSON: {e.msg} at line {e.lineno} column {e.colno}")

    parts: List[str] = []
    # Limits
    sample_preview_len = _safe_int_env("FILE_EXTRACT_JSON_STRING_TRIM", 300)

    if isinstance(data, list):
        if not data:
            return "[JSON Array] The array is empty."
        # Check if majority are dicts
        dict_count = sum(1 for x in data if isinstance(x, dict))
        if dict_count >= max(1, len(data) // 2):
            # Treat as table
            summary = _json_array_of_objects_summary(data, title="[JSON Array of objects]")
            parts.append(summary)
        else:
            # Primitives/mixed array
            types = sorted({type(x).__name__ for x in data})
            parts.append(f"[JSON Array of primitives/mixed] Length: {len(data)}")
            parts.append(f"Element types: {types}")
            sample_vals = [x for x in data[:min(len(data), 25)]]
            # Coerce to printable and trim
            sample_strs = []
            for v in sample_vals:
                if isinstance(v, (dict, list)):
                    s = _safe_str_len(json.dumps(v, ensure_ascii=False), sample_preview_len)
                else:
                    s = _safe_str_len(str(v), sample_preview_len)
                sample_strs.append(s)
            parts.append("Sample values:")
            parts.append(json.dumps(sample_strs, ensure_ascii=False, indent=2))
        return "\n".join(parts).strip()

    if isinstance(data, dict):
        keys = list(data.keys())
        parts.append("[JSON Object]")
        parts.append(f"Top-level keys ({len(keys)}): {json.dumps(keys, ensure_ascii=False)}")

        # Key types summary
        type_map = {k: type(data[k]).__name__ for k in keys}
        parts.append("Key types:")
        parts.append(json.dumps(type_map, ensure_ascii=False, indent=2))

        # For array-of-objects fields, produce dataset blocks
        for k in keys:
            v = data.get(k)
            if isinstance(v, list) and v:
                dict_count = sum(1 for x in v if isinstance(x, dict))
                if dict_count >= max(1, len(v) // 2):
                    parts.append(f"[Dataset: {k}]")
                    parts.append(_json_array_of_objects_summary(v, title=f"[Array of objects: {k}]"))
                else:
                    # Primitive array
                    types = sorted({type(x).__name__ for x in v})
                    parts.append(f"[Array field: {k}] Length: {len(v)}; element types: {types}")
                    sample_vals = [x for x in v[:min(len(v), 25)]]
                    sample_strs = []
                    for sv in sample_vals:
                        if isinstance(sv, (dict, list)):
                            s = _safe_str_len(json.dumps(sv, ensure_ascii=False), sample_preview_len)
                        else:
                            s = _safe_str_len(str(sv), sample_preview_len)
                        sample_strs.append(s)
                    parts.append("Sample values:")
                    parts.append(json.dumps(sample_strs, ensure_ascii=False, indent=2))
            elif isinstance(v, dict):
                # Provide flattened nested keys (up to depth 2)
                nested_keys = _flatten_keys(v, parent=k, max_depth=2)
                if nested_keys:
                    parts.append(f"[Nested object keys under '{k}'] count={len(nested_keys)}")
                    parts.append(json.dumps(sorted(nested_keys), ensure_ascii=False, indent=2))

        # Include a compact JSON sample of the root object
        parts.append("JSON sample (object):")
        try:
            parts.append(_safe_str_len(json.dumps(data, ensure_ascii=False, indent=2), 6000))
        except Exception:
            parts.append(_safe_str_len(str(data), 6000))

        return "\n".join(parts).strip()

    # Fallback for unusual JSON types (e.g., str/number at root)
    return f"[JSON Scalar] {type(data).__name__}: {_safe_str_len(str(data), 1000)}"
    
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


def _update_uniques(uniques: Dict[str, set], headers: List[str], record: Dict[str, Any], limit: int) -> None:
    """
    Update per-column unique values set using a record. Collect up to 'limit' distinct values per column.
    Only string-like non-empty values are recorded to support name/listing queries.
    """
    if not uniques:
        return
    for h in headers:
        try:
            if len(uniques.get(h, set())) >= limit:
                continue
            v = record.get(h)
            if v is None:
                continue
            s = str(v).strip()
            if not s:
                continue
            # Normalize whitespace
            s = " ".join(s.split())
            # Initialize set if needed (robustness if headers changed)
            if h not in uniques:
                uniques[h] = set()
            if len(uniques[h]) < limit:
                uniques[h].add(s)
        except Exception:
            # Do not break extraction on edge cases
            continue


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


# PUBLIC_INTERFACE
def extract_xlsx_as_rowwise_json(
    filename: str,
    content: bytes,
    include_sheet_name: bool = True,
    row_limit_env: str = "FILE_EXTRACT_XLSX_JSON_ROWS_MAX",
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    PUBLIC_INTERFACE
    Parse an Excel (.xlsx) workbook and convert it into a JSON array where each object corresponds
    to a row in the spreadsheet, using the detected column headers as keys.

    Notes:
    - This function processes all worksheets.
    - The header row is detected heuristically (first row with at least 2 non-empty cells and not mostly numeric).
    - Header names are sanitized to be non-empty and unique.
    - If include_sheet_name is True, a 'sheet' key is added to each row object with the worksheet title.
    - Empty rows are skipped.
    - Performance safeguards via environment variables:
        * FILE_EXTRACT_MAX_XLSX_CELLS: Max total cells processed across workbook (default: 200000)
        * FILE_EXTRACT_MAX_XLSX_ROWS_PER_SHEET: Max rows per sheet (default: 20000)
        * FILE_EXTRACT_XLSX_JSON_ROWS_MAX (row_limit_env): Max total row objects returned across workbook (default: 15000)

    Args:
        filename (str): The original filename (unused except for context; kept for parity).
        content (bytes): Raw XLSX file bytes.
        include_sheet_name (bool): Whether to add a 'sheet' field per row object.
        row_limit_env (str): Environment variable name that controls the total row cap.

    Returns:
        Tuple[List[Dict[str, Any]], Optional[str]]:
            - rows (List[Dict[str, Any]]): The row-wise JSON array (may be truncated to limit).
            - error (Optional[str]): Error message if parsing failed, otherwise None.
    """
    try:
        max_cells_total = _safe_int_env("FILE_EXTRACT_MAX_XLSX_CELLS", 200000)
        max_rows_per_sheet = _safe_int_env("FILE_EXTRACT_MAX_XLSX_ROWS_PER_SHEET", 20000)
        max_total_rows = _safe_int_env(row_limit_env, 15000)

        bio = io.BytesIO(content)
        wb = load_workbook(bio, data_only=True, read_only=True)

        rows: List[Dict[str, Any]] = []
        processed_cells = 0
        total_rows_accumulated = 0

        for ws in wb.worksheets:
            header_found = False
            headers: List[str] = []
            row_index = 0
            buffered_rows: List[List[Any]] = []

            for row in ws.iter_rows(values_only=True):
                row_index += 1
                if row_index > max_rows_per_sheet or total_rows_accumulated >= max_total_rows:
                    break

                row_vals = list(row) if row is not None else []
                processed_cells += len(row_vals)
                if processed_cells > max_cells_total:
                    break

                buffered_rows.append(row_vals)

                if not header_found:
                    if _is_potential_header(row_vals):
                        headers = _sanitize_headers(row_vals)
                        header_found = True
                        # process buffered data rows after header
                        start_index = buffered_rows.index(row_vals) + 1
                        for data_row in buffered_rows[start_index:]:
                            if total_rows_accumulated >= max_total_rows:
                                break
                            if not any(v is not None and str(v).strip() != "" for v in data_row):
                                continue
                            record = _row_to_record(headers, data_row)
                            if include_sheet_name:
                                record = {"sheet": ws.title, **record}
                            rows.append(record)
                            total_rows_accumulated += 1
                        buffered_rows = []
                    else:
                        continue
                else:
                    if total_rows_accumulated >= max_total_rows:
                        break
                    if not any(v is not None and str(v).strip() != "" for v in row_vals):
                        continue
                    record = _row_to_record(headers, row_vals)
                    if include_sheet_name:
                        record = {"sheet": ws.title, **record}
                    rows.append(record)
                    total_rows_accumulated += 1

            # Fallback: if header never found but non-empty rows exist, take first non-empty as header.
            if not header_found and total_rows_accumulated < max_total_rows:
                first_non_empty_idx = None
                for i, r in enumerate(buffered_rows):
                    if any(v is not None and str(v).strip() != "" for v in r):
                        first_non_empty_idx = i
                        break
                if first_non_empty_idx is not None:
                    raw_headers = buffered_rows[first_non_empty_idx]
                    headers = _sanitize_headers(raw_headers)
                    header_found = True
                    for data_row in buffered_rows[first_non_empty_idx + 1:]:
                        if total_rows_accumulated >= max_total_rows:
                            break
                        if not any(v is not None and str(v).strip() != "" for v in data_row):
                            continue
                        record = _row_to_record(headers, data_row)
                        if include_sheet_name:
                            record = {"sheet": ws.title, **record}
                        rows.append(record)
                        total_rows_accumulated += 1

            if processed_cells > max_cells_total or total_rows_accumulated >= max_total_rows:
                break

        return rows, None
    except Exception as e:
        return [], f"Failed to parse XLSX rows for '{filename}': {e}"


# PUBLIC_INTERFACE
def extract_json_as_rowwise_records(
    filename: str,
    content: bytes,
    row_limit_env: str = "FILE_EXTRACT_JSON_ROWS_MAX",
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    PUBLIC_INTERFACE
    Parse a JSON file into row-wise records suitable for RAG.

    Behavior:
      - If the JSON root is an array:
          * For each element that is an object, flatten up to depth 2 and include as a record.
          * Mixed arrays are supported; non-object elements are skipped.
      - If the JSON root is an object:
          * Search for the largest array-of-objects field and flatten those objects into records.
          * If no array-of-objects field exists, returns [] with an explanatory error.

    Limits:
      - FILE_EXTRACT_JSON_ROWS_MAX controls the maximum number of records (default: 15000).

    Args:
        filename (str): The file name (for error context).
        content (bytes): Raw JSON bytes.
        row_limit_env (str): Environment variable name that caps number of records.

    Returns:
        Tuple[List[Dict[str, Any]], Optional[str]]:
            - rows: List of flattened record dicts.
            - error: None on success, else an explanatory message.
    """
    try:
        max_total_rows = _safe_int_env(row_limit_env, 15000)
        raw = content.decode("utf-8", errors="ignore").strip()
        if raw == "":
            return [], "Empty JSON content."

        data = json.loads(raw)

        def _take_from_array(arr: List[Any]) -> List[Dict[str, Any]]:
            out: List[Dict[str, Any]] = []
            for item in arr:
                if isinstance(item, dict):
                    out.append(_flatten_to_record(item, max_depth=2))
                    if len(out) >= max_total_rows:
                        break
            return out

        # Root array
        if isinstance(data, list):
            rows = _take_from_array(data)
            if rows:
                return rows, None
            return [], "JSON array does not contain object items to extract."

        # Root object: find largest array-of-objects
        if isinstance(data, dict):
            best_key: Optional[str] = None
            best_len = 0
            for k, v in data.items():
                if isinstance(v, list) and v:
                    dict_count = sum(1 for x in v if isinstance(x, dict))
                    if dict_count > 0 and len(v) > best_len:
                        best_key = k
                        best_len = len(v)
            if best_key is not None:
                rows = _take_from_array(data[best_key])
                if rows:
                    return rows, None
                return [], f"Array field '{best_key}' does not contain extractable object items."
            return [], "JSON object does not contain an array of objects to extract."

        return [], "Unsupported JSON structure for row-wise extraction."

    except json.JSONDecodeError as e:
        return [], f"Invalid JSON: {e.msg} at line {e.lineno} column {e.colno}"
    except Exception as e:
        return [], f"Failed to parse JSON rows for '{filename}': {e}"


def _value_type_name(v: Any) -> str:
    """Return a concise JSON type name for value v."""
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "boolean"
    if isinstance(v, (int, float)):
        return "number"
    if isinstance(v, str):
        return "string"
    if isinstance(v, list):
        return "array"
    if isinstance(v, dict):
        return "object"
    return type(v).__name__


def _detect_array_is_objects(arr: List[Any]) -> bool:
    """True if majority of elements are dicts."""
    if not arr:
        return False
    dict_count = sum(1 for x in arr if isinstance(x, dict))
    return dict_count >= max(1, len(arr) // 2)


def _collect_types_for_fields(records: List[Dict[str, Any]]) -> Dict[str, str]:
    """Infer a primary type per field from sample records."""
    type_map: Dict[str, str] = {}
    for rec in records:
        for k, v in rec.items():
            t = _value_type_name(v)
            # Prefer a stable type, but if multiple seen, mark as 'mixed'
            if k not in type_map:
                type_map[k] = t
            elif type_map[k] != t:
                type_map[k] = "mixed"
    return type_map


def _build_uniques_map(records: List[Dict[str, Any]], columns: List[str], limit: int) -> Dict[str, List[str]]:
    """Collect up to 'limit' unique non-empty string representations per column."""
    uniques: Dict[str, set] = {c: set() for c in columns}
    for rec in records:
        _update_uniques(uniques, columns, rec, limit)
    # Convert to sorted lists
    return {
        col: sorted([v for v in vals if isinstance(v, str) and v.strip() != ""])[: limit]
        for col, vals in uniques.items()
        if any((isinstance(v, str) and v.strip() != "") for v in vals)
    }


def _chunk_text_block(title: str, body_lines: List[str]) -> str:
    """Construct a readable chunk text with a title header and body lines."""
    lines = [title]
    lines.extend(body_lines)
    return "\n".join(lines).strip()


def _cap_list(items: List[Any], limit: int) -> List[Any]:
    return items[:limit] if len(items) > limit else items


# PUBLIC_INTERFACE
def generate_json_rag_chunks(
    filename: str,
    content: bytes,
    *,
    record_sample_limit: int = None,
    unique_values_limit: int = None,
    chunks_max: int = None,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    PUBLIC_INTERFACE
    Generate embeddable chunks from a JSON file with adaptive, schema-aware strategy.

    Produces:
      - Schema overview chunks for array-of-objects datasets (root or nested)
      - Per-field chunks containing type and unique/sample values
      - Sample rows chunks, grouped to keep examples compact
      - For nested objects/arrays under a root object, creates path-prefixed chunks
      - For primitive/mixed arrays, produces an element-types summary with sample values

    Each chunk is a dict:
      {
        "text": "<embeddable text>",
        "meta": {
            "type": "json_schema" | "json_field" | "json_rows" | "json_array_summary" | "json_nested_keys",
            "filename": "<filename>",
            "dataset": "<root or object key path>",
            "path": "<field dot path>" (for json_field),
            "field_type": "<string|number|boolean|null|object|array|mixed>",
            "approx_count": <int> (rows or elements),
        }
      }

    Limits (overridable via env or parameters):
      - FILE_EXTRACT_JSON_SCAN_ROWS (default 1000): Rows scanned to infer columns
      - FILE_EXTRACT_JSON_SAMPLE_ROWS (default 50): Sample rows count per dataset
      - FILE_EXTRACT_JSON_UNIQUE_LIMIT (default 50): Max unique values per field
      - FILE_JSON_CHUNKS_MAX (default 200): Cap on total chunks returned

    Returns:
      Tuple[List[Dict[str, Any]], Optional[str]]: (chunks, error)
    """
    try:
        raw = (content or b"").decode("utf-8", errors="ignore").strip()
        if raw == "":
            return [], "Empty JSON content."
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        return [], f"Invalid JSON: {e.msg} at line {e.lineno} column {e.colno}"
    except Exception as e:
        return [], f"Failed to parse JSON: {e}"

    scan_rows = _safe_int_env("FILE_EXTRACT_JSON_SCAN_ROWS", 1000)
    sample_rows = record_sample_limit if record_sample_limit is not None else _safe_int_env("FILE_EXTRACT_JSON_SAMPLE_ROWS", 50)
    unique_limit = unique_values_limit if unique_values_limit is not None else _safe_int_env("FILE_EXTRACT_JSON_UNIQUE_LIMIT", 50)
    chunks_cap = chunks_max if chunks_max is not None else _safe_int_env("FILE_JSON_CHUNKS_MAX", 200)

    chunks: List[Dict[str, Any]] = []

    def add_chunk(text: str, meta: Dict[str, Any]):
        if not text or not text.strip():
            return
        if len(chunks) >= chunks_cap:
            return
        chunks.append({"text": text.strip(), "meta": {"filename": filename, **meta}})

    def from_array_of_objects(arr: List[Any], dataset_name: str):
        # Flatten sample records
        flattened: List[Dict[str, Any]] = []
        for item in arr:
            if isinstance(item, dict):
                flattened.append(_flatten_to_record(item, max_depth=2))
                if len(flattened) >= scan_rows:
                    break
        # Columns and types
        columns_set = set()
        for rec in flattened:
            for k in rec.keys():
                columns_set.add(k)
        columns = sorted(list(columns_set))
        type_map = _collect_types_for_fields(flattened)
        approx_count = len([x for x in arr if isinstance(x, dict)])

        # Schema overview chunk
        schema_lines = [
            f"Dataset: {dataset_name}",
            f"Columns ({len(columns)}): {json.dumps(columns, ensure_ascii=False)}",
            "Field types:",
            json.dumps(type_map, ensure_ascii=False, indent=2),
            f"Approximate row count (objects): {approx_count}",
        ]
        add_chunk(
            _chunk_text_block("JSON Schema Overview", schema_lines),
            {"type": "json_schema", "dataset": dataset_name, "approx_count": approx_count},
        )

        # Per-field unique values chunks
        uniques_map = _build_uniques_map(flattened, columns, unique_limit)
        for fld in columns:
            # Per-field type and uniques
            ftype = type_map.get(fld, "unknown")
            uniques = uniques_map.get(fld, [])
            field_lines = [
                f"Field: {fld}",
                f"Type: {ftype}",
            ]
            if uniques:
                field_lines.append(f"Unique values (first {len(uniques)}):")
                field_lines.append(json.dumps(uniques, ensure_ascii=False, indent=2))
            add_chunk(
                _chunk_text_block("JSON Field Details", field_lines),
                {"type": "json_field", "dataset": dataset_name, "path": fld, "field_type": ftype, "approx_count": approx_count},
            )

            if len(chunks) >= chunks_cap:
                break
        # Sample rows chunk(s)
        if flattened:
            sample = _cap_list(flattened, sample_rows)
            rows_text = json.dumps(sample, ensure_ascii=False, indent=2)
            add_chunk(
                _chunk_text_block("JSON Sample Rows", [f"Dataset: {dataset_name}", rows_text]),
                {"type": "json_rows", "dataset": dataset_name, "approx_count": approx_count},
            )

        # Special indexing for train routes: generate directed station-pair keys to support "SRC to DST" queries.
        try:
            # Detect if items look like trains with a 'route' array of station dicts
            route_like_count = 0
            for it in arr:
                if isinstance(it, dict) and isinstance(it.get("route"), list):
                    route_like_count += 1
            if route_like_count > 0:
                for it in arr:
                    if not isinstance(it, dict):
                        continue
                    route = it.get("route")
                    if not isinstance(route, list) or not route:
                        continue
                    # Collect station codes and names in order
                    codes: List[str] = []
                    names: List[str] = []
                    for stop in route:
                        if not isinstance(stop, dict):
                            continue
                        code = (stop.get("station_code") or "").strip().upper()
                        name = (stop.get("station_name") or "").strip()
                        if code or name:
                            codes.append(code)
                            names.append(name)
                    if not codes:
                        continue
                    train_name = str(it.get("name") or it.get("train_name") or it.get("title") or "").strip()
                    train_num = str(it.get("number") or it.get("train_number") or "").strip()

                    # Build adjacent pairs and a capped set of all directed pairs (i<j)
                    pairs_adj: List[str] = []
                    for i in range(len(codes) - 1):
                        a, b = codes[i], codes[i + 1]
                        if a and b:
                            pairs_adj.append(f"{a}->{b}")
                    pairs_all: List[str] = []
                    cap_all = 120  # cap to avoid explosion per train
                    for i in range(len(codes)):
                        for j in range(i + 1, len(codes)):
                            a, b = codes[i], codes[j]
                            if a and b:
                                pairs_all.append(f"{a}->{b}")
                                if len(pairs_all) >= cap_all:
                                    break
                        if len(pairs_all) >= cap_all:
                            break

                    # Prepare lexical variants to improve matching for "to" or "-" phrasing
                    pairs_all_to = [p.replace("->", " to ") for p in pairs_all]
                    pairs_all_dash = [p.replace("->", "-") for p in pairs_all]

                    seq_display = " -> ".join([c if c else (names[idx] if idx < len(names) else "") for idx, c in enumerate(codes)])
                    lines = [
                        "Train Route Segment Index",
                        f"Train: {train_name} ({train_num})".strip(),
                        f"Route sequence (codes): {seq_display}",
                    ]
                    if pairs_adj:
                        lines.append(f"Adjacent segments: {'; '.join(pairs_adj)}")
                    if pairs_all:
                        # Include a subset of all directed pairs and lexical variants
                        lines.append(f"Directed pairs (subset): {'; '.join(pairs_all)}")
                        lines.append(f"Pairs (to-phrase): {'; '.join(pairs_all_to)}")
                        lines.append(f"Pairs (dash): {'; '.join(pairs_all_dash)}")
                    add_chunk(
                        _chunk_text_block("Train Route Pairs", lines),
                        {"type": "route_pairs", "dataset": dataset_name, "path": "route", "approx_count": len(route)},
                    )
                    if len(chunks) >= chunks_cap:
                        break
        except Exception:
            # Do not fail chunking on any route indexing error
            pass

    def from_primitive_or_mixed_array(arr: List[Any], dataset_name: str):
        types = sorted({ _value_type_name(x) for x in arr })
        sample_vals = []
        for v in arr[:min(len(arr), 50)]:
            if isinstance(v, (dict, list)):
                try:
                    sample_vals.append(json.dumps(v, ensure_ascii=False))
                except Exception:
                    sample_vals.append(str(v))
            else:
                sample_vals.append(str(v))
        lines = [
            f"Array: {dataset_name}",
            f"Length: {len(arr)}",
            f"Element types: {types}",
            "Sample values:",
            json.dumps(sample_vals, ensure_ascii=False, indent=2),
        ]
        add_chunk(
            _chunk_text_block("JSON Array Summary", lines),
            {"type": "json_array_summary", "dataset": dataset_name, "approx_count": len(arr)},
        )

    def process_node(node: Any, dataset_name: str):
        # Adaptive behavior per node type
        if isinstance(node, list):
            if _detect_array_is_objects(node):
                from_array_of_objects(node, dataset_name)
            else:
                from_primitive_or_mixed_array(node, dataset_name)
        elif isinstance(node, dict):
            # Nested keys overview
            nested_keys = _flatten_keys(node, parent="", max_depth=2)
            if nested_keys:
                add_chunk(
                    _chunk_text_block(
                        "JSON Nested Keys",
                        [f"Under: {dataset_name}", f"Keys ({len(nested_keys)}):", json.dumps(sorted(nested_keys), ensure_ascii=False, indent=2)],
                    ),
                    {"type": "json_nested_keys", "dataset": dataset_name, "approx_count": len(nested_keys)},
                )
            # Explore child arrays/objects
            for k, v in node.items():
                ds = f"{dataset_name}.{k}" if dataset_name else k
                if isinstance(v, (list, dict)):
                    process_node(v, ds)

    # Root handling
    if isinstance(data, list):
        if _detect_array_is_objects(data):
            process_node(data, dataset_name="root")
        else:
            from_primitive_or_mixed_array(data, dataset_name="root")
    elif isinstance(data, dict):
        # High-level summary for root keys/types
        keys = list(data.keys())
        type_map = {k: _value_type_name(data[k]) for k in keys}
        add_chunk(
            _chunk_text_block(
                "JSON Object Overview",
                [
                    f"Top-level keys ({len(keys)}): {json.dumps(keys, ensure_ascii=False)}",
                    "Key types:",
                    json.dumps(type_map, ensure_ascii=False, indent=2),
                ],
            ),
            {"type": "json_schema", "dataset": "root", "approx_count": len(keys)},
        )
        # Visit children
        process_node(data, dataset_name="root")
    else:
        # Scalar root
        add_chunk(
            f"JSON Scalar root of type { _value_type_name(data) }: {str(data)[:500]}",
            {"type": "json_array_summary", "dataset": "root", "approx_count": 1},
        )

    return chunks, None
