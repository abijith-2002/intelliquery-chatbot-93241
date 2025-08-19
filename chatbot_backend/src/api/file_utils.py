import io
import json
import re
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


# ============ JSON PARSER FOR SEMANTIC DOCUMENTS ============

def _decode_utf8(data: bytes) -> str:
    """Decode bytes into text with UTF-8 using safe error handling and trim BOM."""
    if not data:
        return ""
    text = data.decode("utf-8", errors="ignore")
    # Remove BOM if present
    return text.lstrip("\ufeff").strip()


def _collapse_ws(value: str) -> str:
    """Collapse whitespace to single spaces while preserving newlines between fields."""
    if value is None:
        return ""
    # Replace control characters other than newline and tab
    value = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", " ", str(value))
    # Collapse internal whitespace sequences excluding existing newlines/tabs
    lines = [re.sub(r"\s+", " ", line).strip() for line in str(value).splitlines()]
    # Remove empty lines at ends
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def _sanitize_value(val: Any, max_len: int = 2000) -> str:
    """Sanitize a primitive JSON value into readable, length-limited text."""
    if val is None:
        return ""
    if isinstance(val, bool):
        s = "true" if val else "false"
    elif isinstance(val, (int, float)):
        s = str(val)
    elif isinstance(val, str):
        s = val
    else:
        # For any other primitives (rare), coerce to str
        s = str(val)
    s = _collapse_ws(s)
    if len(s) > max_len:
        s = s[: max_len - 3] + "..."
    return s


def _looks_short_primitive_list(lst: List[Any], max_items: int = 12) -> bool:
    """Heuristic: treat small lists of primitives as inline text."""
    if len(lst) > max_items:
        return False
    return all(not isinstance(x, (dict, list)) for x in lst)


def _find_record_id(obj: Dict[str, Any]) -> Optional[str]:
    """Try to extract a representative identifier from common fields."""
    if not isinstance(obj, dict):
        return None
    candidates = [
        "id", "Id", "ID", "_id", "uuid", "guid", "key",
        "name", "title", "slug", "code", "identifier",
    ]
    for k in candidates:
        if k in obj and obj[k] not in (None, "", []):
            return _sanitize_value(obj[k], max_len=256)
    # Compound fallback: compose from multiple possible fields
    parts = []
    for k in ("name", "title", "code"):
        if k in obj and obj[k]:
            sv = _sanitize_value(obj[k], max_len=128)
            if sv:
                parts.append(sv)
    if parts:
        combo = " | ".join(parts)
        return combo[:256]
    return None


def _obj_to_text(
    obj: Dict[str, Any],
    base_path: str = "$",
    max_field_value_len: int = 2000,
    max_total_len: int = 20000,
) -> Tuple[str, List[str]]:
    """
    Produce a readable text block for an object by including:
    - primitive fields
    - small lists of primitives
    Excludes nested dicts and large lists (handled as separate documents via recursion).

    Returns:
        (text, included_field_paths)
    """
    lines: List[str] = []
    included_paths: List[str] = []

    if not isinstance(obj, dict):
        s = _sanitize_value(obj, max_len=max_total_len)
        return s, [base_path]

    for key, value in obj.items():
        path = f"{base_path}.{key}"
        if isinstance(value, dict):
            # Do not inline nested dicts; will be a separate doc
            continue
        if isinstance(value, list):
            if _looks_short_primitive_list(value):
                val_text = ", ".join(_sanitize_value(v, max_field_value_len) for v in value if v is not None)
                if val_text.strip():
                    lines.append(f"{key}: {val_text}")
                    included_paths.append(path)
            else:
                # Summarize large/complex lists to avoid duplication
                # Exact children will be separate docs
                non_null = [v for v in value if v is not None]
                if non_null:
                    lines.append(f"{key}: {len(non_null)} item(s)")
                    included_paths.append(path)
            continue

        # Primitive
        s_val = _sanitize_value(value, max_len=max_field_value_len)
        if s_val.strip():
            lines.append(f"{key}: {s_val}")
            included_paths.append(path)

    # Assemble text
    text = "\n".join(lines).strip()
    if len(text) > max_total_len:
        text = text[: max_total_len - 3] + "..."
    return text, included_paths


def _collect_docs(
    node: Any,
    path: str,
    structure: str,
    filename: str,
    line_index: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Recursively collect documents from a JSON node. A 'document' corresponds to:
    - Any dict with at least one inline-eligible field (primitive or small lists)
    - Any primitive in a list (where appropriate)
    """
    docs: List[Dict[str, Any]] = []

    if isinstance(node, dict):
        text, fields = _obj_to_text(node, base_path=path)
        if text:
            record_id = _find_record_id(node)
            metadata = {
                "source_filename": filename,
                "json_structure": structure,
                "json_path": path,
                "record_id": record_id,
                "line_index": line_index,
                "field_paths": fields,
            }
            docs.append({"text": text, "metadata": metadata})

        # Recurse into child dicts and list-of-dicts for more granular docs
        for key, val in node.items():
            child_path = f"{path}.{key}"
            if isinstance(val, dict):
                docs.extend(_collect_docs(val, child_path, structure, filename, line_index))
            elif isinstance(val, list):
                for idx, item in enumerate(val):
                    item_path = f"{child_path}[{idx}]"
                    if isinstance(item, dict):
                        docs.extend(_collect_docs(item, item_path, structure, filename, line_index))
                    elif not isinstance(item, (dict, list)):
                        # Primitive list item as its own doc only if meaningful
                        s = _sanitize_value(item)
                        if s:
                            metadata = {
                                "source_filename": filename,
                                "json_structure": structure,
                                "json_path": item_path,
                                "record_id": None,
                                "line_index": line_index,
                                "field_paths": [item_path],
                            }
                            docs.append({"text": s, "metadata": metadata})

    elif isinstance(node, list):
        # Treat each element as its own record at [i]
        for i, item in enumerate(node):
            child_path = f"{path}[{i}]"
            if isinstance(item, dict):
                docs.extend(_collect_docs(item, child_path, structure, filename, line_index))
            elif isinstance(item, list):
                # Nested list; continue traversal
                docs.extend(_collect_docs(item, child_path, structure, filename, line_index))
            else:
                s = _sanitize_value(item)
                if s:
                    metadata = {
                        "source_filename": filename,
                        "json_structure": structure,
                        "json_path": child_path,
                        "record_id": None,
                        "line_index": line_index,
                        "field_paths": [child_path],
                    }
                    docs.append({"text": s, "metadata": metadata})

    else:
        # Primitive at root (rare)
        s = _sanitize_value(node)
        if s:
            metadata = {
                "source_filename": filename,
                "json_structure": structure,
                "json_path": path,
                "record_id": None,
                "line_index": line_index,
                "field_paths": [path],
            }
            docs.append({"text": s, "metadata": metadata})

    return docs


def _parse_as_json(text: str) -> Tuple[Optional[Any], Optional[str]]:
    """Try parsing as regular JSON (object or array)."""
    try:
        return json.loads(text), None
    except Exception as e:
        return None, str(e)


def _parse_as_ndjson(text: str) -> Tuple[List[Any], List[int]]:
    """
    Parse as Newline-Delimited JSON (NDJSON). Returns list of parsed records
    and corresponding line indices (0-based).
    """
    records: List[Any] = []
    line_indices: List[int] = []
    if not text:
        return records, line_indices

    lines = text.splitlines()
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            records.append(obj)
            line_indices.append(i)
        except Exception:
            # Skip non-JSON lines in NDJSON mode
            continue
    return records, line_indices


# PUBLIC_INTERFACE
def parse_json_documents(filename: str, content: bytes) -> List[Dict[str, Any]]:
    """
    PUBLIC_INTERFACE
    Parse a JSON file into semantically meaningful documents suitable for embedding and retrieval.

    This helper supports:
        1) JSON object
        2) JSON array (of objects or primitives)
        3) NDJSON (newline-delimited JSON)

    For nested structures, it recursively flattens into meaningful units:
        - Produces a document for each object with readable (primitive) fields.
        - Produces a document per array element when elements are objects or primitives.
        - Nested objects are emitted as separate documents at their respective JSONPath.

    Each returned document has:
        - text: The flattened, human-readable content.
        - metadata: Rich details including:
            - source_filename: Original filename
            - json_structure: 'object' | 'array' | 'ndjson'
            - json_path: JSONPath-like location of the record (e.g., $.items[2].name)
            - record_id: Best-effort ID/name/title from the record (if any)
            - line_index: NDJSON line index (0-based) when applicable
            - field_paths: List of field JSONPaths included in the text

    Args:
        filename (str): The original file name.
        content (bytes): Raw file bytes of the JSON payload.

    Returns:
        List[Dict[str, Any]]: List of documents, each containing 'text' and 'metadata'.
    """
    # Decode
    text = _decode_utf8(content)
    if not text:
        return []

    # Try standard JSON first
    value, err = _parse_as_json(text)
    if err is None:
        # Determine structure
        if isinstance(value, dict):
            structure = "object"
            return _collect_docs(value, "$", structure, filename, line_index=None)
        elif isinstance(value, list):
            structure = "array"
            return _collect_docs(value, "$", structure, filename, line_index=None)
        else:
            # Primitive JSON root
            structure = "object"
            return _collect_docs(value, "$", structure, filename, line_index=None)

    # Fall back: NDJSON
    records, line_indices = _parse_as_ndjson(text)
    if records:
        all_docs: List[Dict[str, Any]] = []
        for rec, idx in zip(records, line_indices):
            # For NDJSON we treat each line as independent 'object' or 'array'
            if isinstance(rec, dict):
                all_docs.extend(_collect_docs(rec, "$", "ndjson", filename, line_index=idx))
            elif isinstance(rec, list):
                all_docs.extend(_collect_docs(rec, "$", "ndjson", filename, line_index=idx))
            else:
                # Primitive line JSON
                all_docs.extend(_collect_docs(rec, "$", "ndjson", filename, line_index=idx))
        return all_docs

    # If neither JSON nor NDJSON parsable, return empty list
    return []
