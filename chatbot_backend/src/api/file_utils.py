import io
import json
from typing import List, Tuple, Optional, Any, Dict

# Libraries for file parsing
# - TXT: native decode
# - PDF: pdfminer.six
# - DOCX: python-docx
# - XLSX: pandas (reads via openpyxl engine)
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
        - .xlsx : row-wise chunked text using parse_xlsx_to_row_chunks(); returns a joined text preview

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
            # Build a compact joined text from row-chunks for preview/back-compat usage
            chunks, err = parse_xlsx_to_row_chunks(content)
            if err:
                return "", err
            joined = "\n".join(ch["text"] for ch in chunks)
            return joined.strip(), None
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


# PUBLIC_INTERFACE
def parse_xlsx_to_row_chunks(content: bytes, cols_per_chunk: int = 30) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    PUBLIC_INTERFACE
    Read an Excel (.xlsx) file using pandas and convert each non-empty row into one or more
    compact, RAG-friendly text chunks by grouping columns in chunks of up to `cols_per_chunk`.

    Chunk text format:
        "Sheet <sheet_name> | Row <row_number> | Chunk <i>/<n> | col1: val1 | col2: val2 | ..."

    Metadata per chunk is returned alongside text for downstream processing:
        {
            "text": <str>,
            "sheet_name": <str>,
            "row_number": <int>,     # 1-based
            "row_id": <str>,         # e.g., "Sheet1:r12"
            "chunk_index": <int>,    # 0-based
            "chunk_count": <int>,    # total chunks for the row
            "chunk_id": <str>,       # e.g., "Sheet1:r12:c1of3"
        }

    Empty rows (all NaN/blank) are skipped.

    Args:
        content (bytes): Raw file content of the .xlsx file.
        cols_per_chunk (int): Maximum number of column key-value pairs per chunk.

    Returns:
        Tuple[List[Dict[str, Any]], Optional[str]]:
            - List of chunk dicts (see structure above)
            - error message if parsing failed, else None
    """
    try:
        bio = io.BytesIO(content or b"")
        xls = pd.ExcelFile(bio)
        chunks_out: List[Dict[str, Any]] = []

        # Helper: stringify a single cell value robustly
        def _stringify_cell(v: Any) -> str:
            try:
                import datetime  # noqa: F401
            except Exception:
                pass
            # None
            if v is None:
                return ""
            # Pandas NA/NaN/NaT
            try:
                if pd.isna(v):  # type: ignore[arg-type]
                    return ""
            except Exception:
                # If pd.isna fails, fall back to str below
                pass
            # Common types
            if isinstance(v, str):
                sval = v.strip()
            elif hasattr(pd, "Timestamp") and isinstance(v, pd.Timestamp):  # pandas datetime
                sval = v.isoformat()
            elif isinstance(v, (list, tuple, set)):
                parts = []
                for x in v:
                    try:
                        if pd.isna(x):  # type: ignore[arg-type]
                            continue
                    except Exception:
                        pass
                    parts.append(str(x))
                sval = ", ".join(parts)
            elif isinstance(v, dict):
                sval = json.dumps(v, ensure_ascii=False)
            else:
                sval = str(v).strip()
            # Collapse internal whitespace/newlines
            return " ".join(sval.split())

        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet_name, dtype=object)

            # Normalize column names to strings
            raw_cols = [str(c).strip() if c is not None else "" for c in df.columns]

            # Deduplicate column names to avoid ambiguity (e.g., "Name", "Name#1", "Name#2")
            seen = {}
            cols: List[str] = []
            for c in raw_cols:
                base = c or ""
                if base in seen:
                    seen[base] += 1
                    cols.append(f"{base}#{seen[base]}")
                else:
                    seen[base] = 0
                    cols.append(base)

            num_cols = len(cols)

            # Iterate rows using position-based tuples for performance and clarity
            for row_pos, row_vals in enumerate(df.itertuples(index=False, name=None)):
                # Collect non-empty kv pairs for the row
                kv_pairs: List[Tuple[str, str]] = []
                for j in range(num_cols):
                    val = row_vals[j] if j < len(row_vals) else None
                    sval = _stringify_cell(val)
                    if not sval:
                        continue
                    colname = cols[j] if cols[j] else f"col{j+1}"
                    kv_pairs.append((colname, sval))

                if not kv_pairs:
                    continue

                # Determine a human-friendly row number:
                try:
                    row_label = df.index[row_pos]
                    if isinstance(row_label, (int, float)) and not pd.isna(row_label):
                        row_number = int(row_label) + 1
                    else:
                        row_number = row_pos + 1
                except Exception:
                    row_number = row_pos + 1

                # Chunk the kv pairs into groups of up to cols_per_chunk
                total = (len(kv_pairs) + cols_per_chunk - 1) // cols_per_chunk
                if total <= 0:
                    total = 1
                row_id = f"{sheet_name}:r{row_number}"

                for ci in range(total):
                    start = ci * cols_per_chunk
                    end = min(len(kv_pairs), (ci + 1) * cols_per_chunk)
                    sub_pairs = kv_pairs[start:end]
                    # Build compact text with "|" separators
                    pairs_text = " | ".join(f"{k}: {v}" for k, v in sub_pairs)
                    header = f"Sheet {sheet_name} | Row {row_number} | Chunk {ci+1}/{total}"
                    text = f"{header} | {pairs_text}" if pairs_text else header
                    chunk = {
                        "text": text.strip(),
                        "sheet_name": sheet_name,
                        "row_number": row_number,
                        "row_id": row_id,
                        "chunk_index": ci,
                        "chunk_count": total,
                        "chunk_id": f"{row_id}:c{ci+1}of{total}",
                    }
                    chunks_out.append(chunk)

        return chunks_out, None
    except Exception as e:
        return [], f"Failed to parse .xlsx content: {e}"


# PUBLIC_INTERFACE
def parse_xlsx_to_documents(content: bytes) -> Tuple[List[str], Optional[str]]:
    """
    PUBLIC_INTERFACE
    Backward-compatible wrapper that converts an .xlsx into a list of natural-language
    documents by joining the new row-chunk outputs' text.

    Each document is effectively a row-chunk string produced by parse_xlsx_to_row_chunks().

    Args:
        content (bytes): Raw file content of the .xlsx file.

    Returns:
        Tuple[List[str], Optional[str]]:
            - List of document strings (one per row-chunk)
            - error message if parsing failed, else None
    """
    chunks, err = parse_xlsx_to_row_chunks(content)
    if err:
        return [], err
    return [c["text"] for c in chunks], None


def _extract_xlsx(content: bytes) -> str:
    """
    Extract text from XLSX using pandas by converting each row into compact row-chunks
    and joining them with newlines. Prefer parse_xlsx_to_row_chunks() directly when
    you need individual chunks with metadata for embeddings.
    """
    chunks, _err = parse_xlsx_to_row_chunks(content)
    return "\n".join(ch["text"] for ch in chunks).strip()


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
def parse_and_flatten_json(content: bytes) -> Tuple[List[Tuple[str, str]], Optional[str]]:
    """
    PUBLIC_INTERFACE
    Parse the given bytes as JSON and flatten nested objects/arrays to dotted key-value pairs.

    Examples:
        {"order": {"customer": {"name": "Alice"}, "items": [{"sku": "A1"}]}}
        -> [
            ("order.customer.name", "Alice"),
            ("order.items.0.sku", "A1"),
        ]

    Arrays are indexed numerically (0-based) in the dotted path.

    Args:
        content (bytes): Raw JSON file content (UTF-8 expected; errors ignored).

    Returns:
        Tuple[List[Tuple[str, str]], Optional[str]]:
            - List of (dotted_key, value_as_string) pairs for each leaf value
            - error string if parsing failed, otherwise None
    """
    try:
        text = content.decode("utf-8", errors="ignore")
        data = json.loads(text)
    except Exception as e:
        return [], f"Invalid JSON: {e}"

    flattened: List[Tuple[str, str]] = []

    def _stringify(val: Any) -> str:
        if val is None:
            return "null"
        if isinstance(val, bool):
            return "true" if val else "false"
        if isinstance(val, (int, float)):
            return str(val)
        if isinstance(val, (dict, list)):
            # Should not happen for leaf; defensively stringify
            return json.dumps(val, ensure_ascii=False)
        return str(val)

    def _flatten(obj: Any, parent_key: str = ""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                key = f"{parent_key}.{k}" if parent_key else str(k)
                _flatten(v, key)
        elif isinstance(obj, list):
            for idx, v in enumerate(obj):
                key = f"{parent_key}.{idx}" if parent_key else str(idx)
                _flatten(v, key)
        else:
            flattened.append((parent_key, _stringify(obj)))

    _flatten(data)
    return flattened, None


# PUBLIC_INTERFACE
def format_kv_pairs_as_text(pairs: List[Tuple[str, str]]) -> str:
    """
    PUBLIC_INTERFACE
    Render flattened key-value pairs to a human-readable multi-line text.

    Format:
        key1: value1
        key2.subkey: value2
        ...

    Args:
        pairs (List[Tuple[str, str]]): Flattened pairs.

    Returns:
        str: Multi-line string suitable for previews or indexing.
    """
    lines = [f"{k}: {v}" for k, v in pairs]
    return "\n".join(lines).strip()
