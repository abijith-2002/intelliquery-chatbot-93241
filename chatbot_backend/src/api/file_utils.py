import io
import json
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime

# Libraries for file parsing
# - TXT: native decode
# - PDF: pdfminer.six
# - DOCX: python-docx
# - XLSX: pandas for structured processing
import pandas as pd
from pdfminer.high_level import extract_text as pdf_extract_text
from docx import Document as DocxDocument
from openpyxl import load_workbook

# Configure module-level logger
logger = logging.getLogger(__name__)


# PUBLIC_INTERFACE
def extract_text_from_bytes(filename: str, content: bytes) -> Tuple[str, Optional[str]]:
    """
    PUBLIC_INTERFACE
    Extract readable text from a file given its filename and raw bytes.

    Supports:
        - .txt  : UTF-8 decode with errors ignored
        - .pdf  : pdfminer.six text extraction
        - .docx : python-docx extraction (paragraphs and table cells)
        - .xlsx : pandas-based extraction with structured data handling

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


# PUBLIC_INTERFACE
def process_excel_for_session(
    filename: str,
    content: bytes,
    session_id: str
) -> Tuple[Dict[str, Any], Optional[str]]:
    """
    PUBLIC_INTERFACE
    Process Excel file using pandas, store as Parquet, and extract metadata for LLM/SQL inference.

    Args:
        filename (str): Original Excel filename
        content (bytes): Raw Excel file content
        session_id (str): Session identifier for organizing files

    Returns:
        Tuple[Dict[str, Any], Optional[str]]: (metadata, error)
            - metadata: Dictionary containing column info, data types, sample data, file paths
            - error: Error message if processing failed, otherwise None
    """
    try:
        # Create session directory
        session_dir = Path("data/sessions") / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        # Defensive checks for content
        if not content or len(content) == 0:
            return {}, f"Empty Excel file content for '{filename}'."

        # Prefer openpyxl engine for .xlsx explicitly
        engine = "openpyxl"

        # Use a single master BytesIO to enumerate sheet names
        bio_master = io.BytesIO(content)
        try:
            excel_file = pd.ExcelFile(bio_master, engine=engine)
        except Exception as e:
            logger.error(f"pd.ExcelFile failed for '{filename}' with engine={engine}: {e}")
            # Retry without specifying engine (let pandas detect) as fallback
            try:
                bio_master.seek(0)
                excel_file = pd.ExcelFile(bio_master)
            except Exception as e2:
                return {}, f"Unable to open Excel file '{filename}': {e2}"

        sheet_names = list(excel_file.sheet_names or [])
        if not sheet_names:
            # As a fallback, try openpyxl directly to confirm sheet existence
            try:
                wb = load_workbook(io.BytesIO(content), data_only=True, read_only=True)
                sheet_names = [ws.title for ws in wb.worksheets]
            except Exception as e:
                logger.error(f"openpyxl load_workbook failed for '{filename}': {e}")
            if not sheet_names:
                return {
                    "filename": filename,
                    "session_id": session_id,
                    "processed_at": datetime.utcnow().isoformat(),
                    "sheets": {},
                    "parquet_files": [],
                    "total_rows": 0,
                    "total_columns": 0
                }, f"No sheets detected in '{filename}'. Ensure the file is a valid .xlsx."

        metadata: Dict[str, Any] = {
            "filename": filename,
            "session_id": session_id,
            "processed_at": datetime.utcnow().isoformat(),
            "sheets": {},
            "parquet_files": [],
            "total_rows": 0,
            "total_columns": 0
        }

        # Process each sheet robustly
        for sheet_name in sheet_names:
            try:
                # New BytesIO for each read to avoid stream pointer issues
                df = pd.read_excel(io.BytesIO(content), sheet_name=sheet_name, engine=engine)

                # Some Excel sheets can be completely empty; skip them
                if df is None or (hasattr(df, "empty") and df.empty and len(df.columns) == 0):
                    logger.warning(f"Sheet '{sheet_name}' in '{filename}' appears empty. Skipping.")
                    continue

                # Clean column names
                df.columns = [_clean_column_name(str(col)) for col in df.columns]

                # If the sheet has no rows but has columns, keep it but mark counts appropriately
                safe_sheet_name = _clean_column_name(sheet_name)
                safe_filename = _clean_column_name(filename.replace('.xlsx', ''))
                parquet_filename = f"{safe_filename}_{safe_sheet_name}.parquet"
                parquet_path = session_dir / parquet_filename

                parquet_path_str = ""
                try:
                    # Write Parquet only if there are columns (pyarrow cannot write zero-column frames)
                    if len(df.columns) > 0:
                        df.to_parquet(parquet_path, engine='pyarrow', index=False)
                        parquet_path_str = str(parquet_path)
                        metadata["parquet_files"].append(parquet_path_str)
                    else:
                        logger.info(f"Skipping Parquet write for '{sheet_name}' due to zero columns.")
                except Exception as e_parq:
                    logger.error(f"Failed to write parquet for '{sheet_name}' in '{filename}': {e_parq}")

                # Extract column metadata
                sheet_metadata = _extract_column_metadata(df, sheet_name)
                if parquet_path_str:
                    sheet_metadata["parquet_path"] = parquet_path_str
                    sheet_metadata["parquet_filename"] = parquet_filename

                metadata["sheets"][sheet_name] = sheet_metadata
                metadata["total_rows"] += int(len(df))
                metadata["total_columns"] += int(len(df.columns))

            except Exception as e_sheet:
                logger.error(f"Error processing sheet '{sheet_name}' in '{filename}': {e_sheet}")
                continue

        # Sanity check and only write metadata to disk after processing
        base_name = _clean_column_name(filename.replace('.xlsx', ''))
        metadata_path = session_dir / f"{base_name}_metadata.json"
        try:
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            metadata["metadata_path"] = str(metadata_path)
        except Exception as e_md:
            logger.error(f"Failed to write metadata JSON for '{filename}': {e_md}")

        if metadata.get("total_rows", 0) == 0 and not metadata.get("sheets"):
            return metadata, f"No rows or sheets detected in '{filename}'. Please verify the Excel file contents."

        return metadata, None

    except Exception as e:
        logger.exception(f"Unexpected failure processing Excel '{filename}': {e}")
        return {}, f"Failed to process Excel file '{filename}': {e}"


# PUBLIC_INTERFACE
def get_session_excel_metadata(session_id: str) -> List[Dict[str, Any]]:
    """
    PUBLIC_INTERFACE
    Retrieve all Excel file metadata for a given session.

    Args:
        session_id (str): Session identifier

    Returns:
        List[Dict[str, Any]]: List of metadata dictionaries for all Excel files in session
    """
    session_dir = Path("data/sessions") / session_id
    if not session_dir.exists():
        return []

    metadata_files = list(session_dir.glob("*_metadata.json"))
    all_metadata = []

    for metadata_file in metadata_files:
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                all_metadata.append(metadata)
        except Exception as e:
            logger.warning(f"Error reading metadata file {metadata_file}: {e}")
            continue

    return all_metadata


# PUBLIC_INTERFACE
def load_session_parquet_data(session_id: str, parquet_filename: str) -> Optional[pd.DataFrame]:
    """
    PUBLIC_INTERFACE
    Load a specific Parquet file from a session.

    Args:
        session_id (str): Session identifier
        parquet_filename (str): Name of the Parquet file to load

    Returns:
        Optional[pd.DataFrame]: DataFrame if file exists and loads successfully, None otherwise
    """
    try:
        session_dir = Path("data/sessions") / session_id
        parquet_path = session_dir / parquet_filename

        if parquet_path.exists():
            return pd.read_parquet(parquet_path, engine='pyarrow')
        return None
    except Exception as e:
        logger.error(f"Error loading Parquet file '{parquet_filename}' for session '{session_id}': {e}")
        return None


def _clean_column_name(name: str) -> str:
    """Clean column name for SQL compatibility and file naming."""
    import re
    # Replace spaces and special characters with underscores
    cleaned = re.sub(r'[^\w]', '_', str(name))
    # Remove multiple consecutive underscores
    cleaned = re.sub(r'_+', '_', cleaned)
    # Remove leading/trailing underscores
    cleaned = cleaned.strip('_')
    # Ensure it starts with a letter (for SQL compatibility)
    if cleaned and not cleaned[0].isalpha():
        cleaned = 'col_' + cleaned
    return cleaned or 'unnamed_column'


def _extract_column_metadata(df: pd.DataFrame, sheet_name: str) -> Dict[str, Any]:
    """Extract comprehensive metadata from a DataFrame for LLM/SQL inference."""
    metadata = {
        "sheet_name": sheet_name,
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": {},
        "sample_data": {},
        "data_summary": {}
    }

    for col in df.columns:
        col_data = df[col]

        # Basic column info
        col_info = {
            "name": col,
            "dtype": str(col_data.dtype),
            "null_count": int(col_data.isnull().sum()),
            "null_percentage": float((col_data.isnull().sum() / len(col_data)) * 100) if len(df) > 0 else 0.0,
            "unique_count": int(col_data.nunique(dropna=True)),
            "is_numeric": pd.api.types.is_numeric_dtype(col_data),
            "is_datetime": pd.api.types.is_datetime64_any_dtype(col_data),
            "is_categorical": pd.api.types.is_categorical_dtype(col_data)
        }

        # Sample values (first 5 non-null values)
        non_null_values = col_data.dropna()
        if len(non_null_values) > 0:
            sample_values = non_null_values.head(5).tolist()
            col_info["sample_values"] = [str(v) for v in sample_values]
        else:
            col_info["sample_values"] = []

        # Statistical summary for numeric columns
        if col_info["is_numeric"] and len(non_null_values) > 0:
            try:
                col_info["min_value"] = float(pd.to_numeric(col_data, errors="coerce").min())
                col_info["max_value"] = float(pd.to_numeric(col_data, errors="coerce").max())
                col_info["mean_value"] = float(pd.to_numeric(col_data, errors="coerce").mean())
                col_info["median_value"] = float(pd.to_numeric(col_data, errors="coerce").median())
            except Exception:
                # If conversion fails, skip stats
                pass

        # Value counts for categorical-like columns (if unique count is reasonable)
        if col_info["unique_count"] <= 20 and len(non_null_values) > 0:
            try:
                value_counts = col_data.value_counts(dropna=True).head(10)
                col_info["value_counts"] = {str(k): int(v) for k, v in value_counts.items()}
            except Exception:
                pass

        metadata["columns"][col] = col_info

    # Overall data summary
    if len(df) > 0:
        try:
            metadata["sample_data"]["first_5_rows"] = df.head(5).to_dict('records')
        except Exception:
            metadata["sample_data"]["first_5_rows"] = []
        try:
            metadata["data_summary"]["memory_usage"] = int(df.memory_usage(deep=True).sum())
        except Exception:
            pass

    return metadata


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
    """Extract text from XLSX using openpyxl (sheet by sheet, TSV rows) - fallback for text extraction."""
    try:
        bio = io.BytesIO(content)
        wb = load_workbook(bio, data_only=True, read_only=True)
    except Exception as e:
        logger.error(f"openpyxl failed to read xlsx content: {e}")
        return ""
    parts: List[str] = []
    for ws in wb.worksheets:
        parts.append(f"[Sheet: {ws.title}]")
        try:
            for row in ws.iter_rows(values_only=True):
                vals = []
                for cell in row:
                    vals.append("" if cell is None else str(cell))
                if any(v.strip() for v in vals):
                    parts.append("\t".join(vals))
        except Exception as e:
            logger.warning(f"Failed iterating rows for sheet '{ws.title}': {e}")
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
