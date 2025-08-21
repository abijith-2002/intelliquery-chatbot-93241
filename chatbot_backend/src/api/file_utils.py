import io
import json
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
        
        # Load Excel file with pandas
        bio = io.BytesIO(content)
        excel_file = pd.ExcelFile(bio)
        
        metadata = {
            "filename": filename,
            "session_id": session_id,
            "processed_at": datetime.utcnow().isoformat(),
            "sheets": {},
            "parquet_files": [],
            "total_rows": 0,
            "total_columns": 0
        }
        
        # Process each sheet
        for sheet_name in excel_file.sheet_names:
            try:
                # Read sheet into DataFrame
                df = pd.read_excel(bio, sheet_name=sheet_name, engine='openpyxl')
                
                # Clean column names (remove spaces, special chars for SQL compatibility)
                df.columns = [_clean_column_name(str(col)) for col in df.columns]
                
                # Generate Parquet filename
                safe_sheet_name = _clean_column_name(sheet_name)
                safe_filename = _clean_column_name(filename.replace('.xlsx', ''))
                parquet_filename = f"{safe_filename}_{safe_sheet_name}.parquet"
                parquet_path = session_dir / parquet_filename
                
                # Store as Parquet
                df.to_parquet(parquet_path, engine='pyarrow', index=False)
                
                # Extract column metadata
                sheet_metadata = _extract_column_metadata(df, sheet_name)
                sheet_metadata["parquet_path"] = str(parquet_path)
                sheet_metadata["parquet_filename"] = parquet_filename
                
                metadata["sheets"][sheet_name] = sheet_metadata
                metadata["parquet_files"].append(str(parquet_path))
                metadata["total_rows"] += len(df)
                metadata["total_columns"] += len(df.columns)
                
            except Exception as e:
                print(f"Error processing sheet '{sheet_name}': {e}")
                continue
        
        # Save metadata as JSON
        metadata_path = session_dir / f"{_clean_column_name(filename.replace('.xlsx', ''))}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        metadata["metadata_path"] = str(metadata_path)
        
        return metadata, None
        
    except Exception as e:
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
            print(f"Error reading metadata file {metadata_file}: {e}")
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
        print(f"Error loading Parquet file '{parquet_filename}' for session '{session_id}': {e}")
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
            "null_count": col_data.isnull().sum(),
            "null_percentage": (col_data.isnull().sum() / len(col_data)) * 100,
            "unique_count": col_data.nunique(),
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
                col_info["min_value"] = float(col_data.min())
                col_info["max_value"] = float(col_data.max())
                col_info["mean_value"] = float(col_data.mean())
                col_info["median_value"] = float(col_data.median())
            except:
                pass
        
        # Value counts for categorical-like columns (if unique count is reasonable)
        if col_info["unique_count"] <= 20 and len(non_null_values) > 0:
            try:
                value_counts = col_data.value_counts().head(10)
                col_info["value_counts"] = {str(k): int(v) for k, v in value_counts.items()}
            except:
                pass
        
        metadata["columns"][col] = col_info
    
    # Overall data summary
    if len(df) > 0:
        metadata["sample_data"]["first_5_rows"] = df.head(5).to_dict('records')
        metadata["data_summary"]["memory_usage"] = df.memory_usage(deep=True).sum()
    
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
