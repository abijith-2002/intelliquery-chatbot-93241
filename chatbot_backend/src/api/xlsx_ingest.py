import io
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Generator, List, Optional

from openpyxl import load_workbook
from openpyxl.worksheet.worksheet import Worksheet


# ---------- Header normalization ----------

# PUBLIC_INTERFACE
def normalize_header(header: str) -> str:
    """
    PUBLIC_INTERFACE
    Normalize header names for consistent downstream use.

    Rules:
    - Trim whitespace.
    - Convert to lowercase.
    - Replace consecutive whitespace with single underscore.
    - Remove non-alphanumeric/underscore characters.
    - Collapse multiple underscores and trim leading/trailing underscores.
    """
    if header is None:
        return ""
    s = str(header).strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_]", "", s)
    s = re.sub(r"_+", "_", s)
    s = s.strip("_")
    return s


# ---------- Type inference and stats ----------

@dataclass
class ColumnStats:
    name: str
    normalized_name: str
    inferred_type: str = "string"
    count: int = 0
    non_null: int = 0
    unique_count: int = 0
    min_value: Optional[float] = None  # for numeric/date-like represented as float
    max_value: Optional[float] = None
    sample_values: List[str] = field(default_factory=list)

    # track uniques with a cap to bound memory
    _unique_set: set = field(default_factory=set, repr=False)
    _unique_cap: int = field(default=5000, repr=False)

    def update(self, raw_value: Any) -> None:
        """Update counters and inferred type from a raw cell value."""
        self.count += 1
        if raw_value is None or (isinstance(raw_value, str) and raw_value.strip() == ""):
            return
        self.non_null += 1

        # Update unique tracking with cap
        if len(self._unique_set) < self._unique_cap:
            self._unique_set.add(str(raw_value))

        # Collect up to a few samples
        if len(self.sample_values) < 5:
            self.sample_values.append(str(raw_value)[:256])

        # Infer type progressively (numeric > boolean > string)
        v_type = infer_cell_type(raw_value)
        self.inferred_type = merge_types(self.inferred_type, v_type)

        # Min/max for numerics
        if v_type == "number":
            try:
                num = float(raw_value)
                if self.min_value is None or num < self.min_value:
                    self.min_value = num
                if self.max_value is None or num > self.max_value:
                    self.max_value = num
            except Exception:
                pass

    def finalize(self) -> None:
        """Finalize stats (compute unique_count and sanity)."""
        self.unique_count = len(self._unique_set)
        # Clean internal fields
        self._unique_set.clear()

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # remove internal caps not meant for external output
        if "_unique_set" in d:
            d.pop("_unique_set", None)
        if "_unique_cap" in d:
            d.pop("_unique_cap", None)
        return d


def infer_cell_type(val: Any) -> str:
    """Best-effort type inference for a single cell."""
    if val is None:
        return "null"
    if isinstance(val, bool):
        return "boolean"
    if isinstance(val, (int, float)):
        return "number"
    s = str(val).strip()
    if s == "":
        return "null"
    # Try numeric
    try:
        float(s.replace(",", ""))  # allow thousands commas
        return "number"
    except Exception:
        pass
    # Try boolean strings
    if s.lower() in {"true", "false", "yes", "no"}:
        return "boolean"
    return "string"


def merge_types(current: str, new: str) -> str:
    """
    Merge two inferred types conservatively:
    - If any 'string' encountered, result becomes 'string'.
    - Prefer 'number' over 'boolean' if mixed.
    - 'null' is neutral.
    """
    if current == "string" or new == "string":
        return "string"
    if current == "number" or new == "number":
        return "number"
    if current == "boolean" or new == "boolean":
        return "boolean"
    return current if current != "null" else new


# ---------- XLSX streaming reader ----------

def _iter_sheet_rows(ws: Worksheet) -> Generator[List[Any], None, None]:
    """
    Stream rows from a worksheet using openpyxl read_only iter_rows(values_only=True).
    """
    for row in ws.iter_rows(values_only=True):
        yield list(row)


# PUBLIC_INTERFACE
def stream_xlsx_catalog(
    content: bytes,
    header_row: int = 1,
    sample_rows_for_types: int = 1000,
) -> Dict[str, Any]:
    """
    PUBLIC_INTERFACE
    Stream a large .xlsx file to build a schema/catalog and lightweight table metadata per sheet.

    - Header normalization for consistent field names.
    - Type inference based on up to 'sample_rows_for_types' data rows (per sheet) for efficiency.
    - Column-level stats: counts, non-null, unique cardinality (capped), min/max for numbers, samples.

    Args:
        content (bytes): The Excel file content (entire file as bytes).
        header_row (int): 1-based index indicating which row contains headers.
        sample_rows_for_types (int): Max number of data rows to scan for type inference/stats.

    Returns:
        Dict[str, Any]: catalog metadata:
            {
              "sheets": [
                {
                  "sheet_name": "...",
                  "headers_original": [...],
                  "headers_normalized": [...],
                  "columns": [ { ColumnStats... }, ... ],
                  "row_count_scanned": int
                },
                ...
              ],
              "total_sheets": int
            }
    """
    bio = io.BytesIO(content)
    wb = load_workbook(bio, data_only=True, read_only=True)
    sheets_meta: List[Dict[str, Any]] = []

    for ws in wb.worksheets:
        rows_iter = _iter_sheet_rows(ws)
        # Skip rows until header_row
        header_values: Optional[List[Any]] = None
        for i, row in enumerate(rows_iter, start=1):
            if i == header_row:
                header_values = [str(c) if c is not None else "" for c in row]
                break
        if header_values is None:
            # Empty sheet or header missing
            sheets_meta.append(
                {
                    "sheet_name": ws.title,
                    "headers_original": [],
                    "headers_normalized": [],
                    "columns": [],
                    "row_count_scanned": 0,
                }
            )
            continue

        normalized_headers = [normalize_header(h) for h in header_values]
        # Initialize column stats
        col_stats: List[ColumnStats] = [
            ColumnStats(name=orig, normalized_name=norm) for orig, norm in zip(header_values, normalized_headers)
        ]

        # Scan up to sample_rows_for_types rows
        scanned = 0
        for row in rows_iter:
            scanned += 1
            # Map row cells to columns; pad/truncate to header length
            # Guard against ragged rows
            values = list(row) + [None] * max(0, len(col_stats) - len(row))
            values = values[: len(col_stats)]

            for stat, val in zip(col_stats, values):
                stat.update(val)

            if scanned >= sample_rows_for_types:
                break

        # Finalize stats
        for stat in col_stats:
            stat.finalize()

        sheets_meta.append(
            {
                "sheet_name": ws.title,
                "headers_original": header_values,
                "headers_normalized": normalized_headers,
                "columns": [stat.to_dict() for stat in col_stats],
                "row_count_scanned": scanned,
            }
        )

    return {
        "sheets": sheets_meta,
        "total_sheets": len(sheets_meta),
    }
