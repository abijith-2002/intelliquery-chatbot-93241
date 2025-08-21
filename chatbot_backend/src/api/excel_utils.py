"""
Utilities for processing uploaded Excel files (.xlsx), extracting schema and statistics,
prompting Gemini to produce safe pandas expressions that reference a DataFrame 'df',
and securely evaluating those expressions with a restricted context.

This module introduces:
- Excel schema extraction with per-column statistics.
- Safe evaluation sandbox for pandas expressions.
- Gemini prompting utilities tailored to return only executable pandas expressions.
"""

from __future__ import annotations

import io
import math
import json
from typing import Any, Dict, Optional

import pandas as pd
from openpyxl import load_workbook


def _safe_is_numeric_dtype(dtype) -> bool:
    """Internal helper to check numeric dtype."""
    try:
        return pd.api.types.is_numeric_dtype(dtype)
    except Exception:
        return False


def _percent(n: int, d: int) -> float:
    if d <= 0:
        return 0.0
    return round((n / d) * 100.0, 2)


# PUBLIC_INTERFACE
def parse_xlsx_to_dataframe(content: bytes) -> Dict[str, pd.DataFrame]:
    """
    PUBLIC_INTERFACE
    Parse an .xlsx file (bytes) into a mapping of sheet_name -> DataFrame.

    Args:
        content: Raw bytes of the xlsx file.

    Returns:
        Dict[str, DataFrame]: Dictionary where each key is the sheet name
        and the value is the corresponding pandas DataFrame.
    """
    # First attempt via pandas for simplicity; fallback to openpyxl manual
    try:
        return pd.read_excel(io.BytesIO(content), sheet_name=None, engine="openpyxl")
    except Exception:
        # Fallback: openpyxl manual parsing with streaming to reduce memory spikes
        wb = load_workbook(io.BytesIO(content), data_only=True, read_only=True)
        sheets: Dict[str, pd.DataFrame] = {}
        for ws in wb.worksheets:
            iter_rows = ws.iter_rows(values_only=True)
            try:
                first_row = next(iter_rows)
            except StopIteration:
                sheets[ws.title] = pd.DataFrame()
                continue
            headers = first_row
            # Collect rows incrementally; still may be large but avoids holding entire sheet at once
            data_rows = []
            max_rows_guard = 1_000_000  # absolute safety guard to avoid unbounded growth
            for idx, row in enumerate(iter_rows):
                data_rows.append(row)
                if idx >= max_rows_guard:
                    break
            try:
                df = pd.DataFrame(data_rows, columns=headers)
            except Exception:
                # If headers are None or duplicated in an unrecoverable manner, auto-generate
                n_cols = len(headers) if headers else (len(data_rows[0]) if data_rows else 0)
                cols = [f"col_{i+1}" for i in range(n_cols)]
                try:
                    df = pd.DataFrame([headers] + data_rows, columns=cols)
                except Exception:
                    df = pd.DataFrame(data_rows, columns=cols)
            sheets[ws.title] = df
        return sheets


def _describe_numeric_series(s: pd.Series) -> Dict[str, Any]:
    s_valid = s.dropna()
    count = int(s_valid.shape[0])
    if count == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "25%": None,
            "50%": None,
            "75%": None,
            "max": None,
        }
    desc = s_valid.describe()  # count, mean, std, min, 25%, 50%, 75%, max
    def _get(k: str) -> Optional[float]:
        try:
            v = desc.get(k, None)
            if isinstance(v, (int, float)) and (not isinstance(v, bool)) and math.isfinite(float(v)):
                return float(v)
            return None
        except Exception:
            return None

    return {
        "count": int(desc.get("count", count)),
        "mean": _get("mean"),
        "std": _get("std"),
        "min": _get("min"),
        "25%": _get("25%"),
        "50%": _get("50%"),
        "75%": _get("75%"),
        "max": _get("max"),
    }


def _describe_non_numeric_series(s: pd.Series) -> Dict[str, Any]:
    s_valid = s.dropna()
    count = int(s_valid.shape[0])
    nunique = int(s_valid.nunique(dropna=True)) if count > 0 else 0
    # Get top 5 most frequent values
    top_vals = []
    if count > 0:
        vc = s_valid.value_counts().head(5)
        for idx, val in vc.items():
            try:
                top_vals.append({"value": idx if idx is not None else None, "count": int(val)})
            except Exception:
                # Fallback to stringified value
                top_vals.append({"value": str(idx), "count": int(val)})
    return {
        "count": count,
        "unique": nunique,
        "top_values": top_vals,
    }


# PUBLIC_INTERFACE
def build_schema_for_gemini(
    sheets: Dict[str, pd.DataFrame],
    max_examples_per_col: int = 3,
    max_rows_per_sheet: Optional[int] = None,
) -> Dict[str, Any]:
    """
    PUBLIC_INTERFACE
    Build a compact schema dictionary to describe Excel content to Gemini.

    The schema includes sheets, column names, dtypes (generalized), count, null counts,
    null percentages, and basic statistics per column. Includes a few sample values.

    Adds a 'notes' list to the schema and per-sheet entries indicating when sampling/truncation
    was applied due to size limits so that the UI can warn users clearly.

    Args:
        sheets: Mapping of sheet name -> DataFrame.
        max_examples_per_col: Number of example values stored for each column.
        max_rows_per_sheet: Optional cap for per-sheet rows when computing schema.

    Returns:
        Dict[str, Any]: Structured schema description suitable for prompting Gemini.
    """
    schema: Dict[str, Any] = {
        "sheets": [],
        "notes": []
    }

    # If no sheets provided, return a minimal schema to avoid empty context to Gemini
    if not sheets or len(sheets) == 0:
        return {
            "sheets": [
                {"name": "Sheet1", "rows": 0, "cols": 0, "columns": []}
            ],
            "notes": ["No sheets found in the uploaded Excel; using empty schema."]
        }

    # Defensive copy to avoid mutating caller's DataFrames
    safe_sheets = {}
    for sname, sdf in sheets.items():
        safe_sheets[sname] = sdf if isinstance(sdf, pd.DataFrame) else pd.DataFrame()

    for sheet_name, df in safe_sheets.items():
        original_rows = int(getattr(df, "shape", (0, 0))[0]) if isinstance(df, pd.DataFrame) else 0
        sampled = False

        # Ensure we keep some representative rows even if df appears empty after cleaning
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame()

        if df.shape[0] == 0 and df.shape[1] > 0:
            # If there are columns but zero rows, synthesize up to 1 example row of NaNs so schema has column names.
            df = pd.DataFrame(columns=list(df.columns))

        # Optionally sample to limit heavy describe() on massive sheets
        if max_rows_per_sheet is not None and isinstance(df, pd.DataFrame) and df.shape[0] > max_rows_per_sheet:
            df = df.head(max_rows_per_sheet)
            sampled = True

        sheet_info: Dict[str, Any] = {
            "name": sheet_name,
            "rows": int(df.shape[0]) if isinstance(df, pd.DataFrame) else 0,
            "cols": int(df.shape[1]) if isinstance(df, pd.DataFrame) else 0,
            "columns": []
        }
        if sampled:
            sheet_info["truncated"] = True
            sheet_info["truncated_at_rows"] = int(max_rows_per_sheet or 0)
            sheet_info["original_rows_estimate"] = original_rows
            schema["notes"].append(
                f"Sheet '{sheet_name}' truncated to {max_rows_per_sheet} rows for schema/stats to avoid long processing."
            )

        # Always include columns if present, even when df is empty
        if not isinstance(df, pd.DataFrame) or (df.empty and df.shape[1] == 0):
            schema["sheets"].append(sheet_info)
            continue

        # Use dtypes from DataFrame even if empty to avoid losing column info
        for col in list(df.columns):
            try:
                series = df[col]
            except Exception:
                # If column access fails, skip but add minimal column info
                sheet_info["columns"].append({"name": str(col), "dtype": "unknown", "non_null": 0, "nulls": 0, "null_pct": 0.0})
                continue

            dtype = str(series.dtype) if hasattr(series, "dtype") else "unknown"
            non_null = int(series.notna().sum()) if hasattr(series, "notna") else 0
            total_rows = int(df.shape[0]) if hasattr(df, "shape") else 0
            nulls = int(max(total_rows - non_null, 0))
            null_pct = _percent(nulls, total_rows)

            col_info: Dict[str, Any] = {
                "name": str(col),
                "dtype": dtype,
                "non_null": int(non_null),
                "nulls": nulls,
                "null_pct": null_pct,
            }

            # Gather example values (non-null head); if empty, try raw head without dropna to preserve some representative values
            examples = []
            try:
                sample_series = series.dropna().head(max_examples_per_col)
                if sample_series.empty:
                    sample_series = series.head(max_examples_per_col)
                for v in sample_series.tolist():
                    try:
                        if isinstance(v, (int, float, str, bool)) or v is None:
                            examples.append(v)
                        else:
                            examples.append(str(v))
                    except Exception:
                        examples.append(str(v))
            except Exception:
                pass
            if examples:
                col_info["examples"] = examples

            # Stats based on dtype
            try:
                if _safe_is_numeric_dtype(series.dtype):
                    col_info["stats"] = _describe_numeric_series(series)
                else:
                    col_info["stats"] = _describe_non_numeric_series(series)
            except Exception:
                # If stats fail (e.g., on empty), provide minimal stats
                col_info["stats"] = {"count": int(non_null)}

            sheet_info["columns"].append(col_info)

        schema["sheets"].append(sheet_info)

    # As a final guard, ensure we have at least one sheet with columns to avoid empty prompt context
    has_columns = any((len(s.get("columns") or []) > 0) for s in schema["sheets"])
    if not has_columns:
        schema["notes"].append("No columns detected across sheets; schema is minimal and may limit Gemini capabilities.")
    return schema


# PUBLIC_INTERFACE
def get_gemini_pandas_prompt(user_query: str, schema: Dict[str, Any]) -> str:
    """
    PUBLIC_INTERFACE
    Create a strict instruction prompt for Gemini to output ONLY a single-line
    valid Python pandas expression referencing an existing DataFrame variable named 'df'.

    Rules:
    - Output must be a single python expression suitable for eval().
    - Must only reference 'df' (and pandas functions via 'pd' if needed).
    - Do not include imports, function defs, print, comments, or text around the code.
    - Prefer pure expression like: df['col'].mean() or df.groupby('A')['B'].sum().to_dict()
    - For tabular results, convert to a JSON-serializable structure (e.g., .to_dict(orient='records')).
    - DO NOT mutate df.

    Args:
        user_query: Natural language question/task from the user.
        schema: Compact schema metadata of the Excel file(s).

    Returns:
        str: A single text prompt to send to Gemini.
    """
    # Make sure schema is serializable and trimmed to avoid token overflow
    full_schema_json = json.dumps(schema, ensure_ascii=False)
    # Provide a compact summary header that is always present and useful even if deep truncation occurs
    try:
        sheet_count = len(schema.get("sheets", []))
        sheet_summ = []
        for s in schema.get("sheets", [])[:5]:
            nm = s.get("name", "Sheet")
            cols = len(s.get("columns", []) or [])
            rows = s.get("rows", 0)
            sheet_summ.append(f"{nm}(rows={rows}, cols={cols})")
        compact_summary = f"Sheets: {sheet_count}; summary: " + ", ".join(sheet_summ)
    except Exception:
        compact_summary = "Sheets: unknown; summary unavailable"

    MAX_SCHEMA_CHARS = 12000
    schema_truncated = False
    if len(full_schema_json) > MAX_SCHEMA_CHARS:
        # Attempt to truncate at the last complete object boundary to avoid malformed JSON preview
        cut = MAX_SCHEMA_CHARS
        # back up to a comma or brace for safer cut
        while cut > 0 and full_schema_json[cut - 1] not in [",", "}", "]"]:
            cut -= 1
        if cut < 4000:
            # ensure we still include a reasonable chunk even if boundary search failed
            cut = MAX_SCHEMA_CHARS
        schema_preview = full_schema_json[:cut]
        schema_truncated = True
    else:
        schema_preview = full_schema_json

    instruction = (
        "You are given a DataFrame 'df' that represents the user's Excel sheet of interest.\n"
        "Your task: Return ONLY a single valid Python pandas expression that computes the answer to the user's request.\n"
        "Constraints:\n"
        "- The output must be only the code expression, no backticks, no commentary.\n"
        "- Reference the DataFrame strictly as 'df'. You may also use 'pd' for pandas functions if necessary.\n"
        "- Do not import modules, do not assign to variables, do not print, do not define functions.\n"
        "- If the result is a DataFrame or Series, convert it to a JSON-serializable structure, e.g., to_dict(orient='records') or .to_list().\n"
        "- Never mutate df.\n"
        "- Keep it as a single line expression.\n"
        "If aggregation or grouping is requested, perform using pandas idioms.\n"
        "\n"
        "User request:\n"
        f"{user_query}\n"
        "\n"
        "Excel schema summary (reference only):\n"
        f"{compact_summary}\n"
        "Detailed schema (may be truncated for length):\n"
        f"{schema_preview}\n"
        f"{'(schema preview truncated for length)\\n' if schema_truncated else ''}"
        "\n"
        "Return ONLY the expression, nothing else."
    )
    return instruction


# PUBLIC_INTERFACE
def safe_eval_pandas_expression(df: pd.DataFrame, expr: str) -> Any:
    """
    PUBLIC_INTERFACE
    Safely evaluate a pandas expression that references only 'df' (and optionally 'pd').

    A restricted eval environment is used to prevent arbitrary code execution.
    Only 'df' and a safe subset of pandas (pd) is exposed.

    Args:
        df: The DataFrame to operate on.
        expr: The expression string generated by Gemini.

    Returns:
        Any: The computed result (should be JSON-serializable by caller).

    Raises:
        ValueError: If the expression is empty or unsafe.
        Exception: If evaluation fails.
    """
    if not expr or not isinstance(expr, str):
        raise ValueError("Empty or invalid expression for evaluation.")
    # Very basic guardrails: forbid suspicious tokens
    forbidden_tokens = [
        "__", "import", "exec", "eval", "open", "os.", "sys.", "subprocess", "lambda", "class", "def",
        "while", "for ", "try:", "except", "raise", "with ", "del", "globals", "locals", "compile",
        "input(", "print(", "setattr", "getattr", "attr", "pd.read_", "to_csv(", "to_excel(", "to_sql(",
    ]
    lowered = expr.replace("\n", " ").strip().lower()
    for tok in forbidden_tokens:
        if tok in lowered:
            raise ValueError("Unsafe expression: contains forbidden token.")

    # Build a minimal safe namespace
    # Allow access to pd and a subset of functions through pandas
    safe_builtins = {}
    import pandas as _pd  # local reference
    safe_globals = {
        "__builtins__": safe_builtins,
        "pd": _pd,
    }
    safe_locals = {"df": df}

    # Evaluate expression
    result = eval(expr, safe_globals, safe_locals)  # noqa: S307 (intentional but restricted)
    return result


# PUBLIC_INTERFACE
def normalize_result_for_json(result: Any, max_rows: int = 10000) -> Any:
    """
    PUBLIC_INTERFACE
    Normalize common pandas return types into JSON-serializable structures.

    - DataFrame -> list[dict]
    - Series -> list or dict depending on index; prefer list of dict if meaningful
    - numpy types -> cast to Python scalars
    - Other -> return as-is if already serializable

    Args:
        result: The evaluated result from the pandas expression.
        max_rows: Maximum rows to include from DataFrame-like results to avoid huge payloads.

    Returns:
        A JSON-serializable object.
    """
    import numpy as np

    try:
        if isinstance(result, pd.DataFrame):
            if result.shape[0] > max_rows:
                result = result.head(max_rows)
            return result.to_dict(orient="records")
        if isinstance(result, pd.Series):
            # Prefer simple list if index is RangeIndex or simple
            if result.shape[0] > max_rows:
                result = result.head(max_rows)
            try:
                if result.index.is_numeric() or result.index.is_monotonic_increasing:
                    return result.tolist()
            except Exception:
                pass
            # Fallback to mapping
            return {str(k): (v.item() if isinstance(v, (np.generic,)) else v) for k, v in result.to_dict().items()}
        # Numpy scalars/arrays
        if isinstance(result, np.generic):
            return result.item()
        if isinstance(result, (list, tuple)):
            out = []
            for v in result:
                if isinstance(v, np.generic):
                    out.append(v.item())
                else:
                    out.append(v)
            return out
        if isinstance(result, dict):
            out_d = {}
            for k, v in result.items():
                if isinstance(v, np.generic):
                    out_d[k] = v.item()
                else:
                    out_d[k] = v
            return out_d
        return result
    except Exception:
        # As a last resort, stringify
        return str(result)
