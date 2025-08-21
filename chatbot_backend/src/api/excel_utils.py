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
        # Fallback: openpyxl manual parsing
        wb = load_workbook(io.BytesIO(content), data_only=True, read_only=True)
        sheets: Dict[str, pd.DataFrame] = {}
        for ws in wb.worksheets:
            rows = list(ws.iter_rows(values_only=True))
            if not rows:
                sheets[ws.title] = pd.DataFrame()
                continue
            headers = rows[0]
            data_rows = rows[1:]
            try:
                df = pd.DataFrame(data_rows, columns=headers)
            except Exception:
                # If headers are None or duplicated in an unrecoverable manner, auto-generate
                n_cols = len(headers) if headers else (len(data_rows[0]) if data_rows else 0)
                cols = [f"col_{i+1}" for i in range(n_cols)]
                df = pd.DataFrame(rows, columns=cols)
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
) -> Dict[str, Any]:
    """
    PUBLIC_INTERFACE
    Build a compact schema dictionary to describe Excel content to Gemini.

    The schema includes sheets, column names, dtypes (generalized), count, null counts,
    null percentages, and basic statistics per column. Includes a few sample values.

    Args:
        sheets: Mapping of sheet name -> DataFrame.
        max_examples_per_col: Number of example values stored for each column.

    Returns:
        Dict[str, Any]: Structured schema description suitable for prompting Gemini.
    """
    schema: Dict[str, Any] = {
        "sheets": []
    }

    for sheet_name, df in sheets.items():
        sheet_info: Dict[str, Any] = {
            "name": sheet_name,
            "rows": int(df.shape[0]),
            "cols": int(df.shape[1]),
            "columns": []
        }
        if df.empty:
            schema["sheets"].append(sheet_info)
            continue

        for col in df.columns:
            series = df[col]
            dtype = str(series.dtype)
            non_null = series.notna().sum()
            nulls = int(df.shape[0] - non_null)
            null_pct = _percent(nulls, df.shape[0])

            col_info: Dict[str, Any] = {
                "name": str(col),
                "dtype": dtype,
                "non_null": int(non_null),
                "nulls": nulls,
                "null_pct": null_pct,
            }

            # Gather example values (non-null head)
            examples = []
            for v in series.dropna().head(max_examples_per_col).tolist():
                try:
                    # Ensure JSON serializable
                    if isinstance(v, (int, float, str, bool)) or v is None:
                        examples.append(v)
                    else:
                        examples.append(str(v))
                except Exception:
                    examples.append(str(v))
            if examples:
                col_info["examples"] = examples

            # Stats based on dtype
            if _safe_is_numeric_dtype(series.dtype):
                col_info["stats"] = _describe_numeric_series(series)
            else:
                col_info["stats"] = _describe_non_numeric_series(series)

            sheet_info["columns"].append(col_info)

        schema["sheets"].append(sheet_info)

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
    schema_preview = json.dumps(schema, ensure_ascii=False)[:12000]
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
        "Excel schema (for reference only, not for copying values):\n"
        f"{schema_preview}\n"
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
