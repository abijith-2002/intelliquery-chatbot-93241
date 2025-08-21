"""
Excel query strategy utilities.

This module encapsulates logic for:
- Estimating DataFrame size and determining whether it's small/medium/large.
- Detecting if a user query asks for aggregation/statistics.
- Building Gemini prompts that may include full schema and optionally data samples or full data for small DataFrames.
- Computing aggregates over the entire DataFrame in Python/pandas before asking Gemini for interpretation on large files.

These utilities are used by the /chat/excel-query endpoint to ensure that Gemini receives
sufficient information to answer comprehensively, while keeping performance acceptable for large files.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import json
import math
import pandas as pd


@dataclass
class SizeThresholds:
    """Thresholds for classifying DataFrame sizes."""
    max_small_rows: int = 5000
    max_small_cols: int = 100
    max_medium_rows: int = 100000
    max_medium_cols: int = 200
    # Hard limits for including raw data in prompts (avoid token blowups)
    max_cells_for_data_in_prompt: int = 200_000  # rows*cols


# PUBLIC_INTERFACE
def estimate_df_size(df: pd.DataFrame, thresholds: Optional[SizeThresholds] = None) -> str:
    """PUBLIC_INTERFACE
    Categorize a DataFrame size as 'small', 'medium', or 'large' based on thresholds.

    Args:
        df: The DataFrame to classify.
        thresholds: Optional thresholds; defaults are suitable for most cases.

    Returns:
        str: One of 'small', 'medium', 'large'.
    """
    thresholds = thresholds or SizeThresholds()
    rows, cols = int(df.shape[0]), int(df.shape[1])
    if rows <= thresholds.max_small_rows and cols <= thresholds.max_small_cols:
        return "small"
    if rows <= thresholds.max_medium_rows and cols <= thresholds.max_medium_cols:
        return "medium"
    return "large"


# PUBLIC_INTERFACE
def is_aggregate_query(user_query: str) -> bool:
    """PUBLIC_INTERFACE
    Heuristic detection of aggregation/statistics intent in a user query.

    Args:
        user_query: The user's natural language question.

    Returns:
        bool: True if the query likely requests an aggregation/summary KPI.
    """
    q = (user_query or "").lower()
    keywords = [
        "total", "sum", "average", "mean", "median", "mode",
        "count", "how many", "min", "max", "standard deviation", "std",
        "group by", "per ", "by ", "aggregate", "aggregation",
        "distribution", "percentile", "quartile",
    ]
    return any(k in q for k in keywords)


def _series_to_scalar(v: Any) -> Any:
    """Convert numpy/pandas scalars to native Python for JSON serialization."""
    try:
        import numpy as np
        if isinstance(v, np.generic):
            return v.item()
    except Exception:
        pass
    try:
        if hasattr(v, "item"):
            return v.item()
    except Exception:
        pass
    return v


# PUBLIC_INTERFACE
def compute_aggregate_answer(df: pd.DataFrame, user_query: str) -> Optional[Dict[str, Any]]:
    """PUBLIC_INTERFACE
    Attempt to compute an aggregate answer directly in pandas based on the user's query.
    This is heuristic and aims to cover common cases. Returns a dict payload capturing
    the aggregate results, or None if it cannot determine a direct aggregation.

    Supported heuristics:
    - Global aggregates: sum/mean/median/min/max/std/count of numeric columns.
    - If 'by <col>' or 'group by <col>' present, attempt groupby aggregates of numeric columns.

    Args:
        df: DataFrame to compute over (whole dataset, not sampled).
        user_query: The user question.

    Returns:
        Optional[Dict[str, Any]]: Aggregate result structure or None if not applicable.
    """
    if df is None or df.empty:
        return None

    q = (user_query or "").lower()

    # Determine columns
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        return None  # nothing to aggregate on

    # Grouping detection
    group_col: Optional[str] = None
    tokens = q.replace(",", " ").split()
    # Try to infer "by <col>" or "group by <col>"
    if "group" in tokens and "by" in tokens:
        try:
            i = tokens.index("by")
            if i + 1 < len(tokens):
                cand = tokens[i + 1]
                # fuzzy match by lowercase
                for c in df.columns:
                    if c.lower() == cand:
                        group_col = c
                        break
        except Exception:
            group_col = None
    elif " by " in q:
        after = q.split(" by ", 1)[1].split()[0].strip()
        for c in df.columns:
            if c.lower() == after:
                group_col = c
                break

    want_sum = "sum" in q or "total" in q
    want_mean = "average" in q or "mean" in q
    want_median = "median" in q
    want_min = "min" in q or "minimum" in q
    want_max = "max" in q or "maximum" in q
    want_std = "std" in q or "standard deviation" in q
    want_count = "count" in q or "how many" in q

    agg_map: Dict[str, List[str]] = {}
    if want_sum:
        agg_map["sum"] = []
    if want_mean:
        agg_map["mean"] = []
    if want_median:
        agg_map["median"] = []
    if want_min:
        agg_map["min"] = []
    if want_max:
        agg_map["max"] = []
    if want_std:
        agg_map["std"] = []
    if want_count:
        agg_map["count"] = []

    # If no explicit metric, default to 'count' as a safe aggregate for large datasets
    if not agg_map:
        agg_map["count"] = []

    # Apply groupby if detected and column seems valid
    try:
        if group_col and group_col in df.columns:
            gb = df.groupby(group_col, dropna=False)
            result_frames: Dict[str, Any] = {}
            for metric in agg_map.keys():
                if metric == "count":
                    res = gb[num_cols].count()
                else:
                    # Use direct function on groupby object if available; else fallback to apply
                    try:
                        res = getattr(gb[num_cols], metric)()
                    except Exception:
                        res = gb[num_cols].agg(metric)
                result_frames[metric] = res.reset_index().to_dict(orient="records")
            return {
                "type": "grouped_aggregates",
                "by": group_col,
                "metrics": list(result_frames.keys()),
                "results": result_frames,
            }
        else:
            # Global aggregates on numeric columns
            results: Dict[str, Any] = {}
            for metric in agg_map.keys():
                if metric == "count":
                    ser = df[num_cols].count()
                else:
                    try:
                        ser = getattr(df[num_cols], metric)()
                    except Exception:
                        ser = df[num_cols].agg(metric)
                results[metric] = {str(k): _series_to_scalar(v) for k, v in ser.to_dict().items()}
            return {
                "type": "global_aggregates",
                "metrics": list(results.keys()),
                "results": results,
            }
    except Exception:
        return None


# PUBLIC_INTERFACE
def build_prompt_with_schema_and_optional_data(
    user_query: str,
    schema: Dict[str, Any],
    df: pd.DataFrame,
    mode: str,
    thresholds: Optional[SizeThresholds] = None,
    max_rows_for_full_data: int = 5000,
) -> Tuple[str, Dict[str, Any]]:
    """PUBLIC_INTERFACE
    Build a Gemini prompt that always includes schema and optionally includes data
    (either full or sampled) depending on DataFrame size and requested mode.

    Args:
        user_query: The user's question.
        schema: Schema dict (from build_schema_for_gemini).
        df: The DataFrame of interest.
        mode: One of 'auto', 'summary', 'entire'. 'summary' favors schema+sample; 'entire' tries to include all data for small DataFrames.
        thresholds: Size thresholds to classify DataFrame size.
        max_rows_for_full_data: Hard cap for rows to include in prompt when 'entire' is chosen and df is small.

    Returns:
        Tuple[prompt, context_meta]: The built prompt string and meta info about what was included.
    """
    thresholds = thresholds or SizeThresholds()
    size_cls = estimate_df_size(df, thresholds)
    context_meta: Dict[str, Any] = {"size": size_cls, "mode": mode, "included": {"schema": True, "data_rows": 0, "data_truncated": False}}

    # Schema always included (may be truncated downstream by caller if needed)
    schema_json = json.dumps(schema, ensure_ascii=False)

    include_full = False
    include_sample = False

    if mode == "entire":
        include_full = True
    elif mode == "summary":
        include_sample = True
    else:
        # auto
        if size_cls == "small":
            include_full = True
        elif size_cls == "medium":
            include_sample = True
        else:
            include_sample = False  # for large, skip data here (compute aggregates or other strategies)

    data_section = ""
    if include_full:
        # Only include full data if under safe cell cap and row cap
        rows = min(df.shape[0], max_rows_for_full_data)
        show_df = df.head(rows)
        cells = rows * max(1, df.shape[1])
        if cells <= thresholds.max_cells_for_data_in_prompt:
            recs = show_df.to_dict(orient="records")
            data_json = json.dumps({"data": recs}, ensure_ascii=False)
            data_section = f"\nFull data (top {rows} rows):\n{data_json}\n"
            context_meta["included"]["data_rows"] = rows
            context_meta["included"]["data_truncated"] = rows < df.shape[0]
        else:
            # Fallback to sample
            include_sample = True
            include_full = False

    if include_sample and not data_section:
        # Provide a stratified/random sample if possible; otherwise head
        try:
            sample_df = df.sample(min(1000, max(50, int(math.sqrt(max(1, df.shape[0]))))), random_state=42)
        except Exception:
            sample_df = df.head(1000)
        recs = sample_df.to_dict(orient="records")
        data_json = json.dumps({"sample": recs}, ensure_ascii=False)
        data_section = f"\nSampled data preview:\n{data_json}\n"
        context_meta["included"]["data_rows"] = int(sample_df.shape[0])
        context_meta["included"]["data_truncated"] = True

    instruction = (
        "You are analyzing tabular data from an Excel sheet represented as a pandas DataFrame 'df'.\n"
        "Use the provided schema and data (if present) to answer the user's request accurately.\n"
        "If a numeric result is needed, compute it explicitly. If explanation is requested, be concise.\n"
        "Do not invent columns or values not present in the data.\n"
        "\n"
        f"User request:\n{user_query}\n\n"
        "Schema:\n"
        f"{schema_json}\n"
        f"{data_section}"
    )

    return instruction, context_meta
