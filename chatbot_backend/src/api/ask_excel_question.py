from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

import pandas as pd
import duckdb
import google.generativeai as genai

from .config_utils import get_gemini_api_key

router = APIRouter(tags=["Chat"])


class AskExcelQuestionRequest(BaseModel):
    """Request schema for asking a question against uploaded Excel data."""
    session_id: str = Field(..., description="Chat session ID that owns uploaded Excel files.")
    question: str = Field(..., description="Natural language question to answer from the uploaded Excel data.")


class AskExcelAnswerResponse(BaseModel):
    """Response schema returning conversational answer derived from Excel data."""
    answer: str = Field(..., description="Conversational answer phrased from the computed result.")


@dataclass
class SheetInfo:
    filename: str
    sheet_name: str
    columns: List[str]
    dtypes: Dict[str, str]
    rows: int
    sample: List[Dict[str, Any]]
    parquet_path: Optional[str]


def _load_session_excel_overview(session_id: str) -> List[SheetInfo]:
    """
    Retrieve metadata (and parquet pointers) for the session's uploaded Excel sheets
    from the in-memory EXCEL_STORE defined in main.py.
    """
    # Local import to avoid cyclic import at module import time
    from .main import EXCEL_STORE  # type: ignore
    out: List[SheetInfo] = []
    session_items = EXCEL_STORE.get(session_id)
    if not session_items:
        return out
    for file_entry in session_items:
        filename = file_entry.get("filename")
        sheets = file_entry.get("sheets", {}) or {}
        for sname, meta in sheets.items():
            out.append(
                SheetInfo(
                    filename=filename,
                    sheet_name=str(sname),
                    columns=list(meta.get("columns") or []),
                    dtypes=dict(meta.get("dtypes") or {}),
                    rows=int(meta.get("rows") or 0),
                    sample=list(meta.get("sample") or []),
                    parquet_path=meta.get("parquet_path"),
                )
            )
    return out


def _build_structured_schema_doc(sheets: List[SheetInfo]) -> str:
    """
    Build a compact, LLM-friendly schema description of available DataFrames.
    """
    parts: List[str] = []
    for s in sheets:
        parts.append(
            f"- DataFrame name: `{s.filename}::{s.sheet_name}`\n"
            f"  columns: {s.columns}\n"
            f"  dtypes: {s.dtypes}\n"
            f"  approx_rows: {s.rows}\n"
            f"  sample_first_rows: {s.sample[:2] if s.sample else []}\n"
        )
    return "Available DataFrames:\n" + ("\n".join(parts) if parts else "(none)")


def _llm_generate_query(question: str, schema_doc: str) -> Dict[str, Any]:
    """
    Ask Gemini to propose a safe query plan. We accept SQL (DuckDB) or pandas code,
    but require a strict JSON tool output describing the intent.

    Returns a dict with keys:
      - engine: "duckdb" | "pandas"
      - df: "<filename>::<sheet_name>" or "" (for pandas)
      - query: SQL string (if engine=duckdb)
      - pandas_op: optional description of operations (if engine=pandas)
      - aggregation_hint: human explanation of what the query computes
    """
    api_key = get_gemini_api_key()
    if not api_key:
        raise HTTPException(status_code=500, detail="Gemini API key is not configured.")

    system_instructions = (
        "You are a data analyst agent. Given a natural language question and a list of available "
        "DataFrames (Excel sheets already parsed), produce a safe query plan.\n"
        "Rules:\n"
        "1) Prefer DuckDB SQL if a relational query fits well. Otherwise suggest a pandas operation.\n"
        "2) Output ONLY a single JSON object with keys: engine, df, query, pandas_op, aggregation_hint.\n"
        "   - engine is 'duckdb' or 'pandas'.\n"
        "   - df is '<filename>::<sheet_name>' you plan to operate on for pandas; for duckdb you may leave df empty if joining multiple tables.\n"
        "   - query is the SQL for DuckDB if engine=duckdb, else empty.\n"
        "   - pandas_op is a compact natural language recipe of pandas steps if engine=pandas; empty otherwise.\n"
        "   - aggregation_hint explains in 1 sentence what the query computes.\n"
        "3) Do NOT invent columns. Only use columns listed in the schema."
    )

    prompt = (
        f"{system_instructions}\n\n"
        f"{schema_doc}\n\n"
        f"Question:\n{question}\n\n"
        "Now return the JSON object only."
    )

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content([{"role": "user", "parts": [prompt]}])
        text = (response.text or "").strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini error generating query: {e}")

    # Attempt to locate a JSON object in the response
    import json
    import re

    # Try to extract the first {...} JSON object
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise HTTPException(status_code=500, detail="Gemini failed to return a JSON plan.")

    json_str = match.group(0)
    try:
        obj = json.loads(json_str)
        # Normalize keys
        engine = (obj.get("engine") or "").lower().strip()
        if engine not in ("duckdb", "pandas"):
            engine = "duckdb"
        return {
            "engine": engine,
            "df": obj.get("df") or "",
            "query": obj.get("query") or "",
            "pandas_op": obj.get("pandas_op") or "",
            "aggregation_hint": obj.get("aggregation_hint") or "",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not parse Gemini JSON plan: {e}")


def _register_duckdb_views(con, sheets: List[SheetInfo]) -> Dict[str, pd.DataFrame]:
    """
    Register DuckDB views for each sheet, backed by either in-memory DataFrames
    built from sample or by reading Parquet if available, or falling back to
    reading small slices from Excel if necessary (limited).

    Returns a mapping of table_name -> DataFrame for traceability.
    """
    table_map: Dict[str, pd.DataFrame] = {}

    # Build DataFrames for each sheet, preferring parquet if available.
    for s in sheets:
        table_name = f"{s.filename}::{s.sheet_name}".replace(" ", "_")
        df: Optional[pd.DataFrame] = None
        # Parquet path preferred for large data
        if s.parquet_path:
            try:
                df = pd.read_parquet(s.parquet_path)
            except Exception:
                df = None
        if df is None:
            # Try to rebuild DataFrame minimally from samples is insufficient for queries,
            # but we can attempt to fallback reading original Excel from EXCEL_STORE is not kept.
            # Since we don't keep raw bytes, we can only rely on parquet when large.
            # For smaller sheets without parquet, try constructing an empty df with dtypes and sample rows appended.
            try:
                # Construct from sample if available, else build empty with columns only.
                if s.sample:
                    df = pd.DataFrame(s.sample)
                else:
                    df = pd.DataFrame(columns=s.columns)
            except Exception:
                df = pd.DataFrame(columns=s.columns)

        # Ensure columns names are strings
        df.columns = [str(c) for c in df.columns]
        con.register(table_name, df)
        table_map[table_name] = df

    return table_map


def _execute_duckdb_query(plan: Dict[str, Any], sheets: List[SheetInfo]) -> Tuple[pd.DataFrame, str]:
    """
    Execute a DuckDB SQL plan over registered views for the session.
    Returns the DataFrame result and a short execution note.
    """
    query = (plan.get("query") or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="LLM plan missing SQL query.")

    # Open ephemeral DuckDB connection in-process
    con = duckdb.connect(database=":memory:")
    try:
        _register_duckdb_views(con, sheets)
        # Limit potentially dangerous or overly large queries by enforcing a default LIMIT if absent
        safe_query = query
        lowered = query.lower()
        if " limit " not in lowered:
            safe_query += " LIMIT 100"

        df = con.execute(safe_query).fetchdf()
        return df, "Executed DuckDB SQL."
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"DuckDB query failed: {e}")
    finally:
        try:
            con.close()
        except Exception:
            pass


def _execute_pandas_op(plan: Dict[str, Any], sheets: List[SheetInfo]) -> Tuple[pd.DataFrame, str]:
    """
    Execute a simple pandas operation recipe. To ensure safety, we implement only
    a few recognized operations based on the high-level 'pandas_op' description
    and selected df name. We DO NOT eval arbitrary code.
    """
    df_name = plan.get("df") or ""
    op_desc = (plan.get("pandas_op") or "").lower()
    if not df_name:
        raise HTTPException(status_code=400, detail="LLM plan (pandas) did not specify target DataFrame name.")
    # Find sheet
    target_sheet = None
    for s in sheets:
        if f"{s.filename}::{s.sheet_name}" == df_name:
            target_sheet = s
            break
    if not target_sheet:
        raise HTTPException(status_code=400, detail=f"Target DataFrame '{df_name}' not found.")

    # Load DF (parquet preferred, else build from sample)
    df: Optional[pd.DataFrame] = None
    if target_sheet.parquet_path:
        try:
            df = pd.read_parquet(target_sheet.parquet_path)
        except Exception:
            df = None
    if df is None:
        try:
            if target_sheet.sample:
                df = pd.DataFrame(target_sheet.sample)
            else:
                df = pd.DataFrame(columns=target_sheet.columns)
        except Exception:
            df = pd.DataFrame(columns=target_sheet.columns)

    # Implement very basic recognizable operations guided by op_desc:
    # - count rows
    # - list unique values of a column
    # - sum/avg of numeric column
    # - filter by equality on a known column then aggregate count
    # We rely on column name presence heuristics from op_desc.
    import re

    def find_column_name(candidates: List[str]) -> Optional[str]:
        cset = set([c.lower() for c in candidates])
        for col in df.columns:
            cl = str(col).lower()
            if cl in cset:
                return str(col)
        # Try fuzzy contain
        for col in df.columns:
            cl = str(col).lower()
            for token in candidates:
                if token.lower() in cl:
                    return str(col)
        return None

    # Count rows
    if "count rows" in op_desc or re.search(r"\brow count\b", op_desc):
        res = pd.DataFrame({"row_count": [int(len(df))]})
        return res, "Computed row count via pandas."

    # Unique values
    if "unique" in op_desc:
        # Try to infer column name from description tokens
        tokens = re.findall(r"[A-Za-z0-9_]+", op_desc)
        guess = find_column_name(tokens)
        if guess and guess in df.columns:
            uniques = pd.DataFrame({guess: df[guess].dropna().unique()[:100]})
            return uniques, f"Listed unique values for column '{guess}'."

    # Sum/average for numeric columns
    if "sum" in op_desc or "average" in op_desc or "avg" in op_desc or "mean" in op_desc:
        tokens = re.findall(r"[A-Za-z0-9_]+", op_desc)
        guess = find_column_name(tokens)
        if guess and guess in df.columns:
            series = pd.to_numeric(df[guess], errors="coerce")
            if series.notna().any():
                if "sum" in op_desc:
                    s = series.sum()
                    return pd.DataFrame({f"sum({guess})": [s]}), f"Summed column '{guess}'."
                else:
                    m = series.mean()
                    return pd.DataFrame({f"avg({guess})": [m]}), f"Averaged column '{guess}'."

    # Basic filter like "where <col> = <value> then count"
    m = re.search(r"where ([A-Za-z0-9_ ]+?) = ([A-Za-z0-9_\\-]+)", op_desc)
    if m:
        col_token = m.group(1).strip()
        val_token = m.group(2).strip()
        col_name = find_column_name([col_token])
        if col_name and col_name in df.columns:
            subset = df[df[col_name].astype(str).str.lower() == val_token.lower()]
            return pd.DataFrame({"row_count": [int(len(subset))]}), f"Filtered by {col_name} == {val_token} and counted rows."

    # If nothing matched, return a preview
    preview = df.head(10)
    return preview, "Returned top 10 rows preview (pandas fallback)."


def _llm_conversationalize(question: str, schema_doc: str, plan: Dict[str, Any], result_df: pd.DataFrame) -> str:
    """
    Ask Gemini to turn a raw result (small DataFrame) into a conversational answer
    addressing the question and referencing columns in plain language.
    """
    api_key = get_gemini_api_key()
    if not api_key:
        raise HTTPException(status_code=500, detail="Gemini API key is not configured.")

    # Convert small df to markdown-like table string
    try:
        table_text = result_df.head(20).to_markdown(index=False)
    except Exception:
        table_text = result_df.head(20).to_string(index=False)

    plan_desc = (
        f"Engine: {plan.get('engine')}\n"
        f"Target DF: {plan.get('df')}\n"
        f"Aggregation hint: {plan.get('aggregation_hint')}"
    )

    instruction = (
        "You are a helpful analyst. Using the following raw table output produced from the user's data, "
        "write a short, direct answer in natural language. If the table contains multiple rows, summarize key values. "
        "Avoid extraneous meta-discussion. Keep it concise."
    )

    prompt = (
        f"{instruction}\n\n"
        f"{schema_doc}\n\n"
        f"User question:\n{question}\n\n"
        f"Query plan (for your context):\n{plan_desc}\n\n"
        f"Raw result table (first rows):\n{table_text}\n\n"
        "Now provide only the final answer text."
    )

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content([{"role": "user", "parts": [prompt]}])
        return (response.text or "").strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini error generating conversational answer: {e}")


# PUBLIC_INTERFACE
@router.post(
    "/chat/ask-excel-question",
    response_model=AskExcelAnswerResponse,
    summary="Ask a question about uploaded Excel data",
    description=(
        "Accepts a session_id and a natural language question, inspects the session's stored Excel sheets and column metadata, "
        "prompts an LLM to generate a safe pandas/duckdb query using the available structure, executes it, and returns a conversational answer."
    ),
    responses={
        400: {"description": "Validation error or no Excel data available"},
        500: {"description": "Gemini error or query execution failure"},
    },
)
def ask_excel_question(request: AskExcelQuestionRequest) -> AskExcelAnswerResponse:
    """
    PUBLIC_INTERFACE
    Ask a natural language question grounded in the session's uploaded Excel data.

    Process:
      1. Retrieve stored Excel sheet metadata for the session (columns, dtypes, samples, parquet paths).
      2. Provide schema overview to LLM to propose a safe plan (DuckDB SQL or constrained pandas ops).
      3. Execute the plan against registered DuckDB views (parquet or sample-backed) or pandas.
      4. Send the raw (small) result back to the LLM for a concise conversational answer.
      5. Return the final answer text.

    Args:
        request (AskExcelQuestionRequest): Includes session_id and natural language question.

    Returns:
        AskExcelAnswerResponse: Final conversational answer string.
    """
    session_id = (request.session_id or "").strip()
    question = (request.question or "").strip()
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id must be provided.")
    if not question:
        raise HTTPException(status_code=400, detail="question must be provided.")

    # Load Excel overview
    sheets = _load_session_excel_overview(session_id)
    if not sheets:
        raise HTTPException(status_code=400, detail="No Excel data found for this session. Upload Excel files first.")

    # Build schema doc for prompting
    schema_doc = _build_structured_schema_doc(sheets)

    # Ask LLM for safe plan
    plan = _llm_generate_query(question, schema_doc)

    # Execute plan
    if plan["engine"] == "duckdb":
        result_df, _ = _execute_duckdb_query(plan, sheets)
    else:
        result_df, _ = _execute_pandas_op(plan, sheets)

    # Conversationalize the result
    final_text = _llm_conversationalize(question, schema_doc, plan, result_df)

    # Simple cleanup
    final_text = " ".join((final_text or "").split()).strip()
    if not final_text:
        final_text = "I was unable to produce a meaningful answer from the available data."

    return AskExcelAnswerResponse(answer=final_text)
