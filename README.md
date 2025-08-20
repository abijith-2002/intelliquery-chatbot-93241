# Project Repository

This is the initial README file for the project.

## DuckDB Integration (Backend) and Excel Processing

- The FastAPI backend integrates DuckDB as an in-memory SQL engine for uploaded Excel/tabular data.
- For each uploaded .xlsx file, the backend:
  - Parses sheets into pandas DataFrames (per-session; in-memory only).
  - Registers each sheet as a DuckDB table for the user's session using the convention:
    - {safe_base_filename}__{safe_sheet_name}
  - Executes inline SQL (queries starting with SELECT/WITH or fenced as ```sql ... ```).
  - Returns compact, preview-sized results for chat responses.

Key notes:
- In-memory only; no persistent DuckDB database is created or stored.
- Useful for repeated or large-data queries; only necessary rows/columns are returned.
- The chat endpoint auto-detects inline SQL and executes it via DuckDB.
- Analytics helpers compute simple stats when SQL is not provided.

### Excel Upload Enhancements (Schema-First + Hybrid Chunking)
- Large Excel files are processed schema-first:
  - We avoid indexing entire sheet contents; instead we:
    - Extract a structured schema (sheet names, column headers, inferred types, sample values).
    - Build hybrid chunks:
      - Column-group chunks (grouped by header theme or fixed-size windows).
      - Row-slice chunks for very wide rows (in small/medium files).
  - Only these schema/chunks are embedded and stored in the in-memory RAG index (never full sheets).
- Small/medium Excel files:
  - We allow compact text previews and create row-slice chunks when rows are very wide.
  - Full-sheet text is not used for vector search beyond minimal slices; previews are for UI only.
- Analytics metadata:
  - Per-sheet numeric summaries and categorical distributions are computed for quick grounding.
- DuckDB:
  - All parsed sheets are registered per session for SQL queries.

### Dependencies and Installation
- duckdb>=0.9.2 (see chatbot_backend/requirements.txt)
- Install via: pip install -r chatbot_backend/requirements.txt

### Table Naming Convention
- Each sheet is registered under a sanitized table name:
  - {safe_base_filename}__{safe_sheet_name}
  - Non-alphanumeric characters are replaced with underscores, lowercased.

### Generating/Using SQL in Chat
- Example queries:
  - SELECT COUNT(*) FROM myfile_xlsx__sheet1;
  - ```sql
    SELECT category, SUM(amount) AS total
    FROM my_sales_xlsx__q1
    GROUP BY category
    ORDER BY total DESC
    LIMIT 10;
    ```
- Responses return a preview with row/column counts and a TSV of results.

### RAG Behavior Summary
- RAG index stores:
  - Text documents (.txt/.pdf/.docx) as chunked text.
  - Excel: only schema-driven column chunks and row-slice chunks (no full-sheet raw text).
- Downstream retrieval/embedding/context for LLM exclusively uses these chunks plus:
  - JIT DuckDB previews,
  - Precomputed analytics summaries,
  - Brief schema notes, and
  - DuckDB schema snapshots when relevant.
