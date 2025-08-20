# Project Repository

This is the initial README file for the project.

## DuckDB Integration (Backend)

- The FastAPI backend integrates DuckDB as an in-memory SQL engine for uploaded Excel/tabular data.
- For each uploaded .xlsx file, the backend:
  - Parses sheets into pandas DataFrames.
  - Registers each sheet as a DuckDB table for the user's session.
  - Executes inline SQL (e.g., queries starting with SELECT/WITH or fenced as ```sql ... ```).
  - Returns compact, preview-sized results for chat responses.

Key notes:
- In-memory only. No persistent DuckDB database is created or stored.
- Useful for repeated or large-data queries; only the necessary rows/columns are returned to the client.
- The chat endpoint detects inline SQL and executes it via DuckDB automatically.
- Analytics helpers still exist; if SQL is not provided, the system attempts to compute simple stats or falls back to LLM with minimal necessary context.

### Dependencies and Installation
- duckdb>=0.9.2 (added to chatbot_backend/requirements.txt)
- Install via: pip install -r chatbot_backend/requirements.txt

### Table Naming Convention
- Each sheet is registered under a sanitized table name:
  - {safe_base_filename}__{safe_sheet_name}
  - Non-alphanumeric characters are replaced with underscores, lowercased.

### Generating/Using SQL in Chat
- You can send queries like:
  - SELECT COUNT(*) FROM myfile_xlsx__sheet1;
  - ```sql
    SELECT category, SUM(amount) AS total
    FROM my_sales_xlsx__q1
    GROUP BY category
    ORDER BY total DESC
    LIMIT 10;
    ```
- The response returns a preview with row and column counts and a TSV of results.
