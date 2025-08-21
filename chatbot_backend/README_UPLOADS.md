# Large file uploads and Excel processing

This backend adds safeguards and tuning knobs to handle large .xlsx and other document uploads without causing upstream 504 Gateway Timeout:

Environment variables (set in .env):
- CHATBOT_MAX_UPLOAD_BYTES: Max size in bytes for a single uploaded file. Default: 52428800 (50 MB).
- CHATBOT_MAX_TOTAL_UPLOAD_BYTES: Max total size in bytes for all files in one request. Default: 104857600 (100 MB).
- CHATBOT_UPLOAD_PROCESS_TIMEOUT_SECS: Max seconds to spend in the request handler before returning a partial response. Default: 25.
- CHATBOT_BUILD_EMBEDDINGS_ON_UPLOAD: Whether to build embeddings during /chat/upload-context. Default: true.
- CHATBOT_PARSE_EXCEL_ON_UPLOAD: Whether to parse uploaded Excel to DataFrames and compute schema during upload. Default: true.
- CHATBOT_EXCEL_SCHEMA_MAX_SAMPLE_ROWS: Cap for rows per sheet used when building schema/stats to avoid heavy memory/time usage. Default: 10000.

Operational notes:
- The upload handler reads files in 1 MB chunks and enforces per-file and total limits.
- If processing exceeds the request wall time, it returns early with partial results to avoid 504. Embedding/index building can be offloaded to background tasks.
- Excel schema extraction performs sampling to limit heavy statistics on very large sheets. The generated schema includes "notes" and per-sheet "truncated" flags when sampling is applied.
- Excel text previews add explicit warnings like "[...] (preview truncated...)" and "[warning] Large sheet preview was truncated..." so users know when data is partially shown for performance.
- If `CHATBOT_PARSE_EXCEL_ON_UPLOAD` is `false`, Excel parsing is deferred but the UI preview includes a notice that schema will be built at query time.

Common pitfalls:
- A 422 Unprocessable Entity typically indicates the request was not sent as multipart/form-data. Ensure you send:
  - form field 'session_id' (string)
  - file field(s) 'files' (one or more). Supported: .txt, .pdf, .docx, .xlsx.

Troubleshooting large Excel files not appearing:
- Ensure total request size and per-file size are below the configured limits and any reverse proxy limits.
- Check that previews show truncation warnings; large sheets are included even when truncated.
- If Excel parsing is skipped due to memory or time limits, the file still appears with a warning in the preview; schema is either deferred or partially built from sampled rows.
- Increase `CHATBOT_EXCEL_SCHEMA_MAX_SAMPLE_ROWS` cautiously if you need deeper schema statistics, or decrease it to reduce CPU/memory pressure.

Additional diagnostics for /chat/excel-query:
- Server logs now include:
  - "[excel_query] Rebuilt schema..." when an empty or invalid schema is detected and rebuilt.
  - "[excel_query] Prompt preview..." showing the first up to 1000 characters of the prompt sent to Gemini (includes a compact per-sheet summary and a safely truncated JSON schema preview).
  - "[excel_query] Schema summary: sheets=..., cols_per_sheet=[...]" to confirm schema richness.
- These logs help confirm that the Gemini prompt is never empty and includes representative schema even for very large files (with sampling/truncation noted in schema notes).
