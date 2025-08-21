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
- The upload handler now reads files in 1 MB chunks asynchronously and enforces per-file and total limits.
- If processing exceeds the request wall time, it returns early with partial results to avoid 504. Embedding/index building can be offloaded to background tasks.
- Excel schema extraction performs sampling to limit heavy statistics on very large sheets. You can set CHATBOT_PARSE_EXCEL_ON_UPLOAD=false to defer full Excel parsing to later endpoints if needed.
