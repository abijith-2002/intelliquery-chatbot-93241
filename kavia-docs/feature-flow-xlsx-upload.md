# XLSX File Upload Flow

## Introduction

### Background
This document explains the end-to-end flow for uploading an .xlsx file in the IntelliQuery chatbot application. It describes how the user initiates the upload from the frontend, how the file is transmitted to the FastAPI backend, how the backend validates, parses, and stores file-derived data, and how this data is used to enhance chat responses. It also enumerates the relevant API contract, validations, storage approach, and dependencies on environment variables and third-party libraries.

### Scope
The scope covers the backend implementation present in the repository and the expected frontend behavior to call the backend. It focuses specifically on the .xlsx file path but also notes shared handling for other supported file types where relevant.

## Frontend Initiation

### User Action
- The user selects one or more files to attach as context for a chat session. XLSX files are supported alongside PDF, DOCX, and TXT.
- The frontend submits a multipart/form-data POST request to the backend endpoint /chat/upload-context with:
  - A form field session_id containing the current chat session identifier (string).
  - One or more file parts files[], each being the selected file. XLSX files should have a filename ending with .xlsx.

### Request Construction
- Method: POST
- URL: {API_BASE_URL}/chat/upload-context
  - In development, API_BASE_URL is typically configured in the frontend via an environment variable such as REACT_APP_API_BASE_URL.
- Headers: Content-Type: multipart/form-data (automatically set by the browser when using FormData)
- Body: FormData including:
  - session_id: string
  - files: File objects (one or multiple); at least one must be provided

Example pseudocode for the frontend:
```javascript
const form = new FormData();
form.append("session_id", sessionId);
files.forEach(f => form.append("files", f, f.name));
await fetch(`${API_BASE_URL}/chat/upload-context`, {
  method: "POST",
  body: form
});
```

## Backend API and Contract

### Endpoint
- Method: POST
- Path: /chat/upload-context
- Module: src/api/main.py
- Tags: Chat
- Summary: Upload context files for a chat session
- Request:
  - Form field session_id: str (required)
  - Form field files: List[UploadFile] (required), supports .docx, .xlsx, .pdf, .txt
- Response model: UploadContextResponse
  - session_id: string
  - files_processed: array of UploadedFileResult
    - filename: string
    - size: integer (bytes)
    - content_chars: integer (characters extracted)
    - preview: string (short, collapsed content preview)
    - error: string or null (error per file if extraction failed)
  - total_chars: integer
  - message: string

### Status Codes
- 200: Files processed. If content was extracted, session context is updated and indexed.
- 400:
  - If session_id is missing/empty
  - If files are not provided or empty
- 415: Unsupported media type (surfaced indirectly through extraction error messages)
- 500: Internal errors

## Backend Processing Flow

### High-Level Steps
When /chat/upload-context is called, the backend performs the following:

1. Input validation
   - Ensures session_id exists and is a non-empty string.
   - Ensures at least one file is provided.

2. Iterate each uploaded file
   - Reads raw bytes from the file (UploadFile).
   - Captures the original filename and size.

3. XLSX-specific caching for later data lookup
   - For files with .xlsx extension:
     - Stores raw XLSX bytes in a per-session in-memory cache at SESSION_META[session_id]["__xlsx_cache__"][filename] = bytes.
     - Caps cache to the last 3 XLSX files to bound memory usage (evicts oldest).

4. Text extraction and preview
   - Calls extract_text_from_bytes(filename, bytes) from src/api/file_utils.py.
     - For .xlsx files, this uses openpyxl to iterate sheets and rows, building a textual representation:
       - Sheet headers as [Sheet: <title>]
       - Tab-separated rows
       - Skips completely empty rows
     - Returns (text, error) where error is None on success.
   - Creates a concise preview using summarize_text_preview(text, max_chars=500).
   - Calculates content_chars as the length of the extracted text.

5. XLSX header preprocessing and embeddings
   - For .xlsx files, attempts to extract column headers using a light-weight routine:
     - _extract_xlsx_headers_from_bytes (in main.py) reads the first non-empty row of each sheet as headers.
     - Deduplicates headers while preserving order.
   - Embeds the header names via Google Gemini embeddings if an API key is configured:
     - _store_session_header_embeddings(session_id, headers) generates embeddings and stores:
       - SESSION_XLSX_HEADER_EMBEDDINGS[session_id] = {
         headers: [...],
         embeddings: [vector or None],
         embedding_model: "models/text-embedding-004"
       }
     - If embeddings are not available (e.g., no API key), the headers are stored with embeddings entries as None.

6. Building a per-session retrieval index (RAG)
   - If text was successfully extracted (any file type), the backend:
     - Splits the text into overlapping chunks (_split_into_chunks, default around 180 words with 40 overlap).
     - Embeds each chunk via Gemini embeddings if available; otherwise stores None for embeddings.
     - Stores chunks and embeddings into RAG_INDEX_STORE[session_id]:
       - {
           chunks: [{ text, filename }, ...],
           embeddings: [[float] or None, ...],
           embedding_model: "models/text-embedding-004"
         }
   - This index supports semantic retrieval when answering future chat queries.

7. Session-wide combined context (legacy)
   - If any text was extracted:
     - The system updates a legacy combined text store in CONTEXT_STORE[session_id] with:
       - files: the list of per-file UploadedFileResult
       - combined: concatenation of previous combined content and the newly extracted text

8. Assemble response
   - files_processed: list of per-file results (filename, size, content_chars, preview, error)
   - total_chars: total characters extracted from all files in the request
   - message:
     - "Processed files successfully. Session context updated and indexed." if total_chars > 0
     - "Processed files, but no readable content was extracted." otherwise

### Data Structures (In-Memory)
- SESSION_META
  - Stores meta per session, including:
    - __xlsx_cache__: dict of filename -> raw XLSX bytes (last up to 3 files)
    - last_classification: recent chat classification metadata (not directly part of upload)
- SESSION_XLSX_HEADER_EMBEDDINGS
  - Per session, stores:
    - headers: list of unique column names from uploaded XLSX files
    - embeddings: per-header embedding vectors or None
    - embedding_model: string identifier
- RAG_INDEX_STORE
  - Per session, stores:
    - chunks: text chunks and their source filename
    - embeddings: vector for each chunk, or None
    - embedding_model: string identifier
- CONTEXT_STORE
  - Legacy utility for storing combined extracted text and file summaries

## Using the Uploaded XLSX in Chat

### Classification and Type 1 Data Lookup
- During subsequent calls to POST /chat, the query is classified by classify_query_type into:
  - Type 1: data lookup (keywords such as "show", "list", "get rows", etc.)
  - Type 2: knowledge/explanation (default when not matched)
- For Type 2:
  - The system bypasses all Pandas/XLSX execution and only uses semantic retrieval of text chunks from RAG_INDEX_STORE for additional context.
- For Type 1:
  - If the per-session XLSX cache contains at least one file:
    - The most recent XLSX bytes are selected.
    - The backend constructs an XLSX schema from bytes via extract_xlsx_schema_from_bytes (src/api/xlsx_pandas_exec.py).
    - It picks top-k relevant headers via get_top_k_relevant_columns_for_query using the previously stored header embeddings/names.
    - If no relevant columns are found, it returns a direct message "No relevant column found in the uploaded Excel."
    - Otherwise, it asks Gemini to generate safe, read-only Pandas code (generate_pandas_code_with_gemini).
    - The code is statically validated for safety and executed against the XLSX in a restricted environment (execute_safe_pandas_code_on_xlsx).
    - If execution fails, the backend retries by prompting Gemini with the error to correct the code once, then executes again.
    - Returns the rendered table or small summary string back to the user as the chat answer.
  - If no XLSX is available or anything fails in the pandas path, the system falls back to a natural-language answer using vector-retrieved text context.

## Validations and Safety

### Upload Validations
- session_id must be a non-empty string (400 if invalid).
- At least one file must be provided (400 if missing).
- File reading failures are reported per-file in files_processed[i].error.
- Unsupported file types return an error message from extract_text_from_bytes but do not crash the entire request.

### XLSX Safety for Data Lookup
- Code generation constraints:
  - Read-only operations
  - No imports
  - No file/network access
  - The result must be assigned to RESULT
  - Output is limited to a small size (DataFrame head(20) at most)
- Static AST validation:
  - Rejects imports, exec/eval, input/open, usage of __builtins__ or other dunder access
- Restricted execution environment:
  - SAFE_BUILTINS only
  - Provides pd (pandas) and dfs (sheet_name -> DataFrame) to the code
- Result rendering:
  - DataFrame rendered to text table with limited rows
  - Non-DataFrame objects summarized to concise text

## Storage and Persistence

### In-Memory Only
- All session state related to uploads (RAG index, header embeddings, XLSX bytes cache, combined text) is kept in memory on the backend process.
- There is no persistent file storage or external object storage used for the uploaded files in the current implementation.
- If the backend restarts, uploaded context is lost.

## Relevant Environment Variables

### Google Gemini API Key
- Required for:
  - Generating chat responses via Gemini
  - Computing embeddings for RAG chunks and header names
  - Generating and correcting Pandas code for Type 1 data lookup
- Resolution order (src/api/config_utils.py):
  1. GEMINI_API_KEY
  2. REACT_APP_GEMINI_API_KEY
  3. GOOGLE_API_KEY
  4. GOOGLE_GEMINI_API_KEY
- If none are provided, Gemini-powered functionality gracefully degrades:
  - Vector embeddings for chunks/headers are stored as None
  - Type 1 data lookup requiring code generation will fail with "Gemini API key not configured" if invoked
  - Natural-language fallback may still respond, but without embedding-based enhancements

### Frontend Base URL
- The frontend commonly uses REACT_APP_API_BASE_URL to direct requests to the backend service URL.
- This variable is handled on the frontend side; the backend is CORS-enabled to accept cross-origin requests.

### Database URL (Auth)
- CHATBOT_SQLALCHEMY_DATABASE_URL (optional)
  - Defaults to sqlite:///./chatbot_users.db
  - Used only for user registration/login; not directly involved in file upload

### Other Provided Frontend Variables
- Additional variables like REACT_APP_SUPABASE_DB_URL, REACT_APP_SUPABASE_URL, REACT_APP_SUPABASE_ANON_KEY, REACT_APP_SUPABASE_SERVICE_ROLE_KEY, REACT_APP_env are not used by the backend upload flow as implemented. There is no Supabase integration in the current backend code.

## Third-Party Libraries and Services

### Libraries
- FastAPI and Starlette: HTTP server and request handling
- pydantic: Request/response schemas
- openpyxl: Reading XLSX content and extracting headers
- pandas: DataFrame operations for Type 1 data lookup
- rapidfuzz: Fuzzy matching (for column relevance)
- google-generativeai: Gemini models for embeddings and content/code generation
- pdfminer.six: PDF text extraction
- python-docx: DOCX text extraction
- langchain: ConversationBufferMemory (chat history)

### External Service
- Google Gemini API:
  - Used for text generation, embeddings, and code generation
  - Requires a valid API key in environment

## Mermaid Sequence Diagram

```mermaid
sequenceDiagram
    participant U as User
    participant F as Frontend (React)
    participant B as Backend (FastAPI)
    participant G as Google Gemini API

    U->>F: Select .xlsx file(s) and submit
    F->>B: POST /chat/upload-context (multipart/form-data: session_id, files[])
    B->>B: Validate inputs (session_id, files)
    alt For each .xlsx
        B->>B: Cache raw bytes in SESSION_META[session_id]["__xlsx_cache__"]
        B->>B: Extract text via file_utils.extract_text_from_bytes
        B->>B: Extract headers via _extract_xlsx_headers_from_bytes
        B->>G: Embed headers (if API key available)
        G-->>B: Header embeddings (or failure -> None)
        B->>B: Store headers+embeddings in SESSION_XLSX_HEADER_EMBEDDINGS
    else Other file types
        B->>B: Extract text via file_utils
    end
    B->>G: Embed chunks for RAG (if API key available)
    G-->>B: Chunk embeddings (or failure -> None)
    B->>B: Update RAG_INDEX_STORE with chunks+embeddings
    B-->>F: UploadContextResponse (files_processed[], total_chars, message)

    U->>F: Ask a question
    F->>B: POST /chat
    B->>B: Classify query (Type 1 vs Type 2)
    opt Type 1 + XLSX available
        B->>B: Build schema (extract_xlsx_schema_from_bytes)
        B->>B: Select top-k headers (get_top_k_relevant_columns_for_query)
        B->>G: Generate Pandas code (generate_pandas_code_with_gemini)
        G-->>B: Code
        B->>B: Validate + execute safely on XLSX
        alt Execution error
            B->>G: Retry code gen with error feedback
            G-->>B: Corrected code
            B->>B: Validate + execute safely
        end
        B-->>F: ChatAnswerResponse (rendered table/summary)
    else Type 2 or fallback
        B->>B: Retrieve top chunks (semantic or lexical)
        B->>G: Generate NL answer with optional context
        G-->>B: Answer
        B-->>F: ChatAnswerResponse (natural language answer)
    end
```

## Example Backend Responses

### Successful Upload (XLSX + others)
- 200 OK
- Body:
  - session_id: "abc123"
  - files_processed: [{
    filename: "sheet.xlsx",
    size: 12345,
    content_chars: 9876,
    preview: "...short preview...",
    error: null
  }, ...]
  - total_chars: 10000
  - message: "Processed files successfully. Session context updated and indexed."

### No Readable Content
- 200 OK with:
  - message: "Processed files, but no readable content was extracted."

### Invalid Input
- 400 with detail indicating missing session_id or files.

## Notes and Limitations

- In-memory storage: Upload-derived state is not persisted. Service restarts clear session context, header embeddings, RAG index, and XLSX cache.
- Memory bounds: XLSX cache is limited to 3 latest files per session. RAG and header stores grow with extracted content; consider pruning strategies in high-volume environments.
- Security: The XLSX data lookup flow enforces strict code safety via static checks and restricted execution. Still, always review logs and consider further sandboxing for untrusted data in production.
- Frontend: Ensure the frontend passes a consistent session_id across the upload and chat requests so the backend can correlate the context with the conversation.

## References

- API and logic:
  - src/api/main.py
  - src/api/file_utils.py
  - src/api/xlsx_pandas_exec.py
- Dependencies:
  - chatbot_backend/requirements.txt

