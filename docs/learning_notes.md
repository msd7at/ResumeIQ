# ResumeIQ — Step-by-Step Learning Notes

> Every step of this project is documented here.
> Each entry has: What it does · Why this approach · AI/LangGraph concept · Interview Questions

---

## Phase 1 — RAG Pipeline

---

### Step 1.1 — Project Setup

**File(s) created:** `requirements.txt`, `.env`, full folder structure, `__init__.py` files

#### What this step does
Creates the complete project skeleton — all folders, all placeholder files, Python dependencies, and environment config. Every folder gets an `__init__.py` so Python treats it as a package and allows cross-module imports.

#### Why this approach
**Separation of concerns** — each folder has one job:
- `rag/` → only knows about parsing and vectors
- `graph/` → only knows about agents and LangGraph state
- `api/` → only knows about HTTP requests
- `db/` → only knows about SQLite

This makes every module independently testable and replaceable.

#### Library-by-library breakdown

| Library | What it does | Why not alternatives |
|---|---|---|
| `fastapi` | Builds REST API | Async-native, auto Swagger docs, faster than Flask |
| `uvicorn` | Runs FastAPI server | Standard ASGI server for FastAPI |
| `python-multipart` | Handles file uploads | FastAPI needs this to accept PDF/DOCX |
| `langgraph` | Wires agents into a state machine | Core differentiator — manages agent flow |
| `langchain` | Prompt templates, tool abstractions | Industry standard, works with everything |
| `langchain-ollama` | Connects LangChain to Ollama | Without this, LangChain can't talk to local LLMs |
| `ollama` | Python client for Ollama server | Calls llama3.1:8b and nomic-embed-text locally |
| `chromadb` | Stores resume vectors on disk | Zero setup, fully local |
| `PyMuPDF` | Extracts text from PDFs | Fastest PDF parser, built on C library MuPDF |
| `python-docx` | Extracts text from DOCX | Standard library for Word documents |
| `duckduckgo-search` | Web search, no API key | Google/Bing need paid keys |
| `pydantic` | Validates request/response data | FastAPI is built on Pydantic |
| `python-dotenv` | Loads `.env` into environment | Keeps secrets out of code |
| `aiofiles` | Async file read/write | FastAPI is async — blocking I/O would freeze server |

#### AI / LangGraph concept
**Project structure** is the foundation for a **multi-agent system**. LangGraph needs a clear separation between state, agents, and tools — which is exactly what this folder structure enforces.

#### Interview Questions

1. **"Why FastAPI over Flask or Django?"**
   FastAPI is async-native, auto-validates with Pydantic, generates Swagger docs at `/docs` automatically, and benchmarks faster than Flask for concurrent requests.

2. **"What is an ASGI server? Why Uvicorn?"**
   ASGI (Async Server Gateway Interface) handles async/concurrent requests. Uvicorn is the standard ASGI server for FastAPI. WSGI (Flask's model) is synchronous.

3. **"Why local LLM instead of OpenAI?"**
   Cost (free), privacy (resume data never leaves machine), no rate limits, no internet dependency.

4. **"What is ChromaDB and why not Pinecone/Weaviate?"**
   ChromaDB is a local, embedded vector database — zero setup, no account, no internet. Pinecone/Weaviate need cloud accounts and have usage limits.

5. **"What does `__init__.py` do?"**
   Makes a directory a Python package, enabling relative imports like `from app.rag.pdf_parser import parse_pdf`.

6. **"What is virtual environment and why use it?"**
   Isolated Python environment per project — avoids dependency version conflicts between different projects on same machine.

---

### Step 1.2 — SQLite Database Setup

**File(s) created:** `app/db/sqlite_client.py`

#### What this step does
Creates the database layer for the entire project. Defines 3 tables and provides helper functions to create, read, and update data. Called once at app startup via `init_db()`.

#### The 3 Tables

```
resume_sessions     → one row per resume upload
                      tracks the full lifecycle of a session

analysis_results    → stores what LangGraph agents produced
                      (issues, skills, questions, salary, companies)

chat_history        → stores every message in the follow-up chat
                      (role = "user" or "assistant")
```

#### Why this approach

**SQLite** chosen because:
- Ships with Python — no install needed
- Single file database (`resumeiq.db`) — easy to inspect
- Zero server setup — perfect for local/portfolio project
- In production → swap with PostgreSQL, same SQL queries mostly work

**JSON stored as TEXT** — SQLite has no native array/object type. Lists and dicts are serialized with `json.dumps()` on write and deserialized with `json.loads()` on read.

**`row_factory = sqlite3.Row`** — enables accessing columns by name (`row["status"]`) instead of index (`row[0]`).

#### AI / LangGraph concept
This is the **State Persistence** layer. LangGraph pipeline updates the `status` column as each agent completes:

```
uploaded → rag_ready → processing → completed
                ↓
        validation_failed
```

The `/status` API endpoint reads this column to show live progress to the frontend.

#### Interview Questions

1. **"SQLite kyun, PostgreSQL kyun nahi?"**
   Local project, zero setup, no server process needed. PostgreSQL for production — handles concurrent writes, has proper JSONB type, better for scale.

2. **"JSON columns good practice hai?"**
   Acceptable for small scale. In production, normalized tables are better (e.g., separate `skills` table). JSONB in PostgreSQL is a good middle ground.

3. **"Do FOREIGN KEY constraints work automatically in SQLite?"**
   No! SQLite requires `PRAGMA foreign_keys = ON` to enforce them. By default they are declared but not enforced — a common gotcha.

4. **"What does `row_factory = sqlite3.Row` do?"**
   Returns rows as dict-like objects so you can access columns by name instead of positional index. Makes code much more readable.

5. **"When is `init_db()` called?"**
   At application startup — in `main.py` inside FastAPI's `lifespan` context manager, before the server starts accepting requests.

6. **"Why TEXT primary key for sessions instead of INTEGER?"**
   UUID string (e.g. `sess_a1b2c3`) doesn't expose row count to API consumers. An integer ID like `id=1` tells attackers exactly how many sessions exist.

---

### Step 1.3 — PDF Parser

**File(s) created:** `app/rag/pdf_parser.py`

#### What this step does
Extracts plain text from a PDF resume file. Takes a file path → returns a single cleaned string. No AI involved — pure text extraction. This is the entry point of the RAG pipeline.

#### Key functions

```python
parse_pdf(file_path)   → opens PDF, extracts text page by page, joins and cleans
_clean_text(text)      → removes consecutive blank lines + non-printable characters
```

#### Why PyMuPDF

| Library | Speed | Accuracy | Notes |
|---|---|---|---|
| **PyMuPDF** | Very fast | High | Wraps C library MuPDF, best choice |
| pdfplumber | Slow | Medium | Good for tables |
| pypdf | Fast | Low | Misses formatting often |
| pdfminer | Medium | Medium | Complex API |

#### Why `import fitz` not `import pymupdf`
PyMuPDF is the Python wrapper — `fitz` is the name of the underlying C library (MuPDF). Legacy naming convention — both refer to the same package.

#### Digital PDF vs Scanned PDF
- **Digital PDF** (most resumes) → text is embedded → `get_text()` works directly
- **Scanned PDF** (image of paper) → text is a photo → needs OCR (e.g. Tesseract)
- This project handles digital PDFs only

#### AI / LangGraph concept
**Document Ingestion** — the first stage of a RAG pipeline:

```
[Document Ingestion]  ← Step 1.3 (PDF Parser)
        ↓
[Chunking]            ← Step 1.6
        ↓
[Embedding]           ← Step 1.7
        ↓
[Vector Store]        ← Step 1.8
        ↓
[Retrieval]           ← used by agents in Phase 2
```

LLMs cannot read binary files. Raw text extraction is mandatory before any AI processing.

#### Interview Questions

1. **"Why PyMuPDF over pdfplumber?"**
   PyMuPDF is 3-5x faster because it wraps a C library. For a web service that processes resumes, speed matters.

2. **"What is `fitz`?"**
   The C library underlying PyMuPDF. `import fitz` is the correct import even though you installed `PyMuPDF`.

3. **"What are the different modes in `page.get_text()`?"**
   `"text"` (plain), `"html"` (with markup), `"json"` (structured blocks), `"words"` (word-level bounding boxes), `"blocks"` (paragraph blocks). We use `"text"` — simplest and fastest.

4. **"What happens if someone uploads a scanned resume?"**
   `get_text()` returns empty string — the text is an image, not embedded. To handle this, you'd integrate OCR using `pytesseract` + `pdf2image` as a fallback.

5. **"Why `doc.close()` explicitly?"**
   Releases the file handle and frees C-level memory. Without it, the file stays locked on Windows. Could also use `with fitz.open(...) as doc:` pattern.

6. **"What is the RAG pipeline in simple terms?"**
   RAG = Retrieval Augmented Generation. Instead of asking LLM from memory, you give it relevant context from your own documents. Steps: ingest → chunk → embed → store → retrieve → generate.

---

### Step 1.4 — DOCX Parser

**File(s) created:** `app/rag/docx_parser.py`

#### What this step does

Extracts plain text from a `.docx` (Word) resume file. Two sources are read:

1. **Paragraphs** — headings, bullet points, normal text blocks
2. **Tables** — some resumes use Word tables to lay out skills or experience side-by-side

Returns a single cleaned string, same contract as the PDF parser.

#### Key functions

```python
parse_docx(file_path)   → opens DOCX, reads paragraphs + table cells, joins and cleans
_clean_text(text)       → same helper as pdf_parser — collapses blank lines, removes non-printable chars
```

#### Why python-docx

| Library | Notes |
|---|---|
| **python-docx** | Standard library for `.docx`, actively maintained, simple API |
| mammoth | Converts DOCX → HTML/Markdown, overkill for plain text extraction |
| textract | Broad format support but heavy dependency chain |

**`.docx` is a ZIP archive** — it contains `word/document.xml` inside. `python-docx` parses that XML and exposes Python objects (`paragraphs`, `tables`, `runs`). You never touch the XML directly.

#### Why read tables too?

Many resume templates (especially downloaded from Canva, Zety, Novoresume) use invisible Word tables to achieve two-column layouts. If you only read `doc.paragraphs`, all content inside table cells is silently skipped — skills section, contact info, work experience could vanish.

#### Difference from PDF Parser

| | PDF Parser (pdf_parser.py) | DOCX Parser (docx_parser.py) |
|---|---|---|
| Library | PyMuPDF (fitz) | python-docx |
| Source | Binary PDF, page-by-page | XML inside ZIP, paragraph-by-paragraph |
| Table support | Not needed (PDFs rarely use tables) | Needed (Word resumes often use tables) |
| OCR risk | Yes (scanned PDFs return empty) | No (DOCX always has real text) |

#### AI / LangGraph concept

Same as Step 1.3 — **Document Ingestion**, the first stage of the RAG pipeline. After this step, both PDF and DOCX resumes produce the same output format (a plain string), so everything downstream — chunker, embedder, vector store — is file-format agnostic.

```text
parse_pdf()   ──┐
                ├──→ plain text string → chunker → embedder → vector store
parse_docx()  ──┘
```

This is the **abstraction boundary** — the rest of the pipeline doesn't know or care whether the original file was PDF or DOCX.

#### Interview Questions

1. **"What is a .docx file internally?"**
   A ZIP archive. Inside is `word/document.xml` with all the content as XML. `python-docx` parses this XML and exposes Python objects. You can actually rename any `.docx` to `.zip` and open it.

2. **"Why did you read table cells separately?"**
   Many resume templates use Word tables for multi-column layouts. `doc.paragraphs` only returns text outside tables — any content inside table cells would be missed without iterating `doc.tables`.

3. **"What is a `Run` in python-docx?"**
   A `Run` is a segment of text within a paragraph that shares the same formatting (bold, italic, font size). A paragraph = multiple runs. For plain text extraction we don't need runs — `para.text` concatenates all runs in a paragraph automatically.

4. **"Can DOCX have scanned/image content like PDFs?"**
   Yes — `.docx` can embed images, and if a resume is a screenshot inserted into Word, there's no extractable text. But this is rare. Unlike scanned PDFs, it's not a common failure mode.

5. **"Why is `_clean_text()` duplicated in both parsers instead of a shared util?"**
   Valid concern. Could be moved to `app/rag/utils.py`. For now each parser is self-contained (easier to test in isolation). In a production codebase, shared helpers belong in a utils module.

6. **"How does this fit into the RAG pipeline?"**
   This is Document Ingestion — step 1 of RAG. The output of `parse_docx()` (raw text) feeds directly into the chunker (Step 1.6), which splits it into smaller pieces before embedding.

---

## Progress Tracker

| Step | Status | File |
|---|---|---|
| 1.1 Project setup | ✅ Done | `requirements.txt`, `.env`, folders |
| 1.2 SQLite setup | ✅ Done | `app/db/sqlite_client.py` |
| 1.3 PDF Parser | ✅ Done | `app/rag/pdf_parser.py` |
| 1.4 DOCX Parser | ⏳ Next | `app/rag/docx_parser.py` |
| 1.5 Validator | ⬜ Pending | `app/rag/validator.py` |
| 1.6 Chunker | ⬜ Pending | `app/rag/chunker.py` |
| 1.7 Embedder | ⬜ Pending | `app/rag/embedder.py` |
| 1.8 Vector Store | ⬜ Pending | `app/rag/vector_store.py` |
| 2.1 State design | ⬜ Pending | `app/graph/state.py` |
| 2.2 Resume Analyser Agent | ⬜ Pending | `app/graph/agents/resume_analyser.py` |
| 2.3 Dynamic Router | ⬜ Pending | `app/graph/agents/router.py` |
| 2.4 Question Generator | ⬜ Pending | `app/graph/agents/question_generator.py` |
| 2.5 Salary Agent | ⬜ Pending | `app/graph/agents/salary_agent.py` |
| 2.6 Report Compiler | ⬜ Pending | `app/graph/agents/report_compiler.py` |
| 2.7 Graph Builder | ⬜ Pending | `app/graph/graph_builder.py` |
| 3.1 DuckDuckGo Tool | ⬜ Pending | `app/tools/web_search.py` |
| 3.2 Company Q Integration | ⬜ Pending | — |
| 3.3 Salary Integration | ⬜ Pending | — |
| 3.4 End-to-end test | ⬜ Pending | — |
| 4.1 main.py | ⬜ Pending | `app/main.py` |
| 4.2 /upload endpoint | ⬜ Pending | `app/api/routes.py` |
| 4.3 /analyse endpoint | ⬜ Pending | `app/api/routes.py` |
| 4.4 /chat endpoint | ⬜ Pending | `app/api/routes.py` |
| 4.5 /status endpoint | ⬜ Pending | `app/api/routes.py` |
| 4.6 Streaming response | ⬜ Pending | `app/api/routes.py` |
| 5.1 Frontend connect | ⬜ Pending | `frontend/index.html` |
| 5.2 Status bar | ⬜ Pending | — |
| 5.3 Chat interface | ⬜ Pending | — |
| 5.4 Error handling | ⬜ Pending | — |
| 5.5 Architecture diagram | ✅ Done | `docs/architecture.html` |
