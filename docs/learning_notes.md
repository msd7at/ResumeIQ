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

### Step 1.5 — Resume Validator

**File(s) created:** `app/rag/validator.py`

#### What this step does

Scans the extracted resume text and checks whether critical fields are present. Returns a `ValidationResult` with three pieces of info:

- `is_valid` — `True` only if all mandatory fields are found
- `missing_fields` — list of things that are definitely absent (email, phone, sections)
- `warnings` — things that are absent but optional (projects, summary)

If `is_valid` is `False`, the LangGraph pipeline sets the session status to `validation_failed` and stops — no point embedding an incomplete resume.

#### What is checked

| Check | Type | Logic used |
|---|---|---|
| Email address | Mandatory | Regex `[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}` |
| Phone number | Mandatory | Regex for digit patterns with optional `+`, spaces, dashes |
| Skills section | Mandatory | Keyword match: "skill", "technologies", "tech stack", "tools" |
| Experience section | Mandatory | Keyword match: "experience", "employment", "work history" |
| Education section | Mandatory | Keyword match: "education", "degree", "university", "college" |
| Projects section | Warning | Keyword match: "project", "portfolio" |
| Summary section | Warning | Keyword match: "summary", "objective", "profile", "about" |
| Word count < 50 | Mandatory | Resume too short — probably parse failure |
| Word count < 150 | Warning | Resume seems thin |

#### Why regex for email/phone?

These are structured patterns — regex is the right tool. LLM would be overkill (slow, non-deterministic) for detecting whether an email address exists.

#### Why keyword matching for sections?

Resume headings vary wildly: "Work Experience", "Professional Experience", "Employment History", "Career" — all mean the same thing. A keyword list covers the real-world variation without needing an LLM.

#### The `@dataclass` + `field(default_factory=list)` pattern

```python
@dataclass
class ValidationResult:
    is_valid: bool
    missing_fields: list[str] = field(default_factory=list)
```

`field(default_factory=list)` is required for mutable defaults in dataclasses. If you wrote `missing_fields: list = []` directly, Python would share the same list object across all instances — a classic Python gotcha.

#### AI / LangGraph concept

This is a **guard node** in the LangGraph pipeline. Before the expensive RAG steps (chunking, embedding), we validate that the input is worth processing. If validation fails, the graph routes to an early exit:

```text
[parse_pdf / parse_docx]
        ↓
  [validate_resume]  ←── Step 1.5
     ↓          ↓
  valid      invalid
    ↓            ↓
[chunker]   status = "validation_failed"
                ↓
         return error to frontend
```

This is the **conditional edge** pattern in LangGraph — one node, two possible next nodes depending on state.

#### Interview Questions

1. **"Why validate before chunking/embedding?"**
   Embedding is the slowest step (calls Ollama for each chunk). No point running it on a resume with no email or no skills section — we catch bad input early and save time.

2. **"Why not use the LLM to detect missing fields?"**
   Regex and keyword matching are deterministic, instant, and free. LLMs are probabilistic, slow (local inference = seconds), and have no advantage here. Use LLMs for judgment calls; use rules for pattern matching.

3. **"What is a Python dataclass?"**
   A class decorator that auto-generates `__init__`, `__repr__`, `__eq__` based on declared fields. Cleaner than a plain dict for structured return values — fields have names, types, and IDE autocomplete.

4. **"Why `field(default_factory=list)` instead of `= []`?"**
   Mutable default arguments in Python are shared across all instances. `field(default_factory=list)` tells the dataclass to call `list()` fresh for each new instance. Classic Python gotcha.

5. **"What happens in the pipeline when validation fails?"**
   `is_valid = False` → agent sets `session.status = "validation_failed"` in SQLite → LangGraph router sends to an early exit node → `/status` API returns the missing fields to the frontend → user sees "Please add your email and skills section".

6. **"How would you extend this validator?"**
   Add LinkedIn URL check, GitHub URL check, detect if dates are missing from experience entries, check if job titles are present, detect very generic skills ("Microsoft Office") that weaken a tech resume.

---

### Step 1.6 — Smart Sectional Chunker

**File(s) created:** `app/rag/chunker.py`

#### What this step does

Splits the raw resume text (one long string) into small focused **chunks** that each get embedded as a separate vector. Each chunk knows which section it came from.

Output: list of `Chunk` dataclass objects, each with:
- `section` — e.g. `"SKILLS"`, `"EXPERIENCE"`, `"HEADER"`
- `text` — content prefixed with `[SECTION]` label
- `chunk_index` — position in resume (0-based)
- `metadata` — dict stored in ChromaDB alongside the vector

#### Why chunk at all?

LLMs and embedding models have a **context window limit**. More importantly, if the whole resume is one vector, it blends all topics. A query like "what Python skills does this person have?" needs only the `SKILLS` chunk — not the entire resume mixed together.

#### Two-level splitting strategy

```text
Level 1 — Section detection  (semantic boundary)
  Lines like "EXPERIENCE", "Skills", "Education" → split here
  Each section = its own group of lines

Level 2 — Size cap  (MAX_CHUNK_CHARS = 800)
  If a section is > 800 chars → split further
  Last OVERLAP_LINES (2) carried over to next chunk for context continuity
```

#### Why section-aware vs fixed-size chunking?

| Strategy | Problem |
|---|---|
| Fixed size (every 500 chars) | Splits mid-sentence, mixes sections randomly |
| Recursive splitter | Better, but still section-unaware |
| **Section-aware (this approach)** | Natural boundaries = coherent chunks = better retrieval |

In a resume, sections ARE the natural semantic boundaries.

#### The `[SECTION]` prefix

Each chunk text starts with `[EXPERIENCE]` or `[SKILLS]`. This tells the LLM which part of the resume it's reading when a chunk is retrieved and injected into a prompt.

#### Overlap (`OVERLAP_LINES = 2`)

Last 2 lines of a chunk repeat at the start of the next chunk within the same section. Prevents a sentence being cut in half — the model always has surrounding context.

#### AI / LangGraph concept

Chunking is **Stage 2** of the RAG pipeline:

```text
[Document Ingestion]  ← Steps 1.3–1.4
        ↓
   [Chunking]         ← Step 1.6 ✅
        ↓
   [Embedding]        ← Step 1.7
        ↓
  [Vector Store]      ← Step 1.8
```

Chunk quality = retrieval quality = LLM answer quality. This is called **chunk granularity** — one of the most critical RAG design decisions.

#### Interview Questions

1. **"Why not send the whole resume to the LLM?"**
   Local LLMs have a limited context window (~8K tokens). More importantly, a focused chunk gives the LLM exactly what it needs — sending 2000 words when only 200 are relevant dilutes answer quality.

2. **"What is chunk overlap and why use it?"**
   Repeating the last N lines at the start of the next chunk avoids cutting a sentence across two chunks where neither chunk has full meaning. Without overlap, "Led a team of 5 engineers" could be split and both halves lose context.

3. **"Fixed-size vs semantic chunking — which is better?"**
   Semantic (section-aware) is better for structured documents like resumes where sections have clear meaning. Fixed-size is simpler but can mix unrelated content in one chunk.

4. **"What goes into the `metadata` dict?"**
   `session_id` and `section`. ChromaDB stores this alongside the vector. Agents can then filter: "retrieve only SKILLS chunks for session XYZ" — not all chunks from all resumes.

5. **"How did you pick `MAX_CHUNK_CHARS = 800`?"**
   Empirical — a typical resume section (skills list, one job entry) fits in 300–600 chars. 800 allows a full experience entry with bullet points without arbitrary cuts. Too small = too many noisy chunks. Too large = embeddings lose focus.

6. **"How would you improve this chunker?"**
   Use spaCy for sentence segmentation at sub-section level. Detect bullet points as natural sub-boundaries. Add page number metadata. Use LLM-based section detection for non-standard headings.

---

### Step 1.7 — Ollama Embeddings

**File(s) created:** `app/rag/embedder.py`

#### What this step does

Takes the list of `Chunk` objects produced by the chunker and converts each chunk's text into a **vector** (a list of floating-point numbers) using Ollama's `nomic-embed-text` model running locally. Also provides a second function to embed a single query string at retrieval time.

Two functions:

```python
embed_chunks(chunks)  → list of dicts ready to insert into ChromaDB
embed_query(query)    → single list[float] used at search time
```

#### What is an embedding?

An embedding is a list of numbers that encodes the **meaning** of a piece of text. `nomic-embed-text` produces 768-dimensional vectors (768 floats per chunk).

Texts that are semantically similar → vectors that are close together in 768-dimensional space. This is what makes similarity search possible — ChromaDB finds the chunks whose vectors are closest to the query vector.

```text
"Python developer with FastAPI experience"
       ↓  nomic-embed-text
[0.021, -0.134, 0.887, ... 768 numbers total]
```

#### Why two separate functions?

| Function | When called | Input |
|---|---|---|
| `embed_chunks()` | Upload time — once per resume | list of Chunk objects |
| `embed_query()` | Every agent question | single string |

At upload time you embed all chunks and store them. At query time you embed just the question and find similar stored chunks. The model must be the same for both — you can't compare vectors from different models.

#### What `embed_chunks()` returns

Each dict in the list has exactly what ChromaDB needs:

```python
{
    "id":        "sess_a1b2c3_4",      # unique — session + chunk index
    "text":      "[SKILLS]\nPython ...", # original chunk text
    "embedding": [0.021, -0.134, ...],  # 768 floats
    "metadata":  {"session_id": "...", "section": "SKILLS"}
}
```

The `id` is `{session_id}_{chunk_index}` — unique per session so two different resumes don't collide in ChromaDB.

#### Why Ollama / nomic-embed-text?

| Option | Cost | Privacy | Speed |
|---|---|---|---|
| OpenAI text-embedding-3 | Paid per token | Resume data leaves machine | Fast (network) |
| **nomic-embed-text (Ollama)** | Free | Fully local | Fast (local GPU/CPU) |
| sentence-transformers | Free | Local | Needs separate Python model |

`nomic-embed-text` is specifically optimized for retrieval tasks and produces high-quality 768-dim vectors. It's the standard choice for local RAG setups.

#### AI / LangGraph concept

Embedding is **Stage 3** of the RAG pipeline:

```text
[Ingestion]   ✅  parse_pdf / parse_docx
[Validation]  ✅  validate_resume
[Chunking]    ✅  chunk_resume
[Embedding]   ✅  embed_chunks / embed_query  ← Step 1.7
[Vector DB]   ⏳  Step 1.8
```

`embed_query()` is called at **retrieval time** by the agents in Phase 2. The query "what Python skills does this person have?" gets embedded, ChromaDB finds the nearest chunk vectors, and those chunks are injected into the LLM prompt.

#### Interview Questions

1. **"What is an embedding?"**
   A fixed-length list of floats that represents the semantic meaning of text. Similar meanings → similar vectors. Produced by an encoder model, not a generative LLM.

2. **"What is nomic-embed-text? Why 768 dimensions?"**
   An open-source embedding model optimized for retrieval. 768 is the output size of the model's encoder — each dimension captures some aspect of meaning. OpenAI's `text-embedding-3-small` uses 1536 dims. More dims ≠ always better.

3. **"Why must `embed_query` use the same model as `embed_chunks`?"**
   Different models produce vectors in completely different spaces. Comparing a nomic vector with an OpenAI vector is meaningless — like comparing temperatures in Celsius and Fahrenheit without converting.

4. **"What is cosine similarity?"**
   The standard metric for comparing embedding vectors. Measures the angle between two vectors (not their length). Value ranges from -1 to 1; closer to 1 = more similar in meaning. ChromaDB uses this by default.

5. **"Why is `embed_query` called at every agent question but `embed_chunks` only once?"**
   Chunks are static — a resume doesn't change after upload. Queries change every time an agent asks a new question. Embedding is fast (~50ms locally) but there's no point re-embedding the same chunks repeatedly.

6. **"What would happen if you sent the whole resume as one embedding instead of chunks?"**
   One 768-dim vector would represent everything — Python skills, Java history, education, hobbies — blended together. A query about Python would get diluted by all the other content. Chunked embeddings give precise, focused retrieval.

---

## Progress Tracker

| Step | Status | File |
|---|---|---|
| 1.1 Project setup | ✅ Done | `requirements.txt`, `.env`, folders |
| 1.2 SQLite setup | ✅ Done | `app/db/sqlite_client.py` |
| 1.3 PDF Parser | ✅ Done | `app/rag/pdf_parser.py` |
| 1.4 DOCX Parser | ✅ Done | `app/rag/docx_parser.py` |
| 1.5 Validator | ✅ Done | `app/rag/validator.py` |
| 1.6 Chunker | ✅ Done | `app/rag/chunker.py` |
| 1.7 Embedder | ✅ Done | `app/rag/embedder.py` |
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
