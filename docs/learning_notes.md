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

### Step 1.8 — ChromaDB Vector Store

**File(s) created:** `app/rag/vector_store.py`

#### What this step does

Wraps ChromaDB operations into 4 clean functions. This is where vectors go to live on disk and where agents come to search.

| Function | Purpose |
|---|---|
| `store_embeddings(embedded_chunks)` | Bulk insert chunks+vectors into ChromaDB |
| `retrieve_chunks(query_embedding, session_id, ...)` | Top-n similarity search for a query |
| `delete_session(session_id)` | Remove all chunks for a session (re-upload cleanup) |
| `count_chunks(session_id)` | How many chunks stored for a session |

#### ChromaDB internals

ChromaDB stores 3 things per entry:
1. **`id`** — unique string identifier
2. **`document`** — the original text (chunk text with `[SECTION]` prefix)
3. **`embedding`** — the 768-float vector
4. **`metadata`** — dict (`session_id`, `section`) for filtering

It uses an **HNSW index** (Hierarchical Navigable Small World) for approximate nearest-neighbor search — fast even with millions of vectors.

#### Why `"hnsw:space": "cosine"`?

ChromaDB defaults to `l2` (Euclidean distance). We set it to `cosine` explicitly because:
- Cosine measures the **angle** between vectors — captures semantic similarity regardless of vector magnitude
- `l2` measures raw distance — can rank long-text vectors differently than short-text vectors even if meanings are similar
- For NLP embeddings, cosine is the standard choice

**Important:** Once a collection is created with a distance metric, you can't change it. Always specify `cosine` upfront.

#### Why always filter by `session_id`?

Multiple users upload different resumes. Without the filter, a question about "Python skills" could retrieve chunks from a completely different person's resume. The `session_id` filter scopes every query to one resume only.

```python
where = {"session_id": session_id}             # single filter
where = {"$and": [{"session_id": ...}, {"section": "SKILLS"}]}  # combined filter
```

#### `PersistentClient` vs `Client`

| Client | Storage | Use case |
|---|---|---|
| `chromadb.Client()` | In-memory only | Testing, throwaway |
| `chromadb.PersistentClient(path=...)` | On disk at `path` | Production, this project |
| `chromadb.HttpClient(host=...)` | Remote server | Distributed/cloud |

We use `PersistentClient` — data survives restarts, stored at `./data/chroma_db`.

#### Complete RAG pipeline — all 5 stages done

```text
[Ingestion]   ✅  parse_pdf() / parse_docx()
[Validation]  ✅  validate_resume()
[Chunking]    ✅  chunk_resume()
[Embedding]   ✅  embed_chunks() / embed_query()
[Vector DB]   ✅  store_embeddings() / retrieve_chunks()
```

Phase 1 complete. Phase 2 begins: LangGraph agents will call `retrieve_chunks()` + `embed_query()` to power RAG-based resume analysis.

#### Interview Questions

1. **"What is ChromaDB? Why not Pinecone?"**
   ChromaDB is a local, embedded vector database — zero setup, no account, stores data on disk. Pinecone is a managed cloud service. For a local/portfolio project, ChromaDB is the right choice — no API keys, no cost, no internet.

2. **"What is HNSW?"**
   Hierarchical Navigable Small World — a graph-based approximate nearest-neighbor algorithm. Instead of comparing a query vector to every stored vector (brute force), HNSW navigates a layered graph to find similar vectors in `O(log n)` time. It's the standard index for production vector databases.

3. **"Why cosine similarity over Euclidean distance for text?"**
   Cosine measures the angle between vectors — two texts can have the same meaning regardless of their length. A short phrase and a long paragraph about Python can score high cosine similarity. Euclidean distance is influenced by vector magnitude, which can vary with text length.

4. **"What does `include=[]` do in `count_chunks`?"**
   Tells ChromaDB not to return documents, embeddings, or metadatas — just the IDs. This makes the call faster since we only need the count, not the actual content.

5. **"What happens if two resumes are stored for the same session_id?"**
   Old and new chunks would both exist with the same `session_id`. That's why `delete_session()` is called before re-embedding on re-upload — it clears old vectors first. Otherwise retrieval would get results from both the old and new resume.

6. **"How would you scale this to production?"**
   Replace `PersistentClient` with `HttpClient` pointing to a dedicated ChromaDB server (or Weaviate/Qdrant/Pinecone). The `retrieve_chunks` and `store_embeddings` interface stays the same — only the client changes. This is why the client is created in `_get_collection()` and not passed in — easy to swap.

---

### Step 2.1 — LangGraph State Design

**File(s) created:** `app/graph/state.py`

#### What this step does

Defines `ResumeState` — a single `TypedDict` that is the **shared memory** of the entire LangGraph pipeline. Every agent reads from it and writes back to it. Also provides `create_initial_state()` to build the starting state before any agent runs.

#### What is a TypedDict?

A Python dict with declared keys and types — no extra class machinery, no `__init__`, just a type hint contract. LangGraph requires state to be a TypedDict (or dataclass). It gives IDE autocomplete and type safety on state fields.

```python
state["skills_found"]   # works — IDE knows it's list[str]
state["made_up_field"]  # type error caught at dev time
```

#### Full state structure

```text
INPUT (set at pipeline entry)
  session_id       str
  resume_text      str
  user_location    str
  target_company   str | None

VALIDATION (set by validator node)
  validation_passed  bool
  missing_fields     list[str]

RAG METADATA (set after chunking + embedding)
  chunks_count     int

AGENT OUTPUTS
  resume_issues    list[str]   ← Agent 1: what's wrong with the resume
  skills_found     list[str]   ← Agent 1: detected skills
  questions        list[dict]  ← Agent 3: interview questions
  salary_range     dict        ← Agent 4: salary + market data
  active_companies list[str]   ← Agent 4: companies hiring now

PIPELINE CONTROL
  current_step     str
  error            str | None

FINAL OUTPUT
  final_report     str | None  ← Agent 5: compiled report
```

#### Why initialise all fields upfront in `create_initial_state()`?

LangGraph passes the state dict to every node. If `resume_issues` doesn't exist yet when Agent 2 tries to read it, you get a `KeyError`. Initialising everything to empty values (`[]`, `{}`, `None`) means every node can safely read any field without guards.

#### How LangGraph nodes update state

Each node (agent) receives the full state and returns a **partial dict** with only the fields it changed:

```python
def resume_analyser_node(state: ResumeState) -> dict:
    # read from state
    text = state["resume_text"]
    # ... run analysis ...
    # return ONLY what changed
    return {
        "resume_issues": ["No quantified achievements", "Missing LinkedIn"],
        "skills_found":  ["Python", "FastAPI", "Docker"],
        "current_step":  "resume_analysed",
    }
```

LangGraph merges this partial dict back into the full state. The unchanged fields stay as-is.

#### AI / LangGraph concept

`ResumeState` is the **single source of truth** for the entire pipeline. This is LangGraph's core design pattern:

```text
          ┌─────────────────────────────┐
          │         ResumeState          │
          │  (all agents read & write)   │
          └─────────────────────────────┘
                ↑      ↑      ↑      ↑
          Agent1  Agent2  Agent3  Agent4
```

Compare to a chain (LangChain): each step passes output to the next as a simple value. LangGraph's shared state means any agent can access any prior result — Agent 4 can read `skills_found` set by Agent 1.

#### Interview Questions

1. **"What is a TypedDict and why does LangGraph use it?"**
   A TypedDict is a dict with declared key types — gives IDE autocomplete and type safety without runtime overhead. LangGraph uses it because state is fundamentally a dict that gets serialized, checkpointed, and passed between nodes.

2. **"How does LangGraph merge state updates?"**
   Each node returns a partial dict. LangGraph shallow-merges it into the existing state. Lists and dicts are replaced (not appended) unless you explicitly use `Annotated[list, operator.add]` as the field type with a reducer.

3. **"What is `str | None` in Python?"**
   Union type — the field can be either a string or None. Equivalent to `Optional[str]` from `typing`. Available since Python 3.10. Used for optional fields like `target_company` and `error`.

4. **"Why `list[dict]` for `questions` instead of a Pydantic model?"**
   State fields need to be JSON-serializable for LangGraph checkpointing. A plain dict is always serializable. A Pydantic model would need custom serialization. For state, plain types win.

5. **"What is `current_step` used for?"**
   Tracks which node last ran. Written to SQLite via `update_session_status()` so the `/status` API can show live progress to the frontend: "validating" → "chunking" → "analysing" → "generating questions" → "completed".

6. **"What is the difference between LangGraph state and LangChain chain output?"**
   LangChain chain: output of step N is the input to step N+1 — linear, one value flows through. LangGraph state: all agents share one dict — any agent can read any field from any prior step. State enables non-linear flows (parallel nodes, conditional edges, loops).

---

### Step 2.2 — Resume Analyser Agent

**File(s) created:** `app/graph/agents/resume_analyser.py`

#### What this step does

The first real **LangGraph agent**. It combines RAG retrieval with the LLM to produce two outputs:

- `skills_found` — list of technical skills detected in the resume
- `resume_issues` — list of 3–7 specific, actionable weaknesses

This is the first node where the entire RAG pipeline (Phase 1) and the LLM come together.

#### The 3-step flow

```text
Step 1: RAG retrieval
  embed_query("skills technologies...")  →  retrieve_chunks(session_id, n=3)
  embed_query("work experience...")       →  retrieve_chunks(session_id, n=3)
  embed_query("education...")             →  retrieve_chunks(session_id, n=3)
  → deduplicate → joined context

Step 2: LLM call
  llama3.1:8b  +  format="json"  +  strict prompt
  → JSON response

Step 3: parse → state update
  skills_found, resume_issues, current_step
```

#### Why 3 separate RAG queries?

A single query like "analyse this resume" returns whatever ChromaDB thinks is most similar — usually a random mix. Three targeted queries (skills / experience / education) each retrieve the most relevant chunks for that area, giving the LLM a **balanced view** of the resume.

This is called **multi-query retrieval** — a standard technique to improve RAG context quality.

#### Why `format="json"` in the Ollama call?

Llama 3.1 is a generative model — by default it outputs free-form text. `format="json"` is Ollama's structured output mode that constrains the model to produce valid JSON. Without this, you'd often get markdown-wrapped JSON (` ```json ... ``` `) or trailing commentary that breaks `json.loads()`.

#### Why `temperature=0.2`?

| Temperature | Effect |
|---|---|
| 0.0 | Deterministic — same input always gives same output |
| 0.2 | Mostly deterministic, slight variation — good for structured tasks |
| 0.7+ | Creative — for storytelling, brainstorming |

For analysis where we want consistent, factual output, low temperature is correct. Resume analysis isn't a creative task.

#### The prompt design

The prompt does 3 things:
1. **Sets a role** — "You are an expert technical resume reviewer"
2. **Provides context** — the retrieved chunks injected as `{context}`
3. **Constrains output** — strict JSON schema, with positive examples ("GOOD examples") and negative examples ("AVOID generic feedback")

The negative examples are critical — without them, the LLM defaults to bland advice like "improve formatting" or "make it more concise".

#### What the agent returns

```python
return {
    "skills_found":  ["Python", "FastAPI", "Docker", ...],
    "resume_issues": ["No quantified achievements in 2nd job", ...],
    "current_step":  "resume_analysed",
}
```

This is a **partial state update**. LangGraph merges it into the full `ResumeState`. The other state fields (e.g., `salary_range`, `questions`) stay untouched until later agents fill them.

#### AI / LangGraph concept

This is a textbook **RAG-powered agent**:

```text
Question  →  Embed  →  Vector search  →  Retrieved chunks
                                              ↓
                                        Build prompt
                                              ↓
                                          LLM call
                                              ↓
                                       Structured output
```

This pattern repeats for every agent in this project — the difference is the queries used and the prompt.

#### Interview Questions

1. **"Walk me through what your Resume Analyser does."**
   It runs 3 RAG queries (skills, experience, education), retrieves top-3 chunks each, deduplicates, builds a single context string, sends it to llama3.1:8b with `format=json`, parses the JSON response, and writes `skills_found` and `resume_issues` back to the LangGraph state.

2. **"Why multiple RAG queries instead of one?"**
   Multi-query retrieval gives balanced coverage. A single query returns whatever's most similar — often a random mix. Three targeted queries guarantee the LLM sees skills chunks, experience chunks, AND education chunks. This dramatically improves the LLM's ability to give balanced feedback.

3. **"What is `format=json` in Ollama?"**
   A structured output mode that forces the model to emit valid JSON. Internally Ollama uses grammar-constrained sampling to reject any token that would break JSON syntax. This is much more reliable than asking the LLM "please respond in JSON" via the prompt alone.

4. **"What's the difference between temperature 0 and 0.2?"**
   Temperature 0 is fully greedy — always picks the highest-probability token. 0.2 introduces a small amount of randomness — still mostly deterministic but allows slight variation for natural-feeling output. For structured analysis we want consistency, so 0–0.2 is the right range.

5. **"What does the agent return and why is it a partial dict?"**
   It returns only the fields it changed: `skills_found`, `resume_issues`, `current_step`. LangGraph automatically merges this into the full state — the unchanged fields (e.g., `target_company`, `chunks_count`) stay as-is. This makes nodes composable and avoids accidentally overwriting other agents' work.

6. **"How would you improve this agent?"**
   Add a self-reflection step where the LLM critiques its own output. Use chain-of-thought prompting ("First list each weakness with evidence, then format as JSON"). Add few-shot examples of high-quality issue lists. Cache embeddings of the 3 standard queries so they're not re-embedded for every resume.

---

### Step 2.3 — Dynamic Router

**File(s) created:** `app/graph/agents/router.py`

#### What this step does

The router is **not a node** that runs work — it's a set of **decision functions** that examine the current `ResumeState` and return the **name** of the next node to execute. LangGraph calls these functions on conditional edges and routes the flow accordingly.

This file contains 4 router functions, one per decision point:

| Function | Called after | Possible next nodes |
|---|---|---|
| `route_after_validation()` | validator node | `analyse` or `end` |
| `route_after_analysis()` | resume analyser | `generate_questions` or `end` |
| `route_after_questions()` | question generator | `fetch_salary`, `compile_report`, or `end` |
| `route_after_salary()` | salary agent | `compile_report` or `end` |

#### Why "dynamic"?

A static graph has hard-coded edges — node A always goes to node B. A dynamic router examines runtime state and picks the next node based on actual data:

- Validation failed? → skip everything, end pipeline early
- No skills detected? → skip the expensive salary agent (web search would be wasteful)
- LLM error in any node? → bail out gracefully

This is what makes LangGraph more powerful than a simple chain — flow control based on data.

#### Why use string constants for route labels?

```python
ROUTE_END        = "end"
ROUTE_ANALYSE    = "analyse"
ROUTE_QUESTIONS  = "generate_questions"
```

The router returns a string that LangGraph maps to a node name in `add_conditional_edges()`. Using constants instead of magic strings prevents typos. If you typo `"generate_questoins"` in the router, it silently routes to nowhere. Constants give one place to change names and IDE autocomplete.

#### How LangGraph uses these routers

In Step 2.7 (graph_builder.py) we'll wire them up like this:

```python
graph.add_conditional_edges(
    "validator",                    # FROM node
    route_after_validation,          # router function
    {                                # mapping: return value → next node
        ROUTE_ANALYSE: "resume_analyser",
        ROUTE_END:     END,
    },
)
```

The router function gets the current state, returns a label, LangGraph looks up the label in the mapping, and executes that node next.

#### The "skip salary if no skills" branch

```python
def route_after_questions(state):
    if not state["skills_found"]:
        return ROUTE_REPORT  # skip salary
    return ROUTE_SALARY
```

The salary agent uses web search to find market rates. If we don't know the candidate's skills (e.g., the resume was unparseable), the search query has nothing meaningful to ask. Skipping saves ~15 seconds and avoids garbage results.

#### Why every router checks `state.get("error")`?

Defensive routing. If any prior agent set `error` in the state (e.g., an LLM call failed), every router short-circuits to `ROUTE_END`. This means **one error stops the pipeline** instead of cascading bad data through the remaining agents.

#### AI / LangGraph concept

This is the **conditional edge** pattern — the core of LangGraph. The graph isn't a fixed pipeline; it's a state machine where transitions depend on state values. Router functions are pure (no side effects, just read state and return a label).

```text
                ┌──────────────┐
                │  validator   │
                └──────┬───────┘
                       │
              route_after_validation(state)
                       │
                ┌──────┴───────┐
              "end"          "analyse"
                ↓                ↓
              END         resume_analyser
```

#### Interview Questions

1. **"What is a conditional edge in LangGraph?"**
   An edge whose target depends on a runtime function. `add_conditional_edges(from_node, router_fn, mapping)` — LangGraph calls `router_fn(state)`, gets a string label, and looks up the next node in `mapping`.

2. **"What's the difference between a node and a router?"**
   A node does work (calls an LLM, queries a DB, transforms data) and returns a partial state update. A router does no work — it reads state and returns a string label naming the next node. Routers are pure functions.

3. **"Why don't routers update state?"**
   Routers should be deterministic and side-effect-free so the graph's flow logic is transparent. If a router needed to update state, that work belongs in a node that runs *before* the router.

4. **"What if a router returns a label that's not in the mapping?"**
   LangGraph raises an error at runtime. This is why we use `ROUTE_*` string constants — typos are caught by the IDE before runtime.

5. **"How does the router help error handling?"**
   Each router checks `state.get("error")` first. If any prior agent set the error field, the router short-circuits to `ROUTE_END`. This stops a single failure from cascading bad data through 4 more LLM calls.

6. **"Could you replace this with a chain instead?"**
   Yes, for the happy path. But chains can't conditionally skip nodes (e.g., skip salary when no skills) without manual `if` statements inside each step. LangGraph routers make these decisions explicit and testable.

---

### Step 2.4 — Question Generator Agent

**File(s) created:** `app/graph/agents/question_generator.py`

#### What this step does

Generates **25 interview questions** total across THREE sub-agents inside one LangGraph node:

| Sub-agent | Count | Focus |
|---|---|---|
| 3a — Technical | 15 | Skill + company-mandatory topics + coding |
| 3b — Project | 5 | Specific verification questions on resume claims |
| 3c — HR / Behavioral | 5 | Tailored to target company's behavioral framework |

#### THE KEY DESIGN RULE — target_company drives everything

**The target company decides the interview style — NOT the resume.**

Example: a candidate with only Python in their resume targets Netflix. Netflix interviews are heavily HLD / LLD / System Design. The agent will STILL generate System Design questions, because that's what the candidate will face on interview day. Resume coverage is secondary.

This is reflected in the prompt's coverage targets:

```text
~ 50% on target_company's standard topics (resume coverage NOT required)
~ 40% grounded in candidate's listed skills, framed in target_company's style
~ 10% market / location supporting questions
```

Each technical question carries a `covered_in_resume: true|false` flag so the candidate knows where to study extra hard.

#### Company-style cheatsheet baked into the prompt

The technical prompt has a reference cheatsheet for major companies:

| Company tier | Pattern asked |
|---|---|
| Netflix | Distributed systems, HLD/LLD, microservices, observability, JVM tuning, fault tolerance |
| Google / Meta | Algorithms, data structures, large-scale system design, complexity |
| Amazon | Algorithms + system design + Leadership Principles overlay |
| Microsoft / Apple | Balanced coding + design + craft / culture-fit |
| Indian product (Flipkart, Razorpay, Swiggy) | DSA, HLD, LLD, India-scale, payment correctness, latency |
| Indian services (Infosys, TCS, Wipro) | Fundamentals, project walkthroughs, framework basics |
| Startups | Ownership, breadth, real production debugging |

Same idea repeats in HR prompt — Amazon → strict Leadership Principles, Google → googliness, etc.

#### Why three sub-agents in ONE node?

Each sub-agent has different prompt, different RAG context, different temperature. Splitting them into 3 LangGraph nodes would require a list reducer (`Annotated[list, operator.add]`) for the `questions` field. Combining them inside one node keeps the graph simpler — graph stays at 6 nodes instead of 8.

The sub-agents run sequentially, but each pass takes ~5–8 seconds locally, so the user sees a single ~25-second "generating questions…" step instead of three separate ones.

#### What's in each generated question

```json
Technical:
{
  "type": "technical", "category": "System Design",
  "question": "...", "difficulty": "medium",
  "expected_topics": [...],
  "code_snippet": "..." | null,
  "covered_in_resume": true | false,
  "company_style_match": "...",
  "market_relevance": "..."
}

Project:
{
  "type": "project", "category": "system design",
  "question": "...", "difficulty": "medium",
  "expected_topics": [...],
  "based_on": "<exact resume line that prompted this question>",
  "company_style_match": "..."
}

HR:
{
  "type": "hr", "category": "leadership",
  "question": "...", "difficulty": "medium",
  "expected_topics": [...],
  "company_style_match": "..."
}
```

#### Phase 3 follow-up — web search enhancement (TO DO)

Currently the agent relies on the LLM's training-time knowledge of company interview patterns. Llama 3.1's cutoff is ~2024 — solid for FAANG and major Indian product companies, stale for niche or newly trending firms.

**When Phase 3 lands**, we'll add web search (`app/tools/web_search.py` via DuckDuckGo) and inject recent results into all three prompts:

```text
Recent {target_company} interview reports (web search):
{web_results}
```

This will be a single 1-line addition per prompt — the rest of the agent stays the same. The TODO is documented inline at the top of `question_generator.py`.

#### AI / LangGraph concept

This step demonstrates **multi-prompt agents** — one node, multiple LLM calls with different prompts and curated RAG contexts. Useful when:

- Different output schemas are needed (technical Q vs HR Q vs project Q)
- Different parts of the same task need different temperature settings
- Different RAG queries make sense for each sub-task

This pattern recurs in Agent 4 (Salary): one call for India market salary, one call for company-specific salary intel.

#### Interview Questions

1. **"Why does target_company drive question selection over the resume?"**
   The candidate needs to be prepped for what they'll FACE in the interview, not just what they've written down. Netflix asks System Design even from candidates who never mentioned it. Resume-grounding is the secondary signal — useful for personalisation, not the primary anchor.

2. **"How is `covered_in_resume` useful?"**
   It tells the candidate which questions are within their comfort zone vs which need extra study. A Netflix candidate sees 8 of their 15 technical questions are flagged `covered_in_resume: false` → they know System Design is the gap to close before interview day.

3. **"Why three sub-agents in one node instead of three nodes?"**
   They all write to the same `questions` list field. Three separate LangGraph nodes would either overwrite each other (default merge behaviour) or require a list reducer (`Annotated[list, operator.add]`). Combining inside one node sidesteps that complexity. Trade-off: lose parallelism, but each call is fast enough that sequential is fine.

4. **"How would web search improve this in Phase 3?"**
   Llama 3.1's training cutoff is ~2024. For company patterns that have shifted recently (e.g., a startup IPO'd, a FAANG-tier introduced a new round), web search results would refresh the LLM's understanding. Same prompt structure, just one additional context block injected.

5. **"How do you ensure the technical prompt doesn't make up skills?"**
   For grounded (resume-based) questions: the prompt says "use the candidate's actual skills". For company-mandatory questions: we EXPLICITLY allow asking about topics NOT in the resume (system design, HLD/LLD), but flag them with `covered_in_resume: false`. The two categories are kept distinct, not blended.

6. **"What is the temperature trade-off here (0.3 vs 0.4)?"**
   Technical (0.3) — wants consistency, same skill should produce similarly-shaped questions. HR (0.4) — wants natural-language variety, same theme like "leadership" shouldn't always read like a template. Tuning per task is a real lever.

---

### Step 2.5 — Salary + Market Intel Agent

**File(s) created:** `app/graph/agents/salary_agent.py`

#### What this step does

Two outputs in one node:

| Sub-agent | Output state field | What |
|---|---|---|
| 4a — Salary | `salary_range` (dict) | Realistic min / max / median + factors + optional company-specific override |
| 4b — Companies | `active_companies` (list[str]) | 8-12 firms currently hiring for this skill + location combo |

#### `salary_range` schema

```json
{
  "currency": "INR",
  "min": 1200000,
  "max": 2400000,
  "median": 1800000,
  "experience_band": "5-7 years",
  "factors": [
    "FastAPI/Python backend roles command 15-20% premium in Bangalore (2026)",
    "AI/LLM-adjacent skills add 10-15% on top of base",
    "..."
  ],
  "company_specific": {
    "Netflix": {
      "min": 4500000,
      "max": 6500000,
      "note": "Senior backend at Netflix India sits well above market median due to global pay parity"
    }
  },
  "disclaimer": "Estimates based on 2024-2026 market data; verify with Glassdoor / levels.fyi / AmbitionBox before negotiating."
}
```

The `company_specific` block is empty `{}` if no `target_company` was provided. When present, it lets the candidate see how their target's pay differs from market median.

#### `active_companies` format

Each entry is a single string: `"<Company> (<City>) — <Why they match>"`. Example:

```text
Razorpay (Bangalore) — Hiring senior Python/FastAPI backend; matches your stack
Swiggy (Bangalore) — Active backend hiring for payments platform; Python + Kafka
PhonePe (Bangalore) — UPI scaling team hiring senior Java/Kotlin engineers
```

The "why they match" must reference SPECIFIC skills, not generic phrasing.

#### The KNOWN LIMITATION — and why it's the strongest case for Phase 3

This agent currently uses **Llama 3.1's training-time knowledge** (cutoff ~2024). Two problems:

1. **Salary numbers go stale fast.** A 2024 number is already 1-2 years stale — Indian tech salaries shifted 8-15% in that window. Stale data here misleads users in negotiations.
2. **"Currently hiring" is meaningless without live data.** A company that's hiring today may have frozen hiring tomorrow. Without a live signal, the list is at best "companies that historically hire for this profile".

**Phase 3 mitigation** — replace LLM-only generation with live DuckDuckGo lookups:

```text
embed_query → DuckDuckGo
   "Python FastAPI backend salary Bangalore 2026"
   "{target_company} backend engineer salary site:levels.fyi"
   "Razorpay careers backend Python 2026"
→ inject results into the salary + companies prompts
```

The TODO is commented at the top of `salary_agent.py`. Out of all Phase 3 integrations, this is the highest-priority one.

#### Why temperature 0.3 (salary) vs 0.4 (companies)?

| Output | Temp | Reason |
|---|---|---|
| Salary numbers | 0.3 | Want consistency — same profile should produce similar numbers across runs |
| Company list | 0.4 | Want variety — same skills shouldn't always produce identical 10-company list |

Salary is a precision task, company list is a recall task. Different temperatures match the goal.

#### Why two LLM calls instead of one?

A single combined prompt would dilute focus. Salary estimation needs the LLM to think about market rates, skill premiums, company tier, location effect. Company recall needs the LLM to think about who's hiring + skill match. Different mental models → cleaner outputs from separate calls.

This is the same pattern as Agent 3 (Question Generator): one node, multiple LLM calls.

#### AI / LangGraph concept

**Multi-output agents** — when an agent produces two distinct artifacts, give each its own LLM call with its own prompt. State updates can include multiple fields:

```python
return {
    "salary_range":     {...},
    "active_companies": [...],
    "current_step":     "salary_analysed",
}
```

LangGraph merges the dict — both fields update in one node transition.

#### Interview Questions

1. **"Why isn't the salary agent using web search?"**
   It will — in Phase 3. Phase 2 deliberately builds the LangGraph mechanics first, then Phase 3 layers in DuckDuckGo as a shared tool used by both this agent and the Question Generator. The TODO is documented at the top of `salary_agent.py`.

2. **"How accurate is LLM-generated salary data?"**
   For broad strokes (band, currency, factors that drive comp): reasonably accurate based on 2024 training data. For exact numbers in 2026: stale and shouldn't drive negotiation decisions. The `disclaimer` field in the output makes this explicit to the user.

3. **"Why include `company_specific` only when target_company is set?"**
   A null/empty target means the user is exploring the broader market. Inventing a specific company override would either be arbitrary or misleading. Conditional schema fields are a clean way to handle optional state inputs.

4. **"Why is the active_companies list a `list[str]` instead of `list[dict]`?"**
   Initial state schema chose `list[str]` for simplicity. Each string carries the structure inline ("Company (City) — Why"). For a richer UI in Phase 5, we may upgrade to `list[dict]`. Trade-off: simpler to render now vs flexibility later.

5. **"How would you defend the salary range to a sceptical user?"**
   Show the `factors` array — each factor is a specific market signal (skill premium, location effect, India vs global parity). Factors are auditable; raw numbers aren't. Plus the `disclaimer` directs the user to triangulate with Glassdoor / levels.fyi / AmbitionBox.

6. **"What if the target_company has no global pay parity? (e.g., service company)"**
   The LLM should still produce a `company_specific` block but with realistic service-tier numbers (e.g., 8L-15L for senior Java engineer at a service co). The prompt rule says "include {target_company} with realistic min/max" — the realism comes from the LLM understanding company tier from name alone.

---

### Step 2.6 — Report Compiler Agent

**File(s) created:** `app/graph/agents/report_compiler.py`

#### What this step does

Combines all prior agents' outputs into a single polished **markdown report** stored in `state["final_report"]`. This is the artifact the frontend renders for the user.

The report has 6 sections:

```text
1. Executive Summary             (LLM-generated prose)
2. Resume Analysis               (skills + issues from Agent 1)
3. Interview Preparation         (25 questions from Agent 3, grouped by type)
4. Salary Insights               (range + factors + company-specific from Agent 4a)
5. Active Hiring Companies       (list from Agent 4b)
6. Action Plan                   (LLM-generated 5 prioritised steps)
```

#### Key design choice — TEMPLATE-BASED with only 2 LLM calls

The naïve approach is to feed everything into one big LLM prompt and ask it to "write a report". That's wasteful and unreliable:

- We already have **structured data** from earlier agents — re-formatting via LLM risks hallucinating numbers
- LLM-generated tables can drift from their source data
- A single big prompt is slow (one big call vs many small focused ones)

Instead, this agent uses LLM only for the parts that genuinely need prose:

| Part | Method | Why |
|---|---|---|
| Executive Summary | LLM (4-6 sentences) | Sets tone — needs narrative cohesion |
| Action Plan | LLM (5 prioritised items, JSON) | Needs prioritisation + company-specific context |
| Skills, issues, questions, salary, companies | Template formatting | Data already structured — LLM would only risk distortion |

This pattern is called **structured + free-form hybrid generation**. Use templates for known structure, LLM for genuine creative work.

#### The two LLM calls

```python
1) Summary  → temperature 0.4, free-form prose
2) Actions  → temperature 0.3, format="json"
```

Different temperature per task — the summary benefits from natural variation, the action plan needs precision.

#### Indian-style number formatting

Salary numbers are formatted with the Indian lakh/crore grouping: `1500000 → 15,00,000` (not Western `1,500,000`). This matters because the audience is primarily Indian candidates — `15,00,000` reads as "fifteen lakh" which is how Indians actually discuss salary.

```python
def _format_inr(amount):
    # 1500000 → "15,00,000"
    # 12500   → "12,500"
```

#### Defensive formatting throughout

Every helper guards against missing data:

```python
if not salary_range or salary_range.get("min") is None:
    return ["_Salary range could not be estimated._"]

if not questions:
    parts.append("_None generated._")
```

If any earlier agent failed silently, the report still assembles — just with placeholder text in the missing section. The pipeline doesn't crash because of one bad LLM response upstream.

#### Question rendering — rich context per question

Each question shows:

- Difficulty badge `[EASY/MEDIUM/HARD]`
- Category
- Expected topics
- Code snippet (only if present)
- Source resume line (for project questions, via `based_on`)
- Why this matches the company style (`company_style_match`)
- Market context (`market_relevance`)
- Warning if `covered_in_resume: false` — "study extra hard"

This gives the candidate a complete prep package per question, not just the question text.

#### AI / LangGraph concept

**Final reducer node** — the last node in a multi-agent pipeline that consolidates everything into a single deliverable. In LangGraph terms, this is the agent that converts shared state into output the user actually consumes.

```text
ResumeState (rich, structured)
    ↓
report_compiler_node()
    ↓
final_report (single markdown string)
    ↓
user / frontend
```

After this node, the pipeline ends — `current_step = "report_compiled"`. The graph builder (Step 2.7) wires this as the terminal node before `END`.

#### Interview Questions

1. **"Why mix templates and LLM instead of using LLM for the whole report?"**
   We already have structured data from earlier agents (skill list, salary numbers, question objects). Asking the LLM to re-format them risks hallucination — it might "tidy up" a salary number or paraphrase a question. Templates preserve the source data exactly. LLM is reserved for the parts that need prose: tone-setting summary and prioritised action plan.

2. **"Why use Indian-style number formatting (15,00,000)?"**
   Audience matters. The user types "Bangalore" and expects to see Indian-style salary. Showing `1,500,000` reads as "one million five hundred thousand" — Western framing — and slows comprehension. Numbers in the audience's native format reduce cognitive friction.

3. **"How does the report stay assembled if Agent 3 fails?"**
   Every section has a fallback like `"_None generated._"`. The template iterates over an empty list and produces a placeholder, not an exception. The report's structure is preserved even if one upstream node returned bad data — degraded output beats a broken report.

4. **"Why two separate LLM calls instead of one combined?"**
   The summary needs free-form prose (`format=text`, temperature 0.4). The action plan needs strict JSON (`format=json`, temperature 0.3). Combining would force one shared format/temperature and dilute the focus of each. Two small calls cost about the same time as one big one with local LLMs.

5. **"What does `covered_in_resume = false` mean in the report?"**
   It's a flag attached to technical questions where the topic was added because of the target company's interview pattern (e.g., System Design for Netflix) but is NOT something the candidate's resume mentions. The report renders a "study extra hard" note next to those questions so the candidate prioritises them.

6. **"Why is `final_report` a markdown string instead of structured data?"**
   The frontend renders markdown directly. Storing a string keeps the frontend dumb (it doesn't need to know all the field shapes). Trade-off: harder to introspect later. For this project's scope (single report → single render), markdown wins.

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
| 1.8 Vector Store | ✅ Done | `app/rag/vector_store.py` |
| 2.1 State design | ✅ Done | `app/graph/state.py` |
| 2.2 Resume Analyser Agent | ✅ Done | `app/graph/agents/resume_analyser.py` |
| 2.3 Dynamic Router | ✅ Done | `app/graph/agents/router.py` |
| 2.4 Question Generator | ✅ Done | `app/graph/agents/question_generator.py` |
| 2.5 Salary Agent | ✅ Done | `app/graph/agents/salary_agent.py` |
| 2.6 Report Compiler | ✅ Done | `app/graph/agents/report_compiler.py` |
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
