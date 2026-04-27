# ResumeIQ — AI-Powered Resume Analyser

An AI backend project built with FastAPI, LangGraph, and local LLMs (Ollama).
Analyses resumes, generates interview questions, shows salary ranges, and finds active hiring companies.

---

## Tech Stack

| Component | Technology |
| --------- | ---------- |
| LLM | Ollama + Llama 3.1 8B (local) |
| Embeddings | Ollama nomic-embed-text (local) |
| Vector DB | ChromaDB (local) |
| Web Search | DuckDuckGo (free, no API key) |
| PDF/DOCX Parse | PyMuPDF + python-docx |
| Backend | FastAPI (Python) |
| AI Orchestration | LangGraph |
| Metadata DB | SQLite |
| Frontend | HTML/CSS |

---

## Project Setup

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) installed

### 1. Clone / open the project

```bash
cd d:/AI/Project/resume-ai-agent
```

### 2. Create virtual environment

```bash
python -m venv venv
```

### 3. Activate virtual environment

```bash
# Windows
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Pull Ollama models (one time)

```bash
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

### 6. Run the server (after full setup)

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open browser at: `http://localhost:8000`

---

## Folder Structure

```text
resume-ai-agent/
├── app/
│   ├── main.py                      # FastAPI app entry point
│   ├── api/
│   │   └── routes.py                # All API endpoints
│   ├── graph/
│   │   ├── state.py                 # LangGraph ResumeState
│   │   ├── graph_builder.py         # Wires all agents together
│   │   └── agents/
│   │       ├── resume_analyser.py   # Agent 1 — RAG-powered analysis
│   │       ├── router.py            # Agent 2 — routing logic
│   │       ├── question_generator.py# Agent 3a + 3b
│   │       ├── salary_agent.py      # Agent 4 — salary + market intel
│   │       └── report_compiler.py   # Agent 5 — final report
│   ├── rag/
│   │   ├── pdf_parser.py            # PyMuPDF PDF parser
│   │   ├── docx_parser.py           # python-docx DOCX parser
│   │   ├── validator.py             # Missing fields check
│   │   ├── chunker.py               # Smart sectional chunker
│   │   ├── embedder.py              # Ollama embeddings
│   │   └── vector_store.py          # ChromaDB operations
│   ├── tools/
│   │   ├── web_search.py            # DuckDuckGo search tool
│   │   └── rag_retriever.py         # ChromaDB retriever tool
│   ├── db/
│   │   └── sqlite_client.py         # All SQLite operations
│   └── models/
│       └── schemas.py               # Pydantic request/response models
├── frontend/
│   └── index.html                   # UI
├── data/
│   └── chroma_db/                   # ChromaDB local storage
├── venv/                            # Virtual environment (not committed)
├── resumeiq.db                      # SQLite DB file (auto-created)
├── .env                             # Environment variables
└── requirements.txt
```

---

## Build Progress

### Phase 1 — RAG Pipeline

- [x] Step 1.1 — Project setup, folder structure, requirements.txt, venv
- [x] Step 1.2 — SQLite setup (3 tables)
- [x] Step 1.3 — PDF parser (PyMuPDF)
- [x] Step 1.4 — DOCX parser (python-docx)
- [x] Step 1.5 — Validator (missing fields check)
- [x] Step 1.6 — Smart sectional chunker
- [x] Step 1.7 — Ollama embeddings setup
- [x] Step 1.8 — ChromaDB setup + store + retrieve

### Phase 2 — LangGraph Agents

- [x] Step 2.1 — State design (state.py)
- [ ] Step 2.2 — Resume Analyser Agent
- [ ] Step 2.3 — Dynamic Router
- [ ] Step 2.4 — Question Generator Agent
- [ ] Step 2.5 — Salary + Market Intel Agent
- [ ] Step 2.6 — Report Compiler Agent
- [ ] Step 2.7 — Graph Builder

### Phase 3 — Web Search Integration

- [ ] Step 3.1 — DuckDuckGo tool setup
- [ ] Step 3.2 — Integrate in Company Q Agent
- [ ] Step 3.3 — Integrate in Salary Agent
- [ ] Step 3.4 — End to end test

### Phase 4 — FastAPI Backend

- [ ] Step 4.1 — main.py setup
- [ ] Step 4.2 — /upload endpoint
- [ ] Step 4.3 — /analyse endpoint
- [ ] Step 4.4 — /chat endpoint
- [ ] Step 4.5 — /status endpoint
- [ ] Step 4.6 — Streaming response

### Phase 5 — Frontend + Final Polish

- [ ] Step 5.1 — Connect HTML frontend to FastAPI
- [ ] Step 5.2 — Pipeline status bar
- [ ] Step 5.3 — Chat interface
- [ ] Step 5.4 — Error handling
- [ ] Step 5.5 — Architecture diagram

---

## Learning Notes

Step-by-step documentation with explanations, concepts, and interview questions:
**[docs/learning_notes.md](docs/learning_notes.md)**

---

## Environment Variables (.env)

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_LLM_MODEL=llama3.1:8b
OLLAMA_EMBED_MODEL=nomic-embed-text
CHROMA_PERSIST_DIR=./data/chroma_db
SQLITE_DB_PATH=./resumeiq.db
HOST=0.0.0.0
PORT=8000
```

---

## Author

Built by a Java developer (5.5 YOE) learning Python + AI Engineering.
Project purpose: Showcase AI Backend Engineering skills for job interviews.
