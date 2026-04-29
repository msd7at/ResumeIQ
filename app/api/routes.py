"""
API routes for ResumeIQ.

This file holds the FastAPI APIRouter that app/main.py mounts under /api.

Endpoints (built incrementally across Steps 4.2 - 4.6):
    POST /api/upload                — accept resume file, parse, return session_id   (Step 4.2 ✓)
    POST /api/analyse               — kick off LangGraph analysis for a session      (Step 4.3 ✓)
    POST /api/chat                  — follow-up chat about an analysed resume        (Step 4.4)
    GET  /api/status/{session_id}   — current pipeline step                          (Step 4.5)
"""

import json
import os
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from ollama import Client
from pydantic import BaseModel, Field

from app.db.sqlite_client import (
    add_chat_message,
    create_session,
    get_analysis,
    get_chat_history,
    get_session,
    save_analysis,
    update_session_status,
)
from app.graph.graph_builder import get_graph
from app.graph.state import create_initial_state
from app.rag.docx_parser import parse_docx
from app.rag.embedder import embed_query
from app.rag.pdf_parser import parse_pdf
from app.rag.vector_store import retrieve_chunks

load_dotenv()

_OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
_LLM_MODEL       = os.getenv("OLLAMA_LLM_MODEL", "llama3.1:8b")

# /chat tuning knobs
_CHAT_HISTORY_LIMIT = 10   # last N prior messages forwarded to the LLM
_CHAT_RAG_CHUNKS    = 4    # resume chunks retrieved per chat turn
_CHAT_MIN_MSG_LEN   = 1
_CHAT_MAX_MSG_LEN   = 2000


router = APIRouter()


# ──────────────────────────────────────────────────────────────────
#  Local storage for uploaded files + parsed text
#  Layout under project root:
#    data/uploads/{session_id}.pdf|docx   ← original file
#    data/sessions/{session_id}.txt        ← parsed plain text
# ──────────────────────────────────────────────────────────────────

_UPLOAD_DIR = Path("data/uploads")
_TEXT_DIR   = Path("data/sessions")
_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
_TEXT_DIR.mkdir(parents=True, exist_ok=True)

_ACCEPTED_EXT = {".pdf", ".docx"}
_MAX_BYTES    = 5 * 1024 * 1024   # 5 MB cap on uploads


def get_session_text(session_id: str) -> str:
    """
    Read the parsed text for a session — used by /analyse and /chat in later steps.
    Raises 404 if no text file exists for the given session.
    """
    text_path = _TEXT_DIR / f"{session_id}.txt"
    if not text_path.exists():
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    return text_path.read_text(encoding="utf-8")


# ──────────────────────────────────────────────────────────────────
#  Pydantic response models
# ──────────────────────────────────────────────────────────────────


class UploadResponse(BaseModel):
    session_id:   str
    filename:     str
    text_length:  int
    status:       str
    next_step:    str


class AnalyseRequest(BaseModel):
    session_id: str


class AnalyseResponse(BaseModel):
    session_id:        str
    status:            str            # 'analysed' | 'validation_failed'
    validation_passed: bool
    missing_fields:    list[str]
    chunks_count:      int
    role_type:         str
    role_description:  str
    skills_found:      list[str]
    resume_issues:     list[str]
    questions:         list[dict]
    salary_range:      dict
    active_companies:  list[str]
    final_report:      str | None
    elapsed_seconds:   float


class ChatRequest(BaseModel):
    session_id: str
    message:    str = Field(
        ...,
        min_length=_CHAT_MIN_MSG_LEN,
        max_length=_CHAT_MAX_MSG_LEN,
        description="The user's chat message — bounded to keep prompt size predictable.",
    )


class ChatResponse(BaseModel):
    session_id:        str
    user_message:      str
    assistant_message: str
    elapsed_seconds:   float
    history_count:     int   # total chat_history rows for this session AFTER this turn


class StatusResponse(BaseModel):
    session_id:     str
    status:         str
    updated_at:     str
    missing_fields: list[str]
    chunks_count:   int
    message:        str


# ──────────────────────────────────────────────────────────────────
#  Routes
# ──────────────────────────────────────────────────────────────────


@router.get("/")
def api_root():
    """Lists the API surface — useful for `curl /api` to see what's available."""
    return {
        "service":   "ResumeIQ API",
        "endpoints": [
            "POST /api/upload                — upload resume file, get session_id",
            "POST /api/analyse              — run full pipeline, blocks until done",
            "POST /api/analyse/stream       — same pipeline streamed as SSE events",
            "POST /api/chat                 — multi-turn chat about an analysed resume",
            "GET  /api/status/{session_id}  — poll current pipeline status",
        ],
    }


@router.post("/upload", response_model=UploadResponse)
async def upload_resume(
    file:           UploadFile = File(...),
    user_location:  str        = Form(...),
    target_company: str | None = Form(None),
):
    """
    Accept a resume file (.pdf or .docx), parse to plain text,
    create a SQLite session row, and persist the parsed text on disk
    so the subsequent /analyse call can read it.
    """
    # ── 1. Validate file extension ────────────────────────────────
    filename = file.filename or "resume"
    ext      = Path(filename).suffix.lower()
    if ext not in _ACCEPTED_EXT:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Accepted: {sorted(_ACCEPTED_EXT)}",
        )

    # ── 2. Read + size-check the upload ───────────────────────────
    content = await file.read()
    if len(content) > _MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(content)} bytes). Max allowed: {_MAX_BYTES} bytes (5 MB).",
        )
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # ── 3. Generate session ID + persist the file ─────────────────
    session_id = f"sess_{uuid.uuid4().hex[:10]}"
    upload_path = _UPLOAD_DIR / f"{session_id}{ext}"
    upload_path.write_bytes(content)

    # ── 4. Parse to plain text ────────────────────────────────────
    try:
        if ext == ".pdf":
            text = parse_pdf(str(upload_path))
        else:
            text = parse_docx(str(upload_path))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse {ext}: {e}")

    if not text or not text.strip():
        raise HTTPException(
            status_code=400,
            detail="No text could be extracted from the file. "
                   "If this is a scanned PDF, OCR is not yet supported.",
        )

    # ── 5. Persist parsed text for /analyse to read later ─────────
    text_path = _TEXT_DIR / f"{session_id}.txt"
    text_path.write_text(text, encoding="utf-8")

    # ── 6. Create the session row in SQLite ───────────────────────
    create_session(
        session_id=session_id,
        filename=filename,
        user_location=user_location,
        target_company=target_company,
    )

    return UploadResponse(
        session_id=session_id,
        filename=filename,
        text_length=len(text),
        status="uploaded",
        next_step=f"POST /api/analyse with session_id={session_id}",
    )


@router.post("/analyse", response_model=AnalyseResponse)
def analyse_resume(req: AnalyseRequest):
    """
    Run the LangGraph pipeline on a previously uploaded session.

    Synchronous: this call blocks for ~30-90s while the 6-node graph executes
    (validator → embedding → resume_analyser → question_generator →
     salary_agent → report_compiler). Streaming/background variants land in
    Step 4.6.

    Defined as `def` (not `async def`) so FastAPI runs it in its threadpool —
    keeps the event loop free while the long sync graph.invoke() runs.
    """
    session_id = req.session_id

    # ── 1. Verify the session exists in SQLite ────────────────────
    session_row = get_session(session_id)
    if session_row is None:
        raise HTTPException(
            status_code=404,
            detail=f"Session {session_id} not found. Upload a resume first via POST /api/upload.",
        )

    # ── 2. Load the parsed resume text from disk ──────────────────
    resume_text = get_session_text(session_id)   # raises 404 if file missing

    # ── 3. Build the initial graph state ──────────────────────────
    state = create_initial_state(
        session_id=session_id,
        resume_text=resume_text,
        user_location=session_row["user_location"],
        target_company=session_row["target_company"],
    )

    # ── 4. Invoke the graph (long-running) ────────────────────────
    update_session_status(session_id, status="analysing")
    t0 = time.time()
    try:
        final_state = get_graph().invoke(state)
    except Exception as e:
        update_session_status(session_id, status="error")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")
    elapsed = time.time() - t0

    # ── 5. Validation-failed branch — graph routes to END early ──
    if not final_state.get("validation_passed"):
        missing = final_state.get("missing_fields", [])
        update_session_status(
            session_id,
            status="validation_failed",
            missing_fields=missing,
        )
        return AnalyseResponse(
            session_id=session_id,
            status="validation_failed",
            validation_passed=False,
            missing_fields=missing,
            chunks_count=0,
            role_type="",
            role_description="",
            skills_found=[],
            resume_issues=[],
            questions=[],
            salary_range={},
            active_companies=[],
            final_report=None,
            elapsed_seconds=round(elapsed, 2),
        )

    # ── 6. Mid-pipeline error captured in state.error ─────────────
    if final_state.get("error"):
        update_session_status(session_id, status="error")
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline error: {final_state['error']}",
        )

    # ── 7. Persist analysis_results + flip session to 'analysed' ─
    save_analysis(
        session_id=session_id,
        resume_issues=final_state.get("resume_issues", []),
        skills_found=final_state.get("skills_found", []),
        questions=final_state.get("questions", []),
        salary_range=final_state.get("salary_range", {}),
        active_companies=final_state.get("active_companies", []),
    )
    update_session_status(
        session_id,
        status="analysed",
        chunks_count=final_state.get("chunks_count", 0),
    )

    return AnalyseResponse(
        session_id=session_id,
        status="analysed",
        validation_passed=True,
        missing_fields=[],
        chunks_count=final_state.get("chunks_count", 0),
        role_type=final_state.get("role_type", ""),
        role_description=final_state.get("role_description", ""),
        skills_found=final_state.get("skills_found", []),
        resume_issues=final_state.get("resume_issues", []),
        questions=final_state.get("questions", []),
        salary_range=final_state.get("salary_range", {}),
        active_companies=final_state.get("active_companies", []),
        final_report=final_state.get("final_report"),
        elapsed_seconds=round(elapsed, 2),
    )


# ──────────────────────────────────────────────────────────────────
#  /chat helpers — kept module-private so the route stays readable
# ──────────────────────────────────────────────────────────────────


_CHAT_SYSTEM_PROMPT = """You are ResumeIQ, a focused career coach. The candidate has already had their resume
analysed by an automated pipeline. Help them act on the analysis: clarify resume issues,
rehearse interview answers, refine bullet points, discuss salary expectations, and plan
their job search.

Ground every answer in the candidate's actual resume content and the prior analysis below.
If the user asks something the resume does not contain, say so plainly — do not invent
experience, projects, or numbers.

═══ RESUME CONTEXT (retrieved for this question) ═══
{resume_context}

═══ PRIOR ANALYSIS ═══
Role type           : {role_type}
Role description    : {role_description}
Target company      : {target_company}
Location            : {location}
Skills detected     : {skills}
Resume issues found : {issues}
Salary range        : {salary_summary}

═══ STYLE ═══
Concise. Bullet points where helpful. Reference specific resume lines / sections when
giving advice. No filler greetings. No follow-up questions unless genuinely needed."""


def _retrieve_resume_context(session_id: str, query: str) -> str:
    """Embed the user's question and pull the top resume chunks for grounding."""
    try:
        q_vec  = embed_query(query)
        chunks = retrieve_chunks(q_vec, session_id=session_id, n_results=_CHAT_RAG_CHUNKS)
    except Exception:
        return "_(retrieval unavailable)_"

    if not chunks:
        return "_(no relevant resume sections found)_"

    return "\n\n---\n\n".join(c["text"] for c in chunks)


def _summarise_salary(salary_range: dict) -> str:
    if not salary_range or salary_range.get("min") is None:
        return "not estimated"
    currency = salary_range.get("currency", "INR")
    return f"{salary_range.get('min')}-{salary_range.get('max')} {currency}"


def _build_chat_messages(
    *,
    session_row: dict,
    analysis: dict,
    history: list[dict],
    user_message: str,
) -> list[dict]:
    """Construct the messages array passed to ollama.Client.chat()."""
    skills = analysis.get("skills_found", []) or []
    issues = analysis.get("resume_issues", []) or []

    system_content = _CHAT_SYSTEM_PROMPT.format(
        resume_context   = _retrieve_resume_context(session_row["id"], user_message),
        role_type        = analysis.get("role_type") or "general professional",
        role_description = analysis.get("role_description") or "—",
        target_company   = session_row.get("target_company") or "any company",
        location         = session_row.get("user_location") or "—",
        skills           = ", ".join(skills[:15]) if skills else "—",
        issues           = "; ".join(issues[:5]) if issues else "—",
        salary_summary   = _summarise_salary(analysis.get("salary_range", {}) or {}),
    )

    messages: list[dict] = [{"role": "system", "content": system_content}]

    # Forward only the tail of history — older turns add cost without much value
    for row in history[-_CHAT_HISTORY_LIMIT:]:
        role = row.get("role")
        if role in ("user", "assistant"):
            messages.append({"role": role, "content": row["message"]})

    messages.append({"role": "user", "content": user_message})
    return messages


# ──────────────────────────────────────────────────────────────────
#  POST /api/chat — Step 4.4
# ──────────────────────────────────────────────────────────────────


@router.post("/chat", response_model=ChatResponse)
def chat_with_resume(req: ChatRequest):
    """
    Multi-turn chat about an analysed resume.

    Grounding sources (combined into the system prompt every turn):
      1. RAG retrieval over the candidate's resume chunks (query-conditioned)
      2. Latest analysis_results row (skills, issues, salary, target company)
      3. Tail of chat_history (last N messages) for conversational continuity

    Defined as `def` (not `async def`) — same reasoning as /analyse: the LLM call
    is synchronous and FastAPI's threadpool keeps the event loop responsive.

    Status codes:
      200 — success, returns the assistant message and persists both turns.
      404 — session_id unknown.
      409 — session has not been analysed yet (chat needs grounding data to exist).
    """
    session_id   = req.session_id
    user_message = req.message.strip()

    if not user_message:
        raise HTTPException(status_code=400, detail="message must not be empty.")

    # ── 1. Verify session exists ──────────────────────────────────
    session_row = get_session(session_id)
    if session_row is None:
        raise HTTPException(
            status_code=404,
            detail=f"Session {session_id} not found. Upload a resume first via POST /api/upload.",
        )

    # ── 2. Require an analysis to ground the chat ─────────────────
    if session_row["status"] != "analysed":
        raise HTTPException(
            status_code=409,
            detail=(
                f"Session {session_id} is in status '{session_row['status']}'. "
                "Run POST /api/analyse before chatting."
            ),
        )

    analysis = get_analysis(session_id)
    if analysis is None:
        raise HTTPException(
            status_code=409,
            detail=f"No analysis found for session {session_id}. Run POST /api/analyse first.",
        )

    # ── 3. Build the messages array (system + history tail + user) ─
    history  = get_chat_history(session_id)
    messages = _build_chat_messages(
        session_row=session_row,
        analysis=analysis,
        history=history,
        user_message=user_message,
    )

    # ── 4. Call the LLM ───────────────────────────────────────────
    t0 = time.time()
    try:
        client   = Client(host=_OLLAMA_BASE_URL)
        response = client.chat(
            model=_LLM_MODEL,
            messages=messages,
            options={"temperature": 0.4},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {e}")
    elapsed = time.time() - t0

    assistant_message = (response.get("message") or {}).get("content", "").strip()
    if not assistant_message:
        raise HTTPException(status_code=502, detail="LLM returned an empty response.")

    # ── 5. Persist BOTH turns so the next call sees this exchange ─
    add_chat_message(session_id, role="user",      message=user_message)
    add_chat_message(session_id, role="assistant", message=assistant_message)

    return ChatResponse(
        session_id=session_id,
        user_message=user_message,
        assistant_message=assistant_message,
        elapsed_seconds=round(elapsed, 2),
        history_count=len(history) + 2,
    )


# ──────────────────────────────────────────────────────────────────
#  GET /api/status/{session_id} — Step 4.5
# ──────────────────────────────────────────────────────────────────

_STATUS_MESSAGES: dict[str, str] = {
    "uploaded":          "Resume uploaded. Run POST /api/analyse or POST /api/analyse/stream to start analysis.",
    "analysing":         "Analysis is running. Poll again in a few seconds, or use POST /api/analyse/stream for real-time updates.",
    "analysed":          "Analysis complete. Use POST /api/chat to ask questions about the resume.",
    "validation_failed": "Resume failed validation — check missing_fields for what to fix, then re-upload.",
    "error":             "An error occurred during analysis. Try uploading the resume again.",
}


@router.get("/status/{session_id}", response_model=StatusResponse)
async def get_status(session_id: str):
    """
    Return the current processing status of a resume session.

    Useful for polling progress during a long /analyse call, or confirming
    a session is ready before calling /chat.

    Status values:
        uploaded          — file parsed; analysis not yet started
        analysing         — LangGraph pipeline is running
        analysed          — pipeline complete; chat is now available
        validation_failed — resume lacked required fields; see missing_fields
        error             — pipeline crashed; re-upload to retry
    """
    session_row = get_session(session_id)
    if session_row is None:
        raise HTTPException(
            status_code=404,
            detail=f"Session {session_id!r} not found.",
        )

    return StatusResponse(
        session_id=session_id,
        status=session_row["status"],
        updated_at=session_row["updated_at"],
        missing_fields=session_row["missing_fields"],
        chunks_count=session_row["chunks_count"],
        message=_STATUS_MESSAGES.get(session_row["status"], "Unknown status."),
    )


# ──────────────────────────────────────────────────────────────────
#  POST /api/analyse/stream — Step 4.6  (Server-Sent Events)
# ──────────────────────────────────────────────────────────────────

_NODE_LABELS: dict[str, tuple[str, int]] = {
    "validator":          ("Validating resume structure",        1),
    "embedding":          ("Chunking & embedding resume text",   2),
    "resume_analyser":    ("Analysing skills, role & issues",    3),
    "question_generator": ("Generating interview questions",     4),
    "salary_agent":       ("Researching salary & hiring market", 5),
    "report_compiler":    ("Compiling final report",             6),
}

_TOTAL_STEPS = len(_NODE_LABELS)


def _sse(payload: dict) -> str:
    """Encode a dict as a Server-Sent Event data line."""
    return f"data: {json.dumps(payload)}\n\n"


def _node_partial(node_name: str, updates: dict) -> dict:
    """Return only the fields worth surfacing for each node's SSE event."""
    if node_name == "validator":
        return {
            "validation_passed": updates.get("validation_passed"),
            "missing_fields":    updates.get("missing_fields", []),
        }
    if node_name == "embedding":
        return {"chunks_count": updates.get("chunks_count", 0)}
    if node_name == "resume_analyser":
        return {
            "role_type":        updates.get("role_type", ""),
            "role_description": updates.get("role_description", ""),
            "skills_found":     updates.get("skills_found", []),
            "resume_issues":    updates.get("resume_issues", []),
        }
    if node_name == "question_generator":
        return {"questions_count": len(updates.get("questions", []))}
    if node_name == "salary_agent":
        return {
            "salary_range":     updates.get("salary_range", {}),
            "active_companies": updates.get("active_companies", []),
        }
    if node_name == "report_compiler":
        report = updates.get("final_report") or ""
        return {"final_report_preview": report[:300] + "…" if len(report) > 300 else report}
    return {}


@router.post("/analyse/stream")
def stream_analyse(req: AnalyseRequest):
    """
    Stream the LangGraph analysis pipeline as Server-Sent Events (SSE).

    Each event has the shape:
        data: {"event": "<type>", ...fields}

    Event types:
        "started"       — pipeline has begun; total_steps tells the client how many nodes to expect
        "node_complete" — one graph node finished; includes step number, label, and partial results
        "done"          — pipeline finished; includes the full analysis payload (same shape as /analyse)
        "error"         — fatal error; message field describes the failure

    The connection stays open while the graph executes (~30-90 s for a typical resume).
    Clients should consume events until "done" or "error", then close the connection.

    Defined as `def` (not `async def`) so FastAPI runs it in the threadpool — the synchronous
    graph.stream() iterator runs without blocking the event loop.
    """
    session_id = req.session_id

    session_row = get_session(session_id)
    if session_row is None:
        raise HTTPException(
            status_code=404,
            detail=f"Session {session_id!r} not found. Upload a resume first via POST /api/upload.",
        )

    resume_text = get_session_text(session_id)   # raises 404 if file missing

    state = create_initial_state(
        session_id=session_id,
        resume_text=resume_text,
        user_location=session_row["user_location"],
        target_company=session_row["target_company"],
    )

    def event_gen():
        update_session_status(session_id, status="analysing")
        t0 = time.time()

        yield _sse({"event": "started", "session_id": session_id, "total_steps": _TOTAL_STEPS})

        accumulated = dict(state)   # mirrors the full graph state as nodes run

        try:
            for chunk in get_graph().stream(state, stream_mode="updates"):
                for node_name, updates in chunk.items():
                    accumulated.update(updates)
                    label, step_num = _NODE_LABELS.get(node_name, (node_name, 0))
                    yield _sse({
                        "event":       "node_complete",
                        "node":        node_name,
                        "label":       label,
                        "step":        step_num,
                        "total_steps": _TOTAL_STEPS,
                        "data":        _node_partial(node_name, updates),
                    })
        except Exception as e:
            update_session_status(session_id, status="error")
            yield _sse({"event": "error", "message": str(e)})
            return

        elapsed = time.time() - t0

        # ── Validation-failed branch ─────────────────────────────
        if not accumulated.get("validation_passed"):
            missing = accumulated.get("missing_fields", [])
            update_session_status(session_id, status="validation_failed", missing_fields=missing)
            yield _sse({
                "event":             "done",
                "status":            "validation_failed",
                "validation_passed": False,
                "missing_fields":    missing,
                "elapsed_seconds":   round(elapsed, 2),
            })
            return

        # ── Mid-pipeline error captured in state.error ────────────
        if accumulated.get("error"):
            update_session_status(session_id, status="error")
            yield _sse({"event": "error", "message": accumulated["error"]})
            return

        # ── Persist results ──────────────────────────────────────
        save_analysis(
            session_id=session_id,
            resume_issues=accumulated.get("resume_issues", []),
            skills_found=accumulated.get("skills_found", []),
            questions=accumulated.get("questions", []),
            salary_range=accumulated.get("salary_range", {}),
            active_companies=accumulated.get("active_companies", []),
        )
        update_session_status(
            session_id,
            status="analysed",
            chunks_count=accumulated.get("chunks_count", 0),
        )

        yield _sse({
            "event":             "done",
            "status":            "analysed",
            "session_id":        session_id,
            "validation_passed": True,
            "missing_fields":    [],
            "chunks_count":      accumulated.get("chunks_count", 0),
            "role_type":         accumulated.get("role_type", ""),
            "role_description":  accumulated.get("role_description", ""),
            "skills_found":      accumulated.get("skills_found", []),
            "resume_issues":     accumulated.get("resume_issues", []),
            "questions":         accumulated.get("questions", []),
            "salary_range":      accumulated.get("salary_range", {}),
            "active_companies":  accumulated.get("active_companies", []),
            "final_report":      accumulated.get("final_report"),
            "elapsed_seconds":   round(elapsed, 2),
        })

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",   # prevents nginx from buffering SSE frames
        },
    )
