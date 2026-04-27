import os
import json
from ollama import Client
from dotenv import load_dotenv

from app.graph.state import ResumeState
from app.rag.embedder import embed_query
from app.rag.vector_store import retrieve_chunks

load_dotenv()

_OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
_LLM_MODEL       = os.getenv("OLLAMA_LLM_MODEL", "llama3.1:8b")


# Three queries cover the main resume areas — gives the LLM a balanced view
_RAG_QUERIES = [
    "skills technologies programming languages tools frameworks databases",
    "work experience job roles projects responsibilities achievements",
    "education degree certifications",
]


_ANALYSIS_PROMPT = """You are an expert technical resume reviewer.
Analyse the resume content below and respond with strict JSON.

Resume content (relevant sections):
---
{context}
---

Respond with ONLY this JSON object (no markdown, no commentary):
{{
  "skills_found": ["Python", "FastAPI", "Docker", "..."],
  "resume_issues": ["No quantified achievements in 2nd job", "Missing LinkedIn URL", "..."]
}}

Rules:
- skills_found: list ALL technical skills, tools, programming languages, frameworks,
  databases, and cloud services explicitly mentioned in the resume.
- resume_issues: list 3 to 7 specific, actionable weaknesses.
  GOOD examples:
    * "Job descriptions lack quantified impact (no numbers, percentages, scale)"
    * "Missing LinkedIn or GitHub URL in contact section"
    * "Action verbs are weak — 'worked on' instead of 'led', 'designed', 'built'"
    * "No tech stack mentioned for the most recent project"
  AVOID generic feedback like "improve formatting" or "add more details".
- Output ONLY the JSON object, nothing before or after."""


def resume_analyser_node(state: ResumeState) -> dict:
    """
    Agent 1 — RAG-powered resume analyser.

    1. Retrieves the most relevant chunks for skills/experience/education queries
    2. Sends combined context to llama3.1:8b with a strict JSON prompt
    3. Parses the response into skills_found and resume_issues

    Returns a partial state dict — LangGraph merges it into the full state.
    """
    session_id = state["session_id"]

    # ── Step 1: RAG retrieval ────────────────────────────────────
    context_parts: list[str] = []
    seen: set[str] = set()

    for query in _RAG_QUERIES:
        q_vec = embed_query(query)
        chunks = retrieve_chunks(q_vec, session_id=session_id, n_results=3)
        for c in chunks:
            text = c["text"]
            if text not in seen:
                context_parts.append(text)
                seen.add(text)

    context = "\n\n---\n\n".join(context_parts)

    # ── Step 2: send to LLM with JSON-mode forcing ────────────────
    client = Client(host=_OLLAMA_BASE_URL)
    response = client.chat(
        model=_LLM_MODEL,
        messages=[{"role": "user", "content": _ANALYSIS_PROMPT.format(context=context)}],
        format="json",
        options={"temperature": 0.2},
    )

    raw = response["message"]["content"].strip()

    # ── Step 3: parse JSON ────────────────────────────────────────
    try:
        parsed = json.loads(raw)
        skills = parsed.get("skills_found", [])
        issues = parsed.get("resume_issues", [])
    except json.JSONDecodeError:
        skills = []
        issues = ["Resume analysis produced malformed output — please re-run."]

    return {
        "skills_found":  skills,
        "resume_issues": issues,
        "current_step":  "resume_analysed",
    }
