from typing import TypedDict


class ResumeState(TypedDict):
    # ── Input — set once at pipeline entry ────────────────────────
    session_id:     str
    resume_text:    str
    user_location:  str
    target_company: str | None

    # ── Validation — set by the validator node ────────────────────
    validation_passed: bool
    missing_fields:    list[str]

    # ── RAG metadata — set after chunking + embedding ─────────────
    chunks_count: int

    # ── Agent outputs ─────────────────────────────────────────────
    resume_issues:    list[str]   # Agent 1 — what's wrong with the resume
    skills_found:     list[str]   # Agent 1 — skills / competencies detected
    role_type:        str         # Agent 1 — broad profession ("software engineering",
                                  #          "marketing", "data science", "finance",
                                  #          "design", "sales", "healthcare", etc.)
    role_description: str         # Agent 1 — one-line role summary
    questions:        list[dict]  # Agent 3 — interview questions
    salary_range:     dict        # Agent 4 — salary + market data
    active_companies: list[str]   # Agent 4 — companies currently hiring

    # ── Pipeline control ──────────────────────────────────────────
    current_step: str
    error:        str | None

    # ── Final output — set by report compiler ─────────────────────
    final_report: str | None


def create_initial_state(
    session_id: str,
    resume_text: str,
    user_location: str,
    target_company: str | None = None,
) -> ResumeState:
    """
    Build the starting state before any agent runs.
    All agent-output fields are initialised to safe empty values
    so every node can read them without KeyError.
    """
    return ResumeState(
        session_id=session_id,
        resume_text=resume_text,
        user_location=user_location,
        target_company=target_company,
        validation_passed=False,
        missing_fields=[],
        chunks_count=0,
        resume_issues=[],
        skills_found=[],
        role_type="",
        role_description="",
        questions=[],
        salary_range={},
        active_companies=[],
        current_step="start",
        error=None,
        final_report=None,
    )
