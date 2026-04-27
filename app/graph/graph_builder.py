"""
LangGraph wiring for ResumeIQ.

Pipeline (6 nodes + 1 entry):

    [START]
       |
       v
    validator         ← Phase 1 validator wrapped as a guard node
       |
       v   (route_after_validation)
       |---> END   if validation_passed = False
       |---> embedding  if valid
                |
                v
           resume_analyser   ← Agent 1
                |
                v   (route_after_analysis)
                |---> END   on error
                |---> question_generator   ← Agent 3 (sub-agents 3a/3b/3c)
                                |
                                v   (route_after_questions)
                                |---> END   on error
                                |---> report_compiler   if no skills
                                |---> salary_agent      ← Agent 4
                                              |
                                              v   (route_after_salary)
                                              |---> END   on error
                                              |---> report_compiler   ← Agent 5
                                                          |
                                                          v
                                                        [END]

The API handler is expected to pre-parse the file (via parse_pdf / parse_docx)
and supply state['resume_text'] before invoking the graph. Parsing lives outside
the graph because PDF/DOCX selection is purely a file-extension concern.
"""

from langgraph.graph import StateGraph, END

from app.graph.state import ResumeState

from app.graph.agents.resume_analyser    import resume_analyser_node
from app.graph.agents.question_generator import question_generator_node
from app.graph.agents.salary_agent       import salary_node
from app.graph.agents.report_compiler    import report_compiler_node
from app.graph.agents.router import (
    route_after_validation,
    route_after_analysis,
    route_after_questions,
    route_after_salary,
    ROUTE_END, ROUTE_ANALYSE, ROUTE_QUESTIONS, ROUTE_SALARY, ROUTE_REPORT,
)

from app.rag.validator    import validate_resume
from app.rag.chunker      import chunk_resume
from app.rag.embedder     import embed_chunks
from app.rag.vector_store import store_embeddings, delete_session


# ──────────────────────────────────────────────────────────────────
#  Thin wrapper nodes for Phase 1 RAG operations.
#  Live here (not in app/rag/) because they are graph plumbing —
#  they translate state[x] in/out for the underlying pure functions.
# ──────────────────────────────────────────────────────────────────


def validator_node(state: ResumeState) -> dict:
    """Run validate_resume() on state['resume_text'] and update state."""
    result = validate_resume(state["resume_text"])
    return {
        "validation_passed": result.is_valid,
        "missing_fields":    result.missing_fields,
        "current_step":      "validated" if result.is_valid else "validation_failed",
    }


def embedding_node(state: ResumeState) -> dict:
    """
    Chunk + embed + store in ChromaDB.
    Deletes any existing chunks for this session_id first to support re-uploads.
    """
    session_id = state["session_id"]
    delete_session(session_id)
    chunks   = chunk_resume(state["resume_text"], session_id=session_id)
    embedded = embed_chunks(chunks)
    stored   = store_embeddings(embedded)
    return {
        "chunks_count": stored,
        "current_step": "embedded",
    }


# ──────────────────────────────────────────────────────────────────
#  Build & compile
# ──────────────────────────────────────────────────────────────────


def build_graph():
    """
    Wire all nodes + routers into a compiled LangGraph state machine.
    Returns the compiled graph — call .invoke(initial_state) to run.
    """
    graph = StateGraph(ResumeState)

    # ── Register nodes ────────────────────────────────────────────
    graph.add_node("validator",          validator_node)
    graph.add_node("embedding",          embedding_node)
    graph.add_node("resume_analyser",    resume_analyser_node)
    graph.add_node("question_generator", question_generator_node)
    graph.add_node("salary_agent",       salary_node)
    graph.add_node("report_compiler",    report_compiler_node)

    # ── Entry point ───────────────────────────────────────────────
    graph.set_entry_point("validator")

    # ── Unconditional edges ───────────────────────────────────────
    graph.add_edge("embedding",       "resume_analyser")
    graph.add_edge("report_compiler", END)

    # ── Conditional edges (routers) ───────────────────────────────
    graph.add_conditional_edges(
        "validator",
        route_after_validation,
        {
            ROUTE_ANALYSE: "embedding",   # validation passed → run RAG ingestion next
            ROUTE_END:     END,           # validation failed → stop with missing_fields set
        },
    )

    graph.add_conditional_edges(
        "resume_analyser",
        route_after_analysis,
        {
            ROUTE_QUESTIONS: "question_generator",
            ROUTE_END:       END,
        },
    )

    graph.add_conditional_edges(
        "question_generator",
        route_after_questions,
        {
            ROUTE_SALARY: "salary_agent",
            ROUTE_REPORT: "report_compiler",   # skip salary if no skills detected
            ROUTE_END:    END,
        },
    )

    graph.add_conditional_edges(
        "salary_agent",
        route_after_salary,
        {
            ROUTE_REPORT: "report_compiler",
            ROUTE_END:    END,
        },
    )

    return graph.compile()


# ──────────────────────────────────────────────────────────────────
#  Lazy singleton — compile once on first request, reuse for all
#  subsequent invocations. Compilation is idempotent but not free.
# ──────────────────────────────────────────────────────────────────


_compiled_graph = None


def get_graph():
    """Return the compiled graph — compiles on first call, cached afterwards."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph
