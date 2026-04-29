from app.graph.state import ResumeState


# ── Route labels (used as keys in graph_builder's conditional_edges) ────
ROUTE_END        = "end"
ROUTE_ANALYSE    = "analyse"
ROUTE_QUESTIONS  = "generate_questions"
ROUTE_SALARY     = "fetch_salary"
ROUTE_REPORT     = "compile_report"


def route_after_validation(state: ResumeState) -> str:
    """
    Decision point right after the validator runs.
    - validation failed         → end pipeline early (frontend shows missing fields)
    - validation passed         → proceed to resume analyser
    """
    if not state["validation_passed"]:
        return ROUTE_END
    return ROUTE_ANALYSE


def route_after_analysis(state: ResumeState) -> str:
    """
    Decision point after the resume analyser runs.
    - LLM error                 → end
    - any output produced       → continue to question generator
    """
    if state.get("error"):
        return ROUTE_END
    return ROUTE_QUESTIONS


def route_after_questions(state: ResumeState) -> str:
    """
    Decision point after the question generator runs.
    - error                     → end
    - no skills detected        → skip salary agent (web search would be useless)
                                  go straight to the report compiler
    - skills present            → run salary + market intel
    """
    if state.get("error"):
        return ROUTE_END
    if not state["skills_found"]:
        return ROUTE_REPORT
    return ROUTE_SALARY


def route_after_salary(state: ResumeState) -> str:
    """
    After the salary/market agent finishes, the report compiler always runs.
    Kept as an explicit function so the graph stays uniform — every transition
    goes through a router rather than a hard-coded edge.
    """
    if state.get("error"):
        return ROUTE_END
    return ROUTE_REPORT
