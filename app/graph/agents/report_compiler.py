import os
import json
from datetime import datetime
from ollama import Client
from dotenv import load_dotenv

from app.graph.state import ResumeState

load_dotenv()

_OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
_LLM_MODEL       = os.getenv("OLLAMA_LLM_MODEL", "llama3.1:8b")


# ──────────────────────────────────────────────────────────────────
#  Design choice — TEMPLATE-BASED with TWO LLM calls.
#  Everything we need is already in state from earlier agents — there
#  is no benefit in asking the LLM to re-format structured data.
#  LLM is reserved for the two parts that genuinely need prose:
#    1. Executive summary (sets the tone)
#    2. Action plan (prioritised, contextual next steps)
#  The rest is deterministic markdown assembly.
# ──────────────────────────────────────────────────────────────────


_SUMMARY_PROMPT = """You are writing the executive summary of a resume analysis report.

Candidate context:
  Role type                             : {role_type}
  Role description                      : {role_description}
  Skills detected ({skill_count})       : {skills_preview}
  Resume issues found ({issue_count})   : {issues_preview}
  Target company                        : {target_company}
  Location                              : {location}
  Salary range estimated                : {salary_summary}

Write a concise executive summary (4 to 6 sentences) that:
  1. Captures who this candidate is at a glance.
  2. Highlights the strongest skill area.
  3. Names the most critical resume issue to fix.
  4. Notes the target compensation band.
  5. Sets the tone for the actionable report below.

Output ONLY the prose. No markdown headings. No bullet points. No labels."""


_ACTION_PLAN_PROMPT = """Based on the candidate analysis below, generate a prioritised 5-point action plan.

Candidate context:
  Role type         : {role_type}
  Skills            : {skills}
  Top resume issues : {issues}
  Target            : {target_company} in {location}

Output ONLY this JSON object (no markdown, no commentary):
{{
  "actions": [
    "1. Specific action with concrete deliverable",
    "2. ..."
  ]
}}

Rules:
- Exactly 5 actions, ordered by impact (most impactful first).
- Each action must be CONCRETE — name the section, the file, the topic, or the
  specific behaviour to change. Avoid vague advice like "improve resume".
- Include 1 action specific to {target_company}'s known interview pattern
  (e.g. "Practice 5 LLD problems on payment-flow style questions for Razorpay").
- Output ONLY the JSON object."""


# ──────────────────────────────────────────────────────────────────
#  Formatting helpers (deterministic — no LLM)
# ──────────────────────────────────────────────────────────────────


def _format_inr(amount: int | None) -> str:
    """Format Indian-style: 1500000 -> 15,00,000"""
    if amount is None:
        return "-"
    s = str(int(amount))
    if len(s) <= 3:
        return s
    last_three = s[-3:]
    rest = s[:-3]
    grouped = ""
    while rest:
        grouped = rest[-2:] + ("," if grouped else "") + grouped
        rest = rest[:-2]
    return grouped + "," + last_three


def _format_currency(amount: int | None, currency: str) -> str:
    if amount is None:
        return "-"
    if currency == "INR":
        return f"INR {_format_inr(amount)}"
    if currency == "USD":
        return f"${int(amount):,}"
    return f"{int(amount):,} {currency}"


def _format_question_group(title: str, questions: list[dict]) -> list[str]:
    parts = [f"### {title} ({len(questions)})", ""]
    if not questions:
        parts.append("_None generated._")
        return parts

    for i, q in enumerate(questions, 1):
        difficulty = (q.get("difficulty") or "?").upper()
        parts.append(f"**Q{i}.** [{difficulty}] {q.get('question', '')}")
        parts.append("")

        if q.get("category"):
            parts.append(f"- **Category:** {q['category']}")

        topics = q.get("expected_topics", [])
        if topics:
            parts.append(f"- **Expected topics:** {', '.join(topics)}")

        snippet = q.get("code_snippet")
        if snippet and snippet not in ("null", "None"):
            parts.append("- **Code:**")
            parts.append("  ```")
            for line in str(snippet).splitlines():
                parts.append(f"  {line}")
            parts.append("  ```")

        if q.get("based_on"):
            parts.append(f"- **From your resume:** _{q['based_on']}_")

        if q.get("company_style_match"):
            parts.append(f"- **Why this style:** {q['company_style_match']}")

        if q.get("market_relevance"):
            parts.append(f"- **Market context:** {q['market_relevance']}")

        if q.get("covered_in_resume") is False:
            parts.append("- **Note:** _Not in your resume — study extra hard._")

        parts.append("")

    return parts


def _format_salary(salary_range: dict) -> list[str]:
    if not salary_range or salary_range.get("min") is None:
        return ["_Salary range could not be estimated._"]

    currency = salary_range.get("currency", "INR")
    parts = [
        f"**Range:** {_format_currency(salary_range.get('min'), currency)} - "
        f"{_format_currency(salary_range.get('max'), currency)} per annum",
    ]
    if salary_range.get("median"):
        parts.append(f"**Median:** {_format_currency(salary_range['median'], currency)}")
    if salary_range.get("experience_band"):
        parts.append(f"**Experience band:** {salary_range['experience_band']}")

    factors = salary_range.get("factors") or []
    if factors:
        parts.append("")
        parts.append("**Factors driving the range:**")
        for f in factors:
            parts.append(f"- {f}")

    company_specific = salary_range.get("company_specific") or {}
    if company_specific:
        parts.append("")
        parts.append("**Company-specific estimate:**")
        for company, data in company_specific.items():
            cmin = _format_currency(data.get("min"), currency)
            cmax = _format_currency(data.get("max"), currency)
            parts.append(f"- **{company}:** {cmin} - {cmax}")
            if data.get("note"):
                parts.append(f"  _{data['note']}_")

    if salary_range.get("disclaimer"):
        parts.append("")
        parts.append(f"> {salary_range['disclaimer']}")

    return parts


def _assemble_report(
    *,
    executive_summary: str,
    skills: list[str],
    issues: list[str],
    questions: list[dict],
    salary_range: dict,
    active_companies: list[str],
    actions: list[str],
    target_company: str,
    location: str,
    role_type: str,
    role_description: str,
) -> str:
    parts: list[str] = [
        "# Resume Analysis Report",
        f"_Role: **{role_type}** — {role_description}_" if role_description else f"_Role: **{role_type}**_",
        f"_Target: **{target_company}** in **{location}**_",
        f"_Date: {datetime.utcnow().strftime('%Y-%m-%d')}_",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
        executive_summary,
        "",
        "---",
        "",
        "## Resume Analysis",
        "",
        "### Skills Detected",
        "",
        ", ".join(skills) if skills else "_No skills detected._",
        "",
        "### Issues to Fix",
        "",
    ]
    if issues:
        parts.extend(f"- {iss}" for iss in issues)
    else:
        parts.append("_No issues identified._")

    parts.extend([
        "",
        "---",
        "",
        "## Interview Preparation",
        "",
        f"_25 questions = 15 skill-based + 5 project / resume + 5 HR. "
        f"Tailored to **{target_company}**'s style for a **{role_type}** candidate._",
        "",
    ])

    # Accept both new "skill" type and legacy "technical" type for backward-compat
    skill_qs   = [q for q in questions if q.get("type") in ("skill", "technical")]
    project_qs = [q for q in questions if q.get("type") == "project"]
    hr_qs      = [q for q in questions if q.get("type") == "hr"]

    parts.extend(_format_question_group("Skill-Based Questions", skill_qs))
    parts.extend(_format_question_group("Resume / Project Questions", project_qs))
    parts.extend(_format_question_group("HR / Behavioral Questions", hr_qs))

    parts.extend([
        "",
        "---",
        "",
        "## Salary Insights",
        "",
    ])
    parts.extend(_format_salary(salary_range))

    parts.extend([
        "",
        "---",
        "",
        f"## Active Hiring Companies ({location})",
        "",
    ])
    if active_companies:
        for c in active_companies:
            parts.append(f"- {c}")
    else:
        parts.append("_No active hiring data available._")

    parts.extend([
        "",
        "---",
        "",
        "## Action Plan",
        "",
        f"_Prioritised next steps to land a role at {target_company}_",
        "",
    ])
    if actions:
        parts.extend(actions)
    else:
        parts.append("_Action plan could not be generated._")

    parts.extend([
        "",
        "---",
        "",
        "_Report generated by ResumeIQ — AI-powered resume analyser._",
    ])

    return "\n".join(parts)


# ──────────────────────────────────────────────────────────────────
#  The LangGraph node
# ──────────────────────────────────────────────────────────────────


def report_compiler_node(state: ResumeState) -> dict:
    """
    Agent 5 — Report Compiler.

    Combines all prior agents' outputs into a single polished markdown report.

    Two LLM calls (executive summary + action plan) — the rest is deterministic
    template-based formatting since we already have structured data from earlier
    agents. There is no need to ask the LLM to re-format what we already know.
    """
    skills           = state["skills_found"]
    issues           = state["resume_issues"]
    questions        = state["questions"]
    salary_range     = state["salary_range"]
    active_companies = state["active_companies"]
    target_company   = state["target_company"] or "any company"
    location         = state["user_location"] or "India"
    role_type        = state.get("role_type") or "general professional"
    role_description = state.get("role_description") or ""

    client = Client(host=_OLLAMA_BASE_URL)

    # ── Build short salary summary string for the summary prompt ──
    salary_summary = "not estimated"
    if salary_range and salary_range.get("min") and salary_range.get("max"):
        currency = salary_range.get("currency", "INR")
        salary_summary = (
            f"{_format_currency(salary_range['min'], currency)} - "
            f"{_format_currency(salary_range['max'], currency)}"
        )

    # ── LLM call 1: Executive Summary (free-form prose) ───────────
    summary_resp = client.chat(
        model=_LLM_MODEL,
        messages=[{"role": "user", "content": _SUMMARY_PROMPT.format(
            role_type=role_type,
            role_description=role_description,
            skill_count=len(skills),
            skills_preview=", ".join(skills[:8]) + ("..." if len(skills) > 8 else ""),
            issue_count=len(issues),
            issues_preview="; ".join(issues[:3]) if issues else "none",
            target_company=target_company,
            location=location,
            salary_summary=salary_summary,
        )}],
        options={"temperature": 0.4},
    )
    executive_summary = summary_resp["message"]["content"].strip()

    # ── LLM call 2: Action Plan (JSON) ────────────────────────────
    action_resp = client.chat(
        model=_LLM_MODEL,
        messages=[{"role": "user", "content": _ACTION_PLAN_PROMPT.format(
            role_type=role_type,
            skills=", ".join(skills[:10]),
            issues="; ".join(issues[:5]) if issues else "no major issues",
            target_company=target_company,
            location=location,
        )}],
        format="json",
        options={"temperature": 0.3},
    )
    try:
        action_data = json.loads(action_resp["message"]["content"].strip())
        actions = action_data.get("actions", [])
    except json.JSONDecodeError:
        actions = []

    # ── Deterministic markdown assembly ───────────────────────────
    final_report = _assemble_report(
        executive_summary=executive_summary,
        skills=skills,
        issues=issues,
        questions=questions,
        salary_range=salary_range,
        active_companies=active_companies,
        actions=actions,
        target_company=target_company,
        location=location,
        role_type=role_type,
        role_description=role_description,
    )

    return {
        "final_report": final_report,
        "current_step": "report_compiled",
    }
