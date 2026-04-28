import os
import json
from ollama import Client
from dotenv import load_dotenv

from app.graph.state import ResumeState
from app.rag.embedder import embed_query
from app.rag.vector_store import retrieve_chunks
from app.tools.web_search import web_search, format_search_results

load_dotenv()

_OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
_LLM_MODEL       = os.getenv("OLLAMA_LLM_MODEL", "llama3.1:8b")

# Total per interview = 25  (15 technical + 5 project + 5 HR)
_TECHNICAL_COUNT = 15
_PROJECT_COUNT   = 5
_HR_COUNT        = 5


# ──────────────────────────────────────────────────────────────────
#  KEY DESIGN RULE for technical questions:
#  The TARGET COMPANY decides the interview pattern. If the company is
#  known for a topic (e.g. Netflix → System Design / HLD / LLD), those
#  questions MUST appear EVEN IF the candidate's resume does not list
#  that skill. The candidate has to face them on interview day either way.
#
#  PHASE 3 — Web search integration (DONE in Step 3.2):
#    Live DuckDuckGo lookups for recent {target_company} interview reports
#    are now injected into all 3 prompts as `{web_context}`. This refreshes
#    the LLM's stale training-time knowledge with current signals.
#    The tool is fail-soft — if web search returns [], prompts gracefully
#    show "(no web search results available)" and the agent falls back to
#    pure training-knowledge mode.
# ──────────────────────────────────────────────────────────────────


_TECHNICAL_PROMPT = """You are a senior technical interviewer at {target_company} in 2026.
Your goal: prepare this candidate for the questions they will ACTUALLY face at {target_company}.

═══════════════════════════════════════════════════════════════════
PRIMARY SIGNAL — {target_company}'s MANDATORY interview topics
═══════════════════════════════════════════════════════════════════
Recall the topics this company is HISTORICALLY known to interview on.
These topics MUST appear in the question set even if the candidate's resume
does NOT list them — the candidate WILL be asked them on interview day.

Reference cheatsheet (apply only what fits {target_company}):
  • Netflix          → Distributed Systems, System Design (HLD + LLD),
                       Microservices, Observability, Chaos Engineering,
                       JVM tuning, Fault Tolerance, Senior judgement
  • Google           → Algorithms, Data Structures, System Design at scale,
                       Complexity analysis, Distributed systems
  • Amazon           → Algorithms, System Design, OOP, Leadership Principles
                       overlay on tech, Scalability, AWS internals
  • Meta / Facebook  → Algorithms, large-scale System Design, Product engineering
  • Microsoft        → Balanced coding + System Design + culture-fit
  • Apple            → Low-level systems, performance, memory, craft
  • Stripe / Airbnb  → Real-world product engineering, API design, idempotency
  • Flipkart / Zomato / Swiggy / Razorpay
                     → DSA, HLD, LLD, India-scale traffic, payment correctness,
                       latency, fault tolerance
  • Infosys / TCS / Wipro / Accenture / Capgemini
                     → Fundamentals, project walkthroughs, basic DSA,
                       framework basics, client communication
  • Series-A / B startup
                     → Ownership, breadth, real production debugging
  • If {target_company} is "unspecified"
                     → balanced FAANG-adjacent style

═══════════════════════════════════════════════════════════════════
LIVE WEB CONTEXT — recent reports on {target_company}'s interview style
═══════════════════════════════════════════════════════════════════
{web_context}

If the web context above shows a TOPIC SHIFT or NEW PATTERN, weight it
heavily and OVERRIDE the cheatsheet above. Recent signal beats stale knowledge.

═══════════════════════════════════════════════════════════════════
SECONDARY SIGNAL — Candidate's actual skills
═══════════════════════════════════════════════════════════════════
Skills in the resume: {skills}
Use these to GROUND some questions in the candidate's stated stack.

═══════════════════════════════════════════════════════════════════
TERTIARY SIGNAL — Location: {location}
═══════════════════════════════════════════════════════════════════
Weight applicable topics toward the local industry where it makes sense.

═══════════════════════════════════════════════════════════════════
COVERAGE TARGETS for {count} questions
═══════════════════════════════════════════════════════════════════
  ~ 50% on {target_company}'s standard interview topics (resume coverage NOT required)
  ~ 40% on the candidate's listed skills, framed in {target_company}'s style
  ~ 10% supporting / market-relevant cross-over questions

Resume context (skills + projects):
---
{context}
---

Output ONLY this JSON object (no markdown, no commentary):
{{
  "questions": [
    {{
      "type": "technical",
      "category": "System Design",
      "question": "Phrased the way {target_company} would ask it.",
      "difficulty": "medium",
      "expected_topics": ["topic1", "topic2", "topic3"],
      "code_snippet": "10-25 lines of code, or null for non-coding",
      "covered_in_resume": true,
      "company_style_match": "why this question matches {target_company}'s pattern",
      "market_relevance": "why this matters in 2026"
    }}
  ]
}}

Rules:
- Generate EXACTLY {count} questions.
- Difficulty mix: 4 easy, 7 medium, 4 hard.
- Set `covered_in_resume = false` when the topic is asked due to {target_company}'s
  pattern but the resume does not show it. Set true when grounded in resume.
- Include 3-5 CODING questions with a real `code_snippet`.
- Each question must be SPECIFIC, TESTABLE, non-generic.
- Output ONLY the JSON object."""


_PROJECT_PROMPT = """You are a hiring manager at {target_company} reading this candidate's resume.
Generate exactly {count} questions tied to SPECIFIC projects and claims in the resume,
framed the way {target_company} would probe these projects.

PRIMARY SIGNAL — Target company: {target_company}
  Recall {target_company}'s style of probing project claims:
    * Netflix → senior-judgement: "why this design? what trade-offs? what broke first?"
    * Amazon  → STAR mapping to Leadership Principles
    * Google  → deep design discussion, scale assumptions, complexity rigour
    * Indian product → scale, latency, payment correctness, fault tolerance probes
    * Service cos    → end-to-end walkthrough, client handling, delivery
  If {target_company} is "unspecified", default to balanced senior-engineer probes.

Recent web reports on {target_company}'s interview style:
{web_context}

If web reports above show how this company probes projects (e.g. specific
question formats, depth expectations), weight that over the cheatsheet.

Candidate's location: {location}

Resume context (projects + experience):
---
{context}
---

Output ONLY this JSON object (no markdown, no commentary):
{{
  "questions": [
    {{
      "type": "project",
      "category": "system design",
      "question": "In your <specific project>, why <specific tech>? What trade-offs?",
      "difficulty": "medium",
      "expected_topics": ["decision rationale", "alternatives", "trade-offs"],
      "based_on": "exact resume line / project that prompted this question",
      "company_style_match": "the angle {target_company} would push on"
    }}
  ]
}}

Rules:
- Generate EXACTLY {count} questions.
- Each question MUST reference a SPECIFIC detail from the resume.
- Probe deeper than the resume states — verify the claim is real.
- `based_on` must quote the exact resume line / phrase that prompted the question.
- Frame each question through {target_company}'s lens.
- Output ONLY the JSON object."""


_HR_PROMPT = """You are an HR interviewer at {target_company}.
Generate exactly {count} behavioral questions in {target_company}'s SPECIFIC framework.

PRIMARY SIGNAL — Target company: {target_company}
  Map every question to that company's known behavioral framework:
    * Amazon         → strict Leadership Principles (Customer Obsession, Ownership,
                       Bias for Action, Dive Deep, Are Right A Lot, Earn Trust, etc.)
    * Google         → "googliness" + structured general behavioral
    * Microsoft      → growth mindset + collaborative impact
    * Netflix        → senior-judgement, freedom & responsibility culture
    * Apple          → attention to craft, owning quality
    * Indian product → ownership at scale, on-call/incident stories, learning velocity
    * Service cos    → client communication, delivery pressure, team collaboration
  If {target_company} is "unspecified", default to balanced senior-engineer behavioral set.

Recent web reports on {target_company}'s behavioral / culture interview:
{web_context}

If the web reports surface specific themes (e.g. "Amazon LP focus shift in 2026",
"new culture round at Netflix"), reflect them in the question set.

Candidate's experience level : {experience_level}
Candidate's location         : {location}

Resume context (experience + summary):
---
{context}
---

Output ONLY this JSON object (no markdown, no commentary):
{{
  "questions": [
    {{
      "type": "hr",
      "category": "leadership",
      "question": "Tell me about a time ...",
      "difficulty": "medium",
      "expected_topics": ["situation", "specific action", "outcome", "stakeholders"],
      "company_style_match": "which {target_company} principle / theme this maps to"
    }}
  ]
}}

Rules:
- Generate EXACTLY {count} questions, each from a DIFFERENT theme.
- For Amazon, every question MUST map to a specific Leadership Principle.
- For other companies, map to that company's known behavioral theme.
- Tailor depth to experience level — do not ask deep leadership questions of a fresher.
- Use STAR-friendly framing (Situation / Task / Action / Result).
- Output ONLY the JSON object."""


def _detect_experience_level(context: str) -> str:
    lower = context.lower()
    if any(k in lower for k in ["principal", "architect", "manager", "head of", "director"]):
        return "senior (7+ years, leadership exposure)"
    if "lead" in lower or "senior" in lower:
        return "mid-senior (4-7 years)"
    if any(k in lower for k in ["intern", "fresher", "graduate", "trainee"]):
        return "junior / fresher (0-1 year)"
    return "mid-level (2-4 years)"


def _build_context(session_id: str, queries: list[str], n_each: int = 3) -> str:
    seen: set[str] = set()
    parts: list[str] = []
    for q in queries:
        chunks = retrieve_chunks(embed_query(q), session_id=session_id, n_results=n_each)
        for c in chunks:
            text = c["text"]
            if text not in seen:
                parts.append(text)
                seen.add(text)
    return "\n\n---\n\n".join(parts)


def _safe_parse(response: dict) -> list[dict]:
    try:
        parsed = json.loads(response["message"]["content"].strip())
        return parsed.get("questions", [])
    except json.JSONDecodeError:
        return []


def question_generator_node(state: ResumeState) -> dict:
    """
    Agent 3 — generates 25 interview questions across THREE sub-agents.
    PRIMARY anchor in every sub-agent is target_company's interview pattern.

    KEY RULE: target_company's MANDATORY topics override the candidate's resume.
    If Netflix is the target, System Design / HLD / LLD questions appear even if
    the resume never mentions them — because the candidate WILL face them on the day.

        Agent 3a → 15 technical questions in target_company's style
        Agent 3b →  5 project-specific verification questions
        Agent 3c →  5 HR / behavioral questions in target_company's framework
    """
    session_id     = state["session_id"]
    skills         = state["skills_found"]
    location       = state["user_location"] or "India (general)"
    target_company = state["target_company"] or "unspecified"

    tech_context = _build_context(session_id, [
        "skills technologies frameworks programming languages tools",
        "projects technical implementation built designed architecture",
    ])

    project_context = _build_context(session_id, [
        "projects portfolio personal projects built developed",
        "experience role responsibilities impact achievements metrics",
    ])

    hr_context = _build_context(session_id, [
        "work experience leadership team responsibility ownership",
        "summary objective career profile background",
    ])

    experience_level = _detect_experience_level(hr_context)

    # ── Live web context (one search, reused across all 3 sub-agents) ──
    # Skip the search when target is unspecified — generic results add noise.
    if target_company != "unspecified":
        interview_results = web_search(
            f"{target_company} software engineer interview process questions 2026",
            max_results=6,
        )
        web_context = format_search_results(interview_results)
    else:
        web_context = "(no target company set — skipping web lookup)"

    client = Client(host=_OLLAMA_BASE_URL)

    tech_resp = client.chat(
        model=_LLM_MODEL,
        messages=[{"role": "user", "content": _TECHNICAL_PROMPT.format(
            count=_TECHNICAL_COUNT,
            target_company=target_company,
            skills=", ".join(skills) if skills else "general programming",
            location=location,
            web_context=web_context,
            context=tech_context,
        )}],
        format="json",
        options={"temperature": 0.3},
    )

    project_resp = client.chat(
        model=_LLM_MODEL,
        messages=[{"role": "user", "content": _PROJECT_PROMPT.format(
            count=_PROJECT_COUNT,
            target_company=target_company,
            location=location,
            web_context=web_context,
            context=project_context,
        )}],
        format="json",
        options={"temperature": 0.3},
    )

    hr_resp = client.chat(
        model=_LLM_MODEL,
        messages=[{"role": "user", "content": _HR_PROMPT.format(
            count=_HR_COUNT,
            target_company=target_company,
            experience_level=experience_level,
            location=location,
            web_context=web_context,
            context=hr_context,
        )}],
        format="json",
        options={"temperature": 0.4},
    )

    all_questions: list[dict] = []
    all_questions.extend(_safe_parse(tech_resp))
    all_questions.extend(_safe_parse(project_resp))
    all_questions.extend(_safe_parse(hr_resp))

    return {
        "questions":    all_questions,
        "current_step": "questions_generated",
    }
