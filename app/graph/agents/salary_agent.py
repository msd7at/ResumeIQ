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


# ──────────────────────────────────────────────────────────────────
#  TODO (Phase 3 — web search integration):
#    Replace LLM-knowledge-based salary numbers with LIVE results from
#    DuckDuckGo searches against Glassdoor / levels.fyi / AmbitionBox /
#    LinkedIn jobs. Salary number staleness is the single biggest
#    accuracy gap in this project — even more critical than refreshing
#    interview question patterns. Address it FIRST in Phase 3.
# ──────────────────────────────────────────────────────────────────


_SALARY_PROMPT = """You are a tech salary expert for the Indian and global market in 2026.
Estimate a realistic salary range for this candidate.

Candidate profile:
  Skills            : {skills}
  Experience level  : {experience_level}
  Location          : {location}
  Target company    : {target_company}

Output ONLY this JSON object (no markdown, no commentary):
{{
  "currency": "INR",
  "min": 1200000,
  "max": 2400000,
  "median": 1800000,
  "experience_band": "5-7 years",
  "factors": [
    "FastAPI / Python backend roles command a 15-20% premium in Bangalore (2026)",
    "AI / LLM-adjacent skills add 10-15% on top of base",
    "..."
  ],
  "company_specific": {{
    "Netflix": {{
      "min": 4500000,
      "max": 6500000,
      "note": "Senior backend at Netflix India sits well above market median due to global pay parity"
    }}
  }},
  "disclaimer": "Estimates based on 2024-2026 market data; verify with Glassdoor / levels.fyi / AmbitionBox before negotiating."
}}

Rules:
- Use INR for Indian locations, USD for US/global, EUR for Europe.
- Numbers must be REALISTIC and grounded in your knowledge of 2024-2026 hiring data.
  Do NOT lazily round to figures like 10L–20L — use specific values like 12,50,000.
- `factors`: 3 to 5 SPECIFIC factors that drove this range
  (skill premium, location effect, company tier, India vs Global parity).
- `company_specific`:
    * If {target_company} is "unspecified" → set it to an empty object {{}}.
    * Otherwise → include {target_company} with realistic min/max and a 1-line note
      explaining why it differs from market median.
- `experience_band`: phrase like "5-7 years" or "0-2 years" or "10+ years".
- Output ONLY the JSON object."""


_COMPANIES_PROMPT = """You are a tech hiring-market analyst for India in 2026.
List companies actively hiring for roles matching this candidate.

Candidate profile:
  Skills            : {skills}
  Experience level  : {experience_level}
  Location          : {location}

Output ONLY this JSON object (no markdown, no commentary):
{{
  "active_companies": [
    "Razorpay (Bangalore) — Hiring senior Python / FastAPI backend; matches your stack and location",
    "Swiggy (Bangalore) — Active backend hiring for payments and order platform; Python + Kafka",
    "..."
  ]
}}

Rules:
- 8 to 12 companies, prioritised by skill + location relevance.
- Mix: 2-3 FAANG / global, 3-4 Indian product cos (Flipkart, Razorpay, Swiggy, Zomato,
  CRED, Meesho, PhonePe, Groww, etc), 2-3 startups, 1-2 service cos if relevant.
- Each line format: "<Company> (<City>) — <Why they match>"
- The "why they match" must reference SPECIFIC skills the candidate has + the role type.
  Avoid generic phrasing like "great culture" or "growing fast".
- Skip companies that don't actually fit the candidate's level or skills.
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


def _safe_parse_json(response: dict) -> dict:
    try:
        return json.loads(response["message"]["content"].strip())
    except json.JSONDecodeError:
        return {}


def salary_node(state: ResumeState) -> dict:
    """
    Agent 4 — Salary + Market Intel.

    Two LLM calls inside one node:
        Agent 4a → estimate salary range
                   (skills + experience + location + target company)
        Agent 4b → list 8-12 active hiring companies
                   (skills + location + experience)

    Phase 3 will inject live web search results into both prompts to
    counter LLM-knowledge staleness on salary numbers and hiring activity.
    """
    session_id     = state["session_id"]
    skills         = state["skills_found"]
    location       = state["user_location"] or "India (general)"
    target_company = state["target_company"] or "unspecified"

    # Experience level inferred from work-history chunks
    hr_context = _build_context(session_id, [
        "work experience leadership team responsibility",
        "summary objective career profile",
    ])
    experience_level = _detect_experience_level(hr_context)
    skills_str       = ", ".join(skills) if skills else "general programming"

    client = Client(host=_OLLAMA_BASE_URL)

    # ── 4a: Salary range ──────────────────────────────────────────
    salary_resp = client.chat(
        model=_LLM_MODEL,
        messages=[{"role": "user", "content": _SALARY_PROMPT.format(
            skills=skills_str,
            experience_level=experience_level,
            location=location,
            target_company=target_company,
        )}],
        format="json",
        options={"temperature": 0.3},
    )

    # ── 4b: Active companies ──────────────────────────────────────
    companies_resp = client.chat(
        model=_LLM_MODEL,
        messages=[{"role": "user", "content": _COMPANIES_PROMPT.format(
            skills=skills_str,
            experience_level=experience_level,
            location=location,
        )}],
        format="json",
        options={"temperature": 0.4},
    )

    salary_data    = _safe_parse_json(salary_resp)
    companies_data = _safe_parse_json(companies_resp)

    return {
        "salary_range":     salary_data,
        "active_companies": companies_data.get("active_companies", []),
        "current_step":     "salary_analysed",
    }
