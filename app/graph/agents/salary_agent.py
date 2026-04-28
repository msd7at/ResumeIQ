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


# ──────────────────────────────────────────────────────────────────
#  PHASE 3 — Web search integration (DONE in Step 3.3):
#    Live DuckDuckGo lookups now ground BOTH sub-agents:
#      4a (salary)    → site-targeted searches against levels.fyi /
#                       glassdoor.com / ambitionbox.com for current numbers
#      4b (companies) → live hiring activity for the candidate's
#                       skill + location combination
#    The tool is fail-soft — if web returns [], prompts gracefully show
#    "(no web search results available)" and the agent falls back to
#    pure training-knowledge mode.
# ──────────────────────────────────────────────────────────────────


_SALARY_PROMPT = """You are a salary / compensation expert for the Indian and global market in 2026,
covering ALL professions (software, marketing, sales, finance, design, healthcare,
consulting, operations, etc.).

Estimate a realistic salary range for this candidate.

Candidate profile:
  Role type         : {role_type}
  Role description  : {role_description}
  Skills            : {skills}
  Experience level  : {experience_level}
  Location          : {location}
  Target company    : {target_company}

LIVE web salary data (levels.fyi / Glassdoor / AmbitionBox / PayScale):
{web_context}

If the live data above shows specific numbers for this role+skill+location+company combo,
USE those numbers as the anchor for your range. Override training-time estimates
when live data is available — currency staleness is the biggest accuracy risk here.

Role-type considerations (apply the relevant ones for {role_type}):
  • Software / data       → skill premium for AI/cloud; FAANG vs services pay gap
  • Marketing             → brand vs performance specialism; agency vs in-house;
                            FMCG vs SaaS pay differential
  • Sales                 → base + variable split (often 60/40 or 70/30); quota
                            attainment multiplier; OTE concept
  • Finance / IB          → base + bonus (bonus often 50-100% of base at IB);
                            buy-side vs sell-side; Big-4 vs corporate
  • Consulting            → tier matters (MBB vs Tier-2 vs boutique); analyst /
                            consultant / EM bands
  • Design                → tech-design vs agency vs in-house; portfolio premium
  • Healthcare            → public vs private; specialism premium; on-call factor
  • Education             → public vs private; tier-1 college vs others

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


_COMPANIES_PROMPT = """You are a hiring-market analyst for India in 2026, covering all professions.
List companies actively hiring for roles matching this candidate.

Candidate profile:
  Role type         : {role_type}
  Role description  : {role_description}
  Skills            : {skills}
  Experience level  : {experience_level}
  Location          : {location}

LIVE hiring activity from the web (LinkedIn jobs / Naukri / careers pages):
{web_context}

If the live data above mentions companies actively hiring for this role+skill+location,
PRIORITISE those companies in your list — they are confirmed hiring NOW.
Only fall back to training-knowledge defaults when web data is empty.

Mix the list to fit {role_type}. Examples:
  • Software / data → 2-3 FAANG, 3-4 Indian product (Flipkart, Razorpay, Swiggy,
                      PhonePe), 2-3 startups, 1-2 service cos
  • Marketing       → 2-3 FMCG (HUL, Nestle, ITC), 2-3 D2C (Mamaearth, boAt),
                      1-2 agencies (Wpp, Publicis), 2-3 SaaS (Zoho, Freshworks)
  • Finance / IB    → 2-3 Big-4 (Deloitte, EY, KPMG, PwC), 2-3 banks (HDFC, ICICI,
                      JPMorgan, GS), 1-2 PE/VC, 1-2 corporate finance
  • Consulting      → MBB (McKinsey, BCG, Bain), Tier-2 (Kearney, Strategy&,
                      Accenture Strat), Big-4 advisory
  • Healthcare      → Apollo, Fortis, Max, AIIMS, Manipal, Narayana
  • Design          → tech-design (Razorpay, Dunzo), agency (RGA, AKQA),
                      product (Microsoft, Adobe), boutique studios
  • Other roles     → adapt the mix sensibly to that profession's hiring landscape

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
    """
    Cheap heuristic — keyword scan covering common seniority markers across
    multiple professions (tech, marketing, finance, ops, healthcare, etc.).
    """
    lower = context.lower()
    if any(k in lower for k in [
        "principal", "architect", "head of", "director", "vice president", "vp,",
        "partner", "chief", "founder", "cxo", "general manager",
    ]):
        return "senior (7+ years, leadership exposure)"
    if any(k in lower for k in ["lead", "senior", "manager", "associate director"]):
        return "mid-senior (4-7 years)"
    if any(k in lower for k in [
        "intern", "fresher", "graduate", "trainee", "associate analyst", "junior",
    ]):
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
    session_id       = state["session_id"]
    skills           = state["skills_found"]
    location         = state["user_location"] or "India (general)"
    target_company   = state["target_company"] or "unspecified"
    role_type        = state.get("role_type") or "general professional"
    role_description = state.get("role_description") or "candidate from a non-specified profession"

    # Experience level inferred from work-history chunks
    hr_context = _build_context(session_id, [
        "work experience leadership team responsibility",
        "summary objective career profile",
    ])
    experience_level = _detect_experience_level(hr_context)
    skills_str       = ", ".join(skills) if skills else "no skills detected"
    top_skill        = skills[0] if skills else role_type

    # ── Live web grounding (per sub-agent — different queries) ────
    # Salary search: role + site-target compensation databases
    salary_query = (
        f"{target_company} {role_type} {top_skill} salary {location} 2026 "
        f"site:levels.fyi OR site:glassdoor.com OR site:ambitionbox.com OR site:payscale.com"
        if target_company != "unspecified"
        else f"{role_type} {top_skill} salary {location} 2026 "
             f"site:levels.fyi OR site:glassdoor.com OR site:ambitionbox.com OR site:payscale.com"
    )
    salary_web = format_search_results(web_search(salary_query, max_results=5))

    # Hiring search: role-aware, location-aware
    hiring_query = f"{role_type} {top_skill} jobs hiring {location} 2026"
    hiring_web   = format_search_results(web_search(hiring_query, max_results=6))

    client = Client(host=_OLLAMA_BASE_URL)

    # ── 4a: Salary range ──────────────────────────────────────────
    salary_resp = client.chat(
        model=_LLM_MODEL,
        messages=[{"role": "user", "content": _SALARY_PROMPT.format(
            role_type=role_type,
            role_description=role_description,
            skills=skills_str,
            experience_level=experience_level,
            location=location,
            target_company=target_company,
            web_context=salary_web,
        )}],
        format="json",
        options={"temperature": 0.3},
    )

    # ── 4b: Active companies ──────────────────────────────────────
    companies_resp = client.chat(
        model=_LLM_MODEL,
        messages=[{"role": "user", "content": _COMPANIES_PROMPT.format(
            role_type=role_type,
            role_description=role_description,
            skills=skills_str,
            experience_level=experience_level,
            location=location,
            web_context=hiring_web,
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
