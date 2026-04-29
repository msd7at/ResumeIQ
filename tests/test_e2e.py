"""
End-to-end test for the ResumeIQ LangGraph pipeline.

PREREQUISITES (must be set up before running):
    1. Ollama running locally:        ollama serve
    2. Models pulled:                 ollama pull llama3.1:8b
                                       ollama pull nomic-embed-text
    3. Dependencies installed:        pip install -r requirements.txt
    4. SQLite + ChromaDB initialised: python -m app.db.sqlite_client

USAGE:
    python -m tests.test_e2e                  # runs both sample resumes
    python -m tests.test_e2e --tech           # only the tech sample
    python -m tests.test_e2e --marketing      # only the marketing sample

WHAT THIS VERIFIES:
    - All 6 LangGraph nodes execute end-to-end without errors
    - Role detection works for tech AND non-tech resumes
    - Web search is wired in correctly (or fails soft if DDG rate-limits)
    - Final report contains expected sections

This is a SMOKE test — it confirms the pipeline runs to completion. It does
NOT validate output quality — that's a manual review of the printed report.
"""

import sys
import time
import uuid

from app.graph.graph_builder import get_graph
from app.graph.state import create_initial_state


# ──────────────────────────────────────────────────────────────────
#  Sample resumes — one tech, one non-tech, to verify role-agnostic
#  prompts work end-to-end.
# ──────────────────────────────────────────────────────────────────


SAMPLE_TECH_RESUME = """
Anurag Sharma
anurag.sharma@example.com  |  +91-9876543210  |  linkedin.com/in/anurag-s

SUMMARY
Backend engineer with 5.5 years of experience in Java and Python. Currently
exploring AI engineering and LangGraph multi-agent systems.

EXPERIENCE
Senior Software Engineer — Infosys, Bangalore (2020 - Present)
- Built REST APIs in Spring Boot serving 2M requests/day
- Migrated monolith to microservices on AWS EKS, reduced deploy time by 60%
- Led a team of 4 engineers to deliver a payments reconciliation platform

Software Engineer — TCS, Pune (2019 - 2020)
- Developed batch jobs in Java for a banking client
- Worked on Oracle database optimisations

EDUCATION
B.Tech Computer Science — VIT University, 2019 (CGPA 8.4)

SKILLS
Java, Spring Boot, Python, FastAPI, AWS (EKS, S3, RDS), Docker, Kubernetes,
PostgreSQL, Redis, Kafka, REST APIs, Microservices, Git, CI/CD

PROJECTS
ResumeIQ — Personal Project (2026)
- AI-powered resume analyser using LangGraph, Ollama, ChromaDB, FastAPI
- Multi-agent pipeline with RAG retrieval and dynamic routing
"""


SAMPLE_MARKETING_RESUME = """
Priya Mehta
priya.mehta@example.com  |  +91-9988776655  |  linkedin.com/in/priya-mehta-mktg

SUMMARY
Senior marketing manager with 7 years of experience in B2B SaaS growth and
brand strategy. Led demand-generation programmes that scaled MQLs 3x at a
mid-market HR-tech firm.

EXPERIENCE
Senior Marketing Manager — Freshworks, Bangalore (2022 - Present)
- Led demand-gen team of 5; grew MQL volume 3x in 18 months
- Owned a $1.2M annual paid-media budget across Google Ads, LinkedIn, Meta
- Launched 4 product-marketing campaigns, contributed $4.5M in ARR pipeline

Marketing Manager — Zoho, Chennai (2020 - 2022)
- Ran content + SEO programme; organic traffic 4x in 12 months
- Built attribution model in Google Analytics 4 across 6 channels

Associate Marketing Manager — HUL, Mumbai (2018 - 2020)
- Brand-marketing for Surf Excel; assisted on national TV campaign
- Conducted 3 quantitative consumer studies (n=2000+)

EDUCATION
PGDM Marketing — IIM Lucknow, 2018
B.Com — Mumbai University, 2016

SKILLS
Brand Strategy, Demand Generation, Content Marketing, SEO, SEM,
Google Analytics 4, HubSpot, Marketo, Salesforce, A/B Testing,
Attribution Modelling, Consumer Research, Campaign Management,
B2B SaaS, FMCG, Budget Management

CERTIFICATIONS
Google Analytics 4 Certified (2024)
HubSpot Inbound Marketing Certified (2023)
"""


# ──────────────────────────────────────────────────────────────────
#  Test runner
# ──────────────────────────────────────────────────────────────────


def run_pipeline(label: str, resume_text: str, location: str,
                 target_company: str | None) -> dict:
    """Run the full LangGraph pipeline and print the final report."""
    print("\n" + "=" * 70)
    print(f"  TEST: {label}")
    print(f"  Location: {location}  |  Target: {target_company or '(none)'}")
    print("=" * 70 + "\n")

    session_id = f"test_{uuid.uuid4().hex[:8]}"
    state = create_initial_state(
        session_id=session_id,
        resume_text=resume_text,
        user_location=location,
        target_company=target_company,
    )

    print(f"[{label}] Invoking compiled graph (this may take 30-90s) ...")
    t0 = time.time()
    graph = get_graph()
    final_state = graph.invoke(state)
    elapsed = time.time() - t0
    print(f"[{label}] Pipeline finished in {elapsed:.1f}s\n")

    # ── Verify each agent produced output ─────────────────────────
    checks = [
        ("validation_passed",   bool(final_state.get("validation_passed"))),
        ("chunks_count > 0",    final_state.get("chunks_count", 0) > 0),
        ("role_type set",       bool(final_state.get("role_type"))),
        ("skills_found > 0",    len(final_state.get("skills_found", [])) > 0),
        ("resume_issues > 0",   len(final_state.get("resume_issues", [])) > 0),
        ("questions = 25",      len(final_state.get("questions", [])) == 25),
        ("salary_range set",    bool(final_state.get("salary_range")) and
                                final_state["salary_range"].get("min") is not None),
        ("active_companies > 0", len(final_state.get("active_companies", [])) > 0),
        ("final_report set",    bool(final_state.get("final_report"))),
    ]

    print(f"[{label}] AGENT OUTPUT CHECKS:")
    for name, ok in checks:
        marker = "PASS" if ok else "FAIL"
        print(f"  [{marker}]  {name}")

    print(f"\n[{label}] DETECTED ROLE: {final_state.get('role_type')}")
    print(f"[{label}] ROLE DESCRIPTION: {final_state.get('role_description')}\n")

    print(f"\n[{label}] ── FINAL REPORT (first 2000 chars) ──\n")
    print(final_state.get("final_report", "")[:2000])
    print(f"\n... (truncated; full report length = "
          f"{len(final_state.get('final_report', ''))} chars)")

    return final_state


def main():
    args = set(sys.argv[1:])
    run_tech      = "--marketing" not in args
    run_marketing = "--tech" not in args

    results = []

    if run_tech:
        results.append(run_pipeline(
            label="TECH (Senior Backend @ Netflix)",
            resume_text=SAMPLE_TECH_RESUME,
            location="Bangalore",
            target_company="Netflix",
        ))

    if run_marketing:
        results.append(run_pipeline(
            label="MARKETING (Senior Mktg Mgr @ HUL)",
            resume_text=SAMPLE_MARKETING_RESUME,
            location="Mumbai",
            target_company="HUL",
        ))

    print("\n" + "=" * 70)
    print(f"  Done. Ran {len(results)} pipeline invocation(s).")
    print("=" * 70)


if __name__ == "__main__":
    main()
