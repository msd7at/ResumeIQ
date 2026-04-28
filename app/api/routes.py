"""
API routes for ResumeIQ.

This file holds the FastAPI APIRouter that app/main.py mounts under /api.

Endpoints (built incrementally across Steps 4.2 - 4.6):
    POST /api/upload     — accept resume file, parse, return session_id   (Step 4.2)
    POST /api/analyse    — kick off LangGraph analysis for a session      (Step 4.3)
    POST /api/chat       — follow-up chat about an analysed resume        (Step 4.4)
    GET  /api/status/{session_id} — current pipeline step                 (Step 4.5)
"""

from fastapi import APIRouter

router = APIRouter()


@router.get("/")
def api_root():
    """Lists the API surface — useful for `curl /api` to see what's available."""
    return {
        "service":   "ResumeIQ API",
        "endpoints": [
            "POST /api/upload",
            "POST /api/analyse",
            "POST /api/chat",
            "GET  /api/status/{session_id}",
        ],
        "note": "Endpoints are added incrementally in Phase 4 (Steps 4.2 - 4.6).",
    }
