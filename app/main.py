"""
ResumeIQ — FastAPI application entry point.

Bootstrap responsibilities:
  - Initialise SQLite (creates 3 tables on first run)
  - Pre-compile the LangGraph (avoids cold-start on the first /analyse request)
  - Mount API routes from app/api/routes.py
  - Mount the static frontend at the root path
  - Configure CORS (so the frontend served from a different host can call the API)

Run locally:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.db.sqlite_client import init_db
from app.graph.graph_builder import get_graph
from app.api.routes import router as api_router

load_dotenv()

_HOST = os.getenv("HOST", "0.0.0.0")
_PORT = int(os.getenv("PORT", "8000"))

_FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan handler — runs once at startup, then once at shutdown.
    Replaces the deprecated @app.on_event("startup"/"shutdown") decorators.
    """
    # ── Startup ───────────────────────────────────────────────────
    print("[startup] Initialising SQLite ...")
    init_db()

    print("[startup] Pre-compiling LangGraph ...")
    get_graph()  # primes the lazy singleton so the first /analyse is fast

    print(f"[startup] ResumeIQ ready on http://{_HOST}:{_PORT}")
    yield

    # ── Shutdown ──────────────────────────────────────────────────
    print("[shutdown] ResumeIQ stopping.")


app = FastAPI(
    title="ResumeIQ",
    description="AI-powered resume analyser using LangGraph + local LLMs",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — wide open for local dev. Tighten before deploying.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# All API endpoints live under /api (defined in app/api/routes.py)
app.include_router(api_router, prefix="/api")


# Health check — useful for container orchestrators / smoke tests
@app.get("/health")
def health():
    return {"status": "ok", "service": "ResumeIQ"}


# ──────────────────────────────────────────────────────────────────
#  Frontend serving — only mounted if the frontend folder exists.
#  The app stays usable as a pure backend if frontend is removed.
# ──────────────────────────────────────────────────────────────────
if _FRONTEND_DIR.exists():
    # Serve index.html at "/" — simpler than relying on StaticFiles' default
    @app.get("/")
    def serve_index():
        return FileResponse(_FRONTEND_DIR / "index.html")

    # Mount any additional assets (CSS, JS, images) under /static
    app.mount("/static", StaticFiles(directory=_FRONTEND_DIR), name="static")
