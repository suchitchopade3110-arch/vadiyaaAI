import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from contextlib import asynccontextmanager

from app.core.config import settings
from app.core.logging import setup_logging
from app.core.middleware import MaxBodySizeMiddleware
from app.api.v1.router import api_router
from app.db.session import engine
from app.db import base  # noqa: F401 — import models for Alembic

log = logging.getLogger(__name__)

# Path to the stitched frontend pages (relative to project root)
_FRONTEND_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "stitch_vaidyaai_clinical_analysis_suite",
    "stitch_vaidyaai_clinical_analysis_suite",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    log.info("VaidyaAI API starting up — frontend served at /ui")
    log.info(f"Frontend path: {_FRONTEND_DIR}")
    yield
    log.info("VaidyaAI API shutting down")


app = FastAPI(
    title="VaidyaAI",
    description="Medical Intelligence Platform — AI-assisted analysis. NOT a medical diagnosis.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Order matters: MaxBodySize before CORS
app.add_middleware(MaxBodySizeMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # dev-mode: open; tighten in prod via settings
    allow_credentials=False,      # must be False when allow_origins=["*"]
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api/v1")


@app.get("/health")
async def health():
    return {"status": "ok", "service": "VaidyaAI"}


@app.get("/", include_in_schema=False)
async def root_redirect():
    """Redirect / to the home page."""
    return RedirectResponse(url="/ui/index.html")


# ── Static file serving ───────────────────────────────────────────────────────
# Mount the stitch frontend directory so browsers can load the HTML pages.
# Pages are accessible at:
#   /ui/index.html

if os.path.isdir(_FRONTEND_DIR):
    app.mount("/ui", StaticFiles(directory=_FRONTEND_DIR, html=True), name="ui")
else:
    log.warning(f"Frontend directory not found at {_FRONTEND_DIR} — /ui not mounted")
