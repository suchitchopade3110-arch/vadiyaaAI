import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from contextlib import asynccontextmanager

from app.core.config import settings
from app.core.logging import setup_logging
from app.core.middleware import (
    APIContractMiddleware,
    CorrelationIDMiddleware,
    ErrorHandlerMiddleware,
    MaxBodySizeMiddleware,
)
from app.api.v1.router import api_router
from app.db.session import engine
from app.db import base  # noqa: F401 — import models for Alembic
from fastapi.exceptions import RequestValidationError
from app.core.errors import validation_exception_handler, general_exception_handler
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(
    key_func=get_remote_address,  # swap to API key func later
    storage_uri="redis://localhost:6380/0"
)
from app.routes import (
    auth,
    claims,
    images,
    jobs,
    pdf_reports,
    reports,
    text_routes,
    websocket_routes,
)

log = logging.getLogger(__name__)

# Path to the stitched frontend pages (relative to project root)
_FRONTEND_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "ui"
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

# Starlette wraps middleware in reverse registration order.
# Keep CorrelationID outermost so APIContract/logging reuse the same request ID.
app.add_middleware(MaxBodySizeMiddleware)
app.add_middleware(APIContractMiddleware)
app.add_middleware(ErrorHandlerMiddleware)
app.add_middleware(CorrelationIDMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # dev-mode: open; tighten in prod via settings
    allow_credentials=False,      # must be False when allow_origins=["*"]
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, general_exception_handler)
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.state.limiter = limiter

app.include_router(api_router, prefix="/api/v1")
app.include_router(text_routes.router, prefix="/api", tags=["Direct Text Pipeline"])
app.include_router(websocket_routes.router)
app.include_router(auth.router, prefix="/auth", tags=["auth"])
app.include_router(claims.router, prefix="/api/v1/verify", tags=["claims"])
app.include_router(images.router, prefix="/api/v1/analyze", tags=["images"])
app.include_router(reports.router, prefix="/api/v1/analyze", tags=["reports"])
app.include_router(pdf_reports.router, prefix="/api/v1", tags=["PDF Reports"])
app.include_router(jobs.router, prefix="/jobs", tags=["jobs"])


@app.get("/health")
async def health():
    return {"status": "ok", "service": "VaidyaAI"}


@app.get("/")
async def root():
    """Platform metadata and medical disclaimer."""
    return {
        "platform": "VaidyaAI Medical Intelligence",
        "version": "1.0.0",
        "medical_disclaimer": settings.MEDICAL_DISCLAIMER,
        "endpoints": {
            "health": "/api/v1/health",
            "docs": "/docs",
            "ui": "/ui/index.html"
        }
    }


# ── Static file serving ───────────────────────────────────────────────────────
# Mount the stitch frontend directory so browsers can load the HTML pages.
# Pages are accessible at:
#   /ui/index.html

if os.path.isdir(_FRONTEND_DIR):
    app.mount("/ui", StaticFiles(directory=_FRONTEND_DIR, html=True), name="ui")
else:
    log.warning(f"Frontend directory not found at {_FRONTEND_DIR} — /ui not mounted")
