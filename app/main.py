import os
import logging
import threading
from datetime import timezone, datetime
UTC = timezone.utc

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from app.core.config import settings
from app.core.logging import setup_logging
from app.core.middleware import (
    APIContractMiddleware,
    CorrelationIDMiddleware,
    ErrorHandlerMiddleware,
    MaxBodySizeMiddleware,
    TimeoutMiddleware,
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
    storage_uri=settings.REDIS_URL,
)
from app.routes import (
    auth,
    claims,
    images,
    jobs,
    patient_history_routes,
    pdf_reports,
    qr_reports,
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


def _ingest_who_multimodal_on_startup() -> None:
    """Idempotently ensure multimodal WHO RAG chunks exist in ChromaDB."""
    enabled = os.getenv("WHO_MULTIMODAL_STARTUP_INGEST", "true").lower() not in {"0", "false", "no"}
    if not enabled:
        log.info("WHO multimodal startup ingest disabled")
        return
    try:
        from app.services.ingest_who_multimodal_criteria import ingest_who_multimodal

        uploaded = ingest_who_multimodal()
        log.info("WHO multimodal startup ingest complete; uploaded=%s", uploaded)
    except Exception as exc:
        log.warning("WHO multimodal startup ingest skipped: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    log.info("VaidyaAI API starting up — frontend served at /ui")
    log.info(f"Frontend path: {_FRONTEND_DIR}")
    try:
        from app.services.speed_optimizations import prewarm_groq, verify_cache_connection

        verify_cache_connection()
        threading.Thread(target=prewarm_groq, daemon=True).start()
        log.info("Groq pre-warm initiated in background")
    except Exception as exc:
        log.warning("Startup speed warmup skipped: %s", exc)
    threading.Thread(target=_ingest_who_multimodal_on_startup, daemon=True).start()
    log.info("WHO multimodal RAG ingest check initiated in background")
    yield
    log.info("VaidyaAI API shutting down")


_IS_PRODUCTION = settings.APP_ENV == "production"

_CORS_ORIGINS_ENV = os.getenv("CORS_ORIGINS")
if _IS_PRODUCTION and not _CORS_ORIGINS_ENV:
    raise RuntimeError(
        "CORS_ORIGINS must be set explicitly when APP_ENV=production "
        "(refusing to fall back to the localhost default)."
    )
_CORS_ORIGINS = (_CORS_ORIGINS_ENV or "http://localhost:3000").split(",")

app = FastAPI(
    title="VaidyaAI",
    description="Medical Intelligence Platform — AI-assisted analysis. NOT a medical diagnosis.",
    version="1.0.0",
    docs_url=None if _IS_PRODUCTION else "/docs",
    redoc_url=None if _IS_PRODUCTION else "/redoc",
    lifespan=lifespan,
)

# Starlette wraps middleware in reverse registration order.
# Keep CorrelationID outermost so APIContract/logging reuse the same request ID.
app.add_middleware(MaxBodySizeMiddleware)
app.add_middleware(TimeoutMiddleware)
app.add_middleware(APIContractMiddleware)
app.add_middleware(ErrorHandlerMiddleware)
app.add_middleware(CorrelationIDMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)

app.add_exception_handler(RequestValidationError, validation_exception_handler)


async def http_exception_handler(request: Request, exc: HTTPException):
    detail = exc.detail if isinstance(exc.detail, dict) else {
        "code": "ERROR",
        "message": str(exc.detail),
    }
    request_id = getattr(request.state, "request_id", "unknown")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                **detail,
                "request_id": request_id,
                "timestamp": datetime.now(UTC).isoformat(),
            }
        },
    )


app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(Exception, general_exception_handler)
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.state.limiter = limiter

app.include_router(api_router, prefix="/api/v1")
app.include_router(text_routes.router, prefix="/api", tags=["Direct Text Pipeline"])
app.include_router(websocket_routes.router)
app.include_router(patient_history_routes.router)
app.include_router(auth.router, prefix="/auth", tags=["auth"])
# Removed duplicated routes that are already included via api_router
app.include_router(pdf_reports.router, prefix="/api/v1", tags=["PDF Reports"])
app.include_router(qr_reports.router)



@app.get("/health")
async def health():
    return {"status": "ok", "service": "VaidyaAI"}


@app.get("/live")
async def live():
    """Liveness probe — process is up. No dependency checks."""
    return {"status": "alive"}


@app.get("/ready")
async def ready():
    """Readiness probe — checks DB and Redis connectivity."""
    from app.api.v1.routes.health import _run_readiness_checks

    checks = await _run_readiness_checks()
    all_ok = all(check["ok"] for check in checks.values())
    payload = {"status": "ready" if all_ok else "not_ready", "checks": checks}
    return JSONResponse(status_code=200 if all_ok else 503, content=payload)


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

_YOLO_OUTPUT_DIR = os.path.join(os.getcwd(), "data", "yolo_outputs")
os.makedirs(_YOLO_OUTPUT_DIR, exist_ok=True)
if os.path.isdir(_YOLO_OUTPUT_DIR):
    app.mount("/yolo_outputs", StaticFiles(directory=_YOLO_OUTPUT_DIR), name="yolo_outputs")
