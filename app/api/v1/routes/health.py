from datetime import timezone, datetime

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from sqlalchemy import text

from app.core.config import settings
from app.db.session import engine

UTC = timezone.utc

router = APIRouter()


@router.get("")
async def health():
    return {
        "status": "ok",
        "service": "VaidyaAI",
        "timestamp": datetime.now(UTC).isoformat(),
        "version": "1.0.0",
    }


@router.get("/live")
async def live():
    """Liveness probe — process is up. No dependency checks."""
    return {"status": "alive", "timestamp": datetime.now(UTC).isoformat()}


@router.get("/ready")
async def ready():
    """Readiness probe — checks DB and Redis connectivity."""
    checks = await _run_readiness_checks()
    all_ok = all(check["ok"] for check in checks.values())
    payload = {
        "status": "ready" if all_ok else "not_ready",
        "checks": checks,
        "timestamp": datetime.now(UTC).isoformat(),
    }
    return JSONResponse(status_code=200 if all_ok else 503, content=payload)


async def _run_readiness_checks() -> dict:
    checks: dict = {}

    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        checks["database"] = {"ok": True}
    except Exception as exc:
        checks["database"] = {"ok": False, "error": str(exc)}

    try:
        import redis

        client = redis.Redis.from_url(settings.REDIS_URL, socket_connect_timeout=2)
        client.ping()
        checks["redis"] = {"ok": True}
    except Exception as exc:
        checks["redis"] = {"ok": False, "error": str(exc)}

    return checks
