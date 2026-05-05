from fastapi import APIRouter
from datetime import UTC, datetime

router = APIRouter()

@router.get("")
async def health():
    return {
        "status": "ok",
        "service": "VaidyaAI",
        "timestamp": datetime.now(UTC).isoformat(),
        "version": "1.0.0",
    }
