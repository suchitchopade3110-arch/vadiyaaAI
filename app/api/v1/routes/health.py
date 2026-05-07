from fastapi import APIRouter
from datetime import timezone, datetime
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
