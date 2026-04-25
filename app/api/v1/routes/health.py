from fastapi import APIRouter
from datetime import datetime

router = APIRouter()

@router.get("")
async def health():
    return {
        "status": "ok",
        "service": "VaidyaAI",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
    }
