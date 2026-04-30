from fastapi import APIRouter, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

from app.models.request_models import QueryRequest
from app.services.pipeline_controller import run_text_pipeline

router = APIRouter()


@router.post("/text")
@limiter.limit("20/minute")
async def text_pipeline(request: Request, data: QueryRequest):
    try:
        result = run_text_pipeline(data.query)
        return {
            "type": "text_pipeline",
            "diagnosis": result["diagnosis"],
            "confidence": result["confidence"],
            "evidence": result["evidence"],
            "explanation": result["explanation"],
            "shap_values": result["shap"],
            "entities": result["entities"],
            "risk_factors": result["risk_factors"],
            "anomalies": result["anomalies"],
            "status": "completed",
        }
    except Exception as exc:
        return {
            "error": {
                "code": "PIPELINE_ERROR",
                "message": str(exc),
                "retryable": False,
            }
        }

