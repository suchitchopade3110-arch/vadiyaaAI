import uuid

from fastapi import APIRouter, Depends, Request

from app.core.auth import get_current_user
from app.core.disclaimer import MEDICAL_DISCLAIMER
from app.schemas.claim import ClaimRequest
from app.workers.claim_tasks import verify_claim as verify_claim_task
from app.workers.db_persist import insert_claim
from app.main import limiter

router = APIRouter()


@router.post(
    "/claim/{claim_id}",
    status_code=202,
    summary="Submit a medical claim for verification",
)
@limiter.limit("15/minute")
async def verify_claim(
    request: Request,
    claim_id: str,
    body: ClaimRequest,
    user=Depends(get_current_user),
):
    """Queue a medical claim for async fact-checking against retrieved evidence.

    Returns immediately with a `job_id`; poll `/jobs/{job_id}` for the
    verdict, confidence score, and citations once the Celery task completes.
    Requires a valid access token.
    """
    job_id = str(uuid.uuid4())
    task = verify_claim_task.apply_async(
        args=[claim_id, body.claim_text],
        kwargs={"patient_id": str(body.patient_id) if body.patient_id else None},
        task_id=job_id,
        priority=9 if body.priority == "high" else 5,
    )
    insert_claim(
        claim_id,
        body.claim_text,
        task.id,
        str(body.patient_id) if body.patient_id else None,
    )
    return {
        "job_id": task.id,
        "task_id": task.id,
        "status": "queued",
        "poll_url": f"/jobs/{task.id}",
        "estimated_seconds": 15,
        "medical_disclaimer": MEDICAL_DISCLAIMER,
    }
