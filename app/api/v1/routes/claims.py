"""
VaidyaAI — Claim Verification Route
POST /verify/claim/{claim_id}
"""

import uuid
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from celery.result import AsyncResult
from datetime import datetime

from app.db.session import get_db
from app.schemas.claim import ClaimRequest, ClaimAsyncResponse, ClaimResult
from app.schemas.job import JobStatus
from app.workers.claim_tasks import verify_claim
from app.core.disclaimer import MEDICAL_DISCLAIMER
from app.workers.db_persist import insert_claim

router = APIRouter()


@router.post("/claim", response_model=ClaimAsyncResponse, status_code=202)
async def submit_claim(
    payload: ClaimRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Submit a medical claim for verification.
    
    Returns immediately with task_id. Poll /verify/claim/status/{task_id} for result.
    
    Pipeline (async):
    ClinicalBERT NER → BioGPT → ChromaDB → GPT-4/Llama → Hallucination Check
    """
    request_id = str(uuid.uuid4())

    # Generate a claim_id since the frontend doesn't provide one
    claim_id = str(uuid.uuid4())

    # ── Dispatch Celery task ───────────────────────────────────────────────
    task = verify_claim.apply_async(
        args=[claim_id, payload.claim_text],
        kwargs={"patient_id": str(payload.patient_id) if payload.patient_id else None},
        priority=9 if payload.priority == "high" else 5,
    )

    # ── Insert pending row in DB ───────────────────────────────────────────
    insert_claim(claim_id, payload.claim_text, task.id,
                 str(payload.patient_id) if payload.patient_id else None)

    return ClaimAsyncResponse(
        request_id=request_id,
        claim_id=uuid.UUID(claim_id),
        task_id=task.id,
        id=task.id,  # Frontend expects 'id'
        status="pending",
        poll_url=f"/api/v1/verify/claim/{task.id}",
        estimated_seconds=15,
        medical_disclaimer=MEDICAL_DISCLAIMER,
    )


@router.get("/claim/{task_id}")
async def get_claim_combined(task_id: str):
    """Combined status and result endpoint for the frontend."""
    result = AsyncResult(task_id)
    state = result.state

    if state == "SUCCESS":
        return result.result
    
    # Return status format the frontend understands
    return {
        "status": state.lower(),
        "task_id": task_id,
        "id": task_id
    }


@router.get("/claim/status/{task_id}", response_model=JobStatus)
async def get_claim_status(task_id: str):
    """Poll claim verification job status."""
    request_id = str(uuid.uuid4())
    result = AsyncResult(task_id)

    state = result.state
    meta = result.info or {}

    if state == "PENDING":
        return JobStatus(request_id=request_id, task_id=task_id, status="pending")
    elif state == "PROGRESS":
        return JobStatus(
            request_id=request_id,
            task_id=task_id,
            status="processing",
            progress_pct=meta.get("pct", 0),
        )
    elif state == "SUCCESS":
        return JobStatus(
            request_id=request_id,
            task_id=task_id,
            status="complete",
            progress_pct=100,
            result_url=f"/api/v1/verify/claim/result/{task_id}",
        )
    elif state == "FAILURE":
        return JobStatus(
            request_id=request_id,
            task_id=task_id,
            status="failed",
            error=str(meta),
        )
    return JobStatus(request_id=request_id, task_id=task_id, status=state.lower())


@router.get("/claim/result/{task_id}")
async def get_claim_result(task_id: str):
    """Retrieve completed claim verification result."""
    result = AsyncResult(task_id)
    
    if result.state != "SUCCESS":
        raise HTTPException(
            status_code=status.HTTP_425_TOO_EARLY,
            detail=f"Task not complete. Status: {result.state}. Poll /api/v1/verify/claim/status/{task_id}",
        )
    
    data = result.result
    data["request_id"] = str(uuid.uuid4())
    return data


def _is_valid_uuid(val: str) -> bool:
    try:
        uuid.UUID(val)
        return True
    except ValueError:
        return False
