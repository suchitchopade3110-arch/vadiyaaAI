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
from app.workers.pipeline_tasks import verify_claim_task
from app.workers.celery_app import celery_app
from app.core.disclaimer import MEDICAL_DISCLAIMER

router = APIRouter()


@router.post("/claim", status_code=202)
async def submit_claim(
    payload: ClaimRequest,
):
    """
    Submit a medical claim for verification.
    
    Returns immediately with task_id. Poll /verify/claim/status/{task_id} for result.
    
    Pipeline (async):
    ClinicalBERT NER → BioGPT → ChromaDB → GPT-4/Llama → Hallucination Check
    """
    job_id = str(uuid.uuid4())

    # ── Dispatch Celery task ───────────────────────────────────────────────
    task = verify_claim_task.apply_async(
        args=[payload.claim_text, str(payload.patient_id) if payload.patient_id else None],
        task_id=job_id,
        queue="claims",
        priority=9 if payload.priority == "high" else 5,
    )

    return {
        "job_id": task.id,
        "status": "queued",
        "poll_url": f"/api/v1/jobs/{task.id}",
        "estimated_seconds": 10,
    }


@router.get("/claim/{task_id}")
async def get_claim_combined(task_id: str):
    """Combined status and result endpoint for the frontend."""
    result = AsyncResult(task_id, app=celery_app)
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
    result = AsyncResult(task_id, app=celery_app)

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
    result = AsyncResult(task_id, app=celery_app)
    
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
