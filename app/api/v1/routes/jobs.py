from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.workers.job_status import get_task_status, revoke_task
from app.db.session import get_db

router = APIRouter()


class JobStatusResponse(BaseModel):
    task_id: str
    state: str
    info: dict


class RecentJobItem(BaseModel):
    job_id: str
    pipeline: str          # "report" | "image" | "claim"
    celery_task_id: Optional[str]
    status: str
    created_at: Optional[datetime]

    model_config = {"from_attributes": True}


class RecentJobsResponse(BaseModel):
    jobs: List[RecentJobItem]
    total: int


@router.get(
    "/{task_id}",
    response_model=JobStatusResponse,
    summary="Poll live Celery task status",
)
async def job_status(task_id: str):
    """
    Real-time task state from Redis backend.
    States: PENDING → STARTED → SUCCESS | FAILURE | RETRY

    Use this for polling after submitting any /verify or /analyze request.
    """
    return get_task_status(task_id)


@router.delete(
    "/{task_id}",
    summary="Cancel a queued or running task",
)
async def cancel_job(task_id: str, terminate: bool = False):
    """
    Revoke a Celery task. Set terminate=true to SIGTERM a running worker.
    Use with caution on GPU image tasks.
    """
    revoke_task(task_id, terminate=terminate)
    return {"task_id": task_id, "cancelled": True}


@router.get(
    "",
    response_model=RecentJobsResponse,
    summary="List recent jobs across all pipelines",
)
async def list_recent_jobs(
    limit: int = 20,
    db: AsyncSession = Depends(get_db),
):
    """
    Returns recent jobs from all three pipelines (reports, images, claims)
    merged and sorted newest-first. Used by the Job Status Tracker frontend.
    """
    from app.models.report import Report
    from app.models.image_analysis import ImageAnalysis
    from app.models.claim import Claim

    jobs: List[RecentJobItem] = []

    # ── Reports ──────────────────────────────────────────────────────────────
    reports_result = await db.execute(
        select(Report).order_by(Report.created_at.desc()).limit(limit)
    )
    for r in reports_result.scalars().all():
        jobs.append(RecentJobItem(
            job_id=str(r.id),
            pipeline="report",
            celery_task_id=r.celery_task_id,
            status=str(r.uncertainty_flag),  # placeholder until report has status col
            created_at=r.created_at,
        ))

    # ── Images ───────────────────────────────────────────────────────────────
    images_result = await db.execute(
        select(ImageAnalysis).order_by(ImageAnalysis.created_at.desc()).limit(limit)
    )
    for img in images_result.scalars().all():
        jobs.append(RecentJobItem(
            job_id=str(img.id),
            pipeline=f"image/{img.image_type.value}",
            celery_task_id=img.celery_task_id,
            status="processing" if img.celery_task_id else "pending",
            created_at=img.created_at,
        ))

    # ── Claims ───────────────────────────────────────────────────────────────
    claims_result = await db.execute(
        select(Claim).order_by(Claim.created_at.desc()).limit(limit)
    )
    for c in claims_result.scalars().all():
        jobs.append(RecentJobItem(
            job_id=str(c.id),
            pipeline="claim",
            celery_task_id=c.celery_task_id,
            status=c.status.value,
            created_at=c.created_at,
        ))

    # Sort merged list newest-first
    jobs.sort(key=lambda j: j.created_at or datetime.min, reverse=True)
    jobs = jobs[:limit]

    return RecentJobsResponse(jobs=jobs, total=len(jobs))

