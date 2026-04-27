"""
VaidyaAI — Report Analysis Route
POST /analyze/report/{report_type}
"""

import uuid
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from celery.result import AsyncResult

from app.db.session import get_db
from app.schemas.report import ReportAsyncResponse, ReportTypeEnum
from app.schemas.job import JobStatus
from app.services.file_upload import file_upload_service
from app.workers.report_tasks import analyze_report
from app.core.disclaimer import MEDICAL_DISCLAIMER
from app.workers.db_persist import insert_report

router = APIRouter()

VALID_REPORT_TYPES = {t.value for t in ReportTypeEnum}


@router.post("/{report_type}", response_model=ReportAsyncResponse, status_code=202)
async def submit_report_analysis(
    report_type: str,
    file: UploadFile = File(..., description="Lab report / clinical note: PDF or CSV"),
    patient_id: str = None,
    db: AsyncSession = Depends(get_db),
):
    """
    Submit lab report / clinical note for analysis.
    
    Supported report types: lab | clinical | discharge
    Supported formats: .pdf, .csv
    
    Pipeline (async):
    OCR → ClinicalBERT NER → XGBoost → SHAP → Anomaly Detection → LLM Explain
    
    Returns immediately. Poll /api/v1/analyze/report/status/{task_id} for result (~10–20s).
    """
    if report_type not in VALID_REPORT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid report_type '{report_type}'. Valid: {VALID_REPORT_TYPES}",
        )

    request_id = str(uuid.uuid4())
    report_id = str(uuid.uuid4())

    # ── Save uploaded file ─────────────────────────────────────────────────
    file_path, extension, pipeline_type = await file_upload_service.validate_and_save(
        file, sub_dir=f"reports/{report_id}"
    )

    if pipeline_type != "text":
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"File type '.{extension}' is not a supported report format. Use PDF or CSV.",
        )

    # ── Dispatch Celery task ───────────────────────────────────────────────
    task = analyze_report.apply_async(
        args=[report_id, file_path, report_type, extension],
    )

    # ── Insert pending row in DB ───────────────────────────────────────────
    insert_report(report_id, report_type, file_path, extension, task.id, patient_id)

    return ReportAsyncResponse(
        request_id=request_id,
        report_id=uuid.UUID(report_id),
        task_id=task.id,
        id=task.id,
        status="pending",
        report_type=ReportTypeEnum(report_type),
        poll_url=f"/api/v1/analyze/report/{task.id}",
        estimated_seconds=20,
        medical_disclaimer=MEDICAL_DISCLAIMER,
    )


@router.get("/{task_id}")
async def get_report_combined(task_id: str):
    """Combined status and result endpoint for the frontend."""
    result = AsyncResult(task_id)
    state = result.state

    if state == "SUCCESS":
        return result.result
    
    return {
        "status": state.lower(),
        "task_id": task_id,
        "id": task_id
    }


@router.get("/report/status/{task_id}", response_model=JobStatus)
async def get_report_status(task_id: str):
    """Poll report analysis job status."""
    request_id = str(uuid.uuid4())
    result = AsyncResult(task_id)
    state = result.state
    meta = result.info or {}

    if state == "PENDING":
        return JobStatus(request_id=request_id, task_id=task_id, status="pending")
    elif state == "PROGRESS":
        return JobStatus(
            request_id=request_id, task_id=task_id,
            status="processing", progress_pct=meta.get("pct", 0),
        )
    elif state == "SUCCESS":
        return JobStatus(
            request_id=request_id, task_id=task_id,
            status="complete", progress_pct=100,
            result_url=f"/api/v1/analyze/report/result/{task_id}",
        )
    elif state == "FAILURE":
        return JobStatus(request_id=request_id, task_id=task_id, status="failed", error=str(meta))
    return JobStatus(request_id=request_id, task_id=task_id, status=state.lower())


@router.get("/report/result/{task_id}")
async def get_report_result(task_id: str):
    """Retrieve completed report analysis result."""
    result = AsyncResult(task_id)
    if result.state != "SUCCESS":
        raise HTTPException(
            status_code=status.HTTP_425_TOO_EARLY,
            detail=f"Task not complete. Status: {result.state}",
        )
    data = result.result
    data["request_id"] = str(uuid.uuid4())
    return data
