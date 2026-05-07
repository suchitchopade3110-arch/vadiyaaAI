"""
VaidyaAI — Report Analysis Route
POST /analyze/report/{report_type}
"""

import uuid
from pathlib import Path

import aiofiles
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from celery.result import AsyncResult

from app.schemas.report import ReportAsyncResponse, ReportTypeEnum
from app.schemas.job import JobStatus
from app.workers.pipeline_tasks import analyze_report_task
from app.workers.celery_app import celery_app

router = APIRouter()

VALID_REPORT_TYPES = {t.value for t in ReportTypeEnum}
REPORT_EXTENSIONS = {".pdf", ".csv", ".txt"}
REPORT_MAX_BYTES = {
    ".pdf": 20 * 1024 * 1024,
    ".csv": 10 * 1024 * 1024,
    ".txt": 10 * 1024 * 1024,
}
UPLOAD_ROOT = Path("uploads").resolve()


async def _save_report_upload(file: UploadFile, job_id: str) -> tuple[str, str]:
    ext = Path(file.filename or "").suffix.lower()
    if ext not in REPORT_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail={
                "code": "FILE_FORMAT_UNSUPPORTED",
                "message": f"Extension '{ext}' not allowed. Allowed: {sorted(REPORT_EXTENSIONS)}",
            },
        )

    content = await file.read()
    max_bytes = REPORT_MAX_BYTES[ext]
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={
                "code": "FILE_TOO_LARGE",
                "message": f"File exceeds {max_bytes // 1024 // 1024}MB limit",
            },
        )

    save_dir = UPLOAD_ROOT / "reports"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{job_id}{ext}"
    async with aiofiles.open(save_path, "wb") as handle:
        await handle.write(content)
    return str(save_path), ext.lstrip(".")


@router.post("/{report_type}", status_code=202)
async def submit_report_analysis(
    report_type: str,
    file: UploadFile = File(..., description="Lab report / clinical note: PDF or CSV"),
    patient_id: str = Form(None),
    gender: str = Form("male"),
    age: int = Form(40),
    explanation_mode: str = Form("brief"),
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

    job_id = str(uuid.uuid4())
    file_path, _extension = await _save_report_upload(file, job_id)

    # ── Dispatch Celery task ───────────────────────────────────────────────
    task = analyze_report_task.apply_async(
        args=[file_path, patient_id, report_type, gender, age, explanation_mode],
        task_id=job_id,
        queue="reports",
    )

    return {
        "job_id": task.id,
        "status": "queued",
        "poll_url": f"/api/v1/jobs/{task.id}",
        "estimated_seconds": 15,
    }


@router.get("/status/{task_id}", response_model=JobStatus)
@router.get("/report/status/{task_id}", response_model=JobStatus)
async def get_report_status(task_id: str):
    """Poll report analysis job status."""
    request_id = str(uuid.uuid4())
    result = AsyncResult(task_id, app=celery_app)
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


@router.get("/{task_id}")
async def get_report_combined(task_id: str):
    """Combined status and result endpoint for the frontend."""
    result = AsyncResult(task_id, app=celery_app)
    state = result.state

    if state == "SUCCESS":
        return result.result
    
    return {
        "status": state.lower(),
        "task_id": task_id,
        "id": task_id
    }


@router.get("/report/result/{task_id}")
async def get_report_result(task_id: str):
    """Retrieve completed report analysis result."""
    result = AsyncResult(task_id, app=celery_app)
    if result.state != "SUCCESS":
        raise HTTPException(
            status_code=status.HTTP_425_TOO_EARLY,
            detail=f"Task not complete. Status: {result.state}",
        )
    data = result.result
    data["request_id"] = str(uuid.uuid4())
    return data
