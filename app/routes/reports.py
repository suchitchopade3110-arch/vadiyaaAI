import uuid

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status

from app.core.auth import get_current_user
from app.core.disclaimer import MEDICAL_DISCLAIMER
from app.schemas.report import ReportTypeEnum
from app.services.file_upload import file_upload_service
from app.workers.db_persist import insert_report
from app.workers.report_tasks import analyze_report as analyze_report_task
from app.main import limiter

router = APIRouter()

VALID_TYPES = {item.value for item in ReportTypeEnum}


@router.post(
    "/report/{report_type}",
    status_code=202,
    summary="Upload a lab report for analysis",
)
@limiter.limit("10/minute")
async def analyze_report(
    request: Request,
    report_type: str,
    file: UploadFile = File(...),
    patient_id: str | None = None,
    user=Depends(get_current_user),
):
    """Queue a lab report (PDF/CSV/image) for async OCR + NER + risk analysis.

    Returns immediately with a `job_id`; poll `/jobs/{job_id}` for the
    extracted entities, risk prediction, anomalies, and explanation once the
    Celery task completes. Requires a valid access token.
    """
    if report_type not in VALID_TYPES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid report_type. Use: {sorted(VALID_TYPES)}",
        )

    report_id = str(uuid.uuid4())
    file_path, extension, pipeline_type = await file_upload_service.validate_and_save(
        file, sub_dir=f"reports/{report_id}"
    )
    if pipeline_type != "text":
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"File type '.{extension}' is not a supported report format.",
        )

    job_id = str(uuid.uuid4())
    task = analyze_report_task.apply_async(
        args=[report_id, file_path, report_type, extension],
        task_id=job_id,
    )
    insert_report(report_id, report_type, file_path, extension, task.id, patient_id)
    return {
        "job_id": task.id,
        "status": "queued",
        "poll_url": f"/jobs/{task.id}",
        "estimated_seconds": 20,
        "medical_disclaimer": MEDICAL_DISCLAIMER,
    }

