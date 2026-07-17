import uuid

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status, Request

from app.core.auth import get_current_user
from app.core.disclaimer import MEDICAL_DISCLAIMER
from app.services.file_upload import file_upload_service
from app.workers.db_persist import insert_image_analysis
from app.workers.image_tasks import analyze_image as analyze_image_task
from app.main import limiter

router = APIRouter()

VALID_TYPES = {"xray", "ct", "mri", "skin", "pathology"}


@router.post("/image/{analysis_type}", status_code=202)
@limiter.limit("10/minute")
async def analyze_image(
    request: Request,
    analysis_type: str,
    file: UploadFile = File(...),
    user=Depends(get_current_user),
):
    if analysis_type not in VALID_TYPES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid type. Use: {sorted(VALID_TYPES)}",
        )

    analysis_id = str(uuid.uuid4())
    file_path, extension, pipeline_type = await file_upload_service.validate_and_save(
        file, sub_dir=f"images/{analysis_id}"
    )
    if pipeline_type != "image":
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"File type '.{extension}' is not a medical image format.",
        )

    job_id = str(uuid.uuid4())
    task = analyze_image_task.apply_async(
        args=[analysis_id, file_path, analysis_type, extension],
        task_id=job_id,
    )
    insert_image_analysis(analysis_id, analysis_type, file_path, extension, task.id)
    return {
        "job_id": task.id,
        "status": "queued",
        "poll_url": f"/jobs/{task.id}",
        "estimated_seconds": 30,
        "medical_disclaimer": MEDICAL_DISCLAIMER,
    }
