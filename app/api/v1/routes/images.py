import uuid
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from celery.result import AsyncResult
from datetime import UTC, datetime

from app.db.session import get_db
from app.schemas.image import ImageAsyncResponse, AnalysisType
from app.schemas.job import JobStatus
from app.services.file_upload import file_upload_service
from app.workers.image_tasks import analyze_image
from app.workers.celery_app import celery_app
from app.core.disclaimer import MEDICAL_DISCLAIMER

router = APIRouter()

VALID_IMAGE_TYPES = {t.value for t in AnalysisType}

@router.post("/{analysis_type}", response_model=ImageAsyncResponse, status_code=status.HTTP_202_ACCEPTED)
async def submit_image_analysis(
    analysis_type: str,
    file: UploadFile = File(..., description="Medical image: DICOM, JPG, PNG"),
    patient_id: str = None,
    db: AsyncSession = Depends(get_db),
):
    """
    Submit medical image for analysis.
    """
    if analysis_type not in VALID_IMAGE_TYPES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid analysis_type '{analysis_type}'. Valid: {VALID_IMAGE_TYPES}",
        )

    request_id = str(uuid.uuid4())
    analysis_id = str(uuid.uuid4())

    # Save uploaded file
    file_path, extension, pipeline_type = await file_upload_service.validate_and_save(
        file, sub_dir=f"images/{analysis_id}"
    )

    if pipeline_type != "image":
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"File type '.{extension}' is not a medical image format.",
        )

    # Dispatch Celery task
    task = analyze_image.apply_async(
        args=[analysis_id, file_path, analysis_type, extension],
    )

    return ImageAsyncResponse(
        request_id=request_id,
        analysis_id=uuid.UUID(analysis_id),
        id=task.id, # Use task_id for consistency in polling
        task_id=task.id,
        status="pending",
        analysis_type=AnalysisType(analysis_type),
        poll_url=f"/api/v1/analyze/image/{task.id}",
        estimated_seconds=45,
        medical_disclaimer=MEDICAL_DISCLAIMER,
    )


@router.get("/status/{task_id}", response_model=JobStatus)
async def get_image_status(task_id: str):
    """Poll image analysis task status."""
    request_id = str(uuid.uuid4())
    result = AsyncResult(task_id, app=celery_app)
    state = result.state
    meta = result.info or {}

    if state == "PENDING":
        return JobStatus(request_id=request_id, task_id=task_id, status="pending")
    if state == "PROGRESS":
        return JobStatus(
            request_id=request_id,
            task_id=task_id,
            status="processing",
            progress_pct=meta.get("pct", 0),
        )
    if state == "SUCCESS":
        return JobStatus(
            request_id=request_id,
            task_id=task_id,
            status="complete",
            progress_pct=100,
            result_url=f"/api/v1/analyze/image/{task_id}",
        )
    if state == "FAILURE":
        return JobStatus(request_id=request_id, task_id=task_id, status="failed", error=str(meta))
    return JobStatus(request_id=request_id, task_id=task_id, status=state.lower())


@router.get("/image/{analysis_id}")
async def get_image_status_or_result(analysis_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    """Poll image analysis job status and get result when complete."""
    from app.services.image_service import ImageService
    
    service = ImageService(db)
    image_record = await service.get(analysis_id)
    
    if not image_record:
        raise HTTPException(status_code=404, detail="Analysis not found")
        
    request_id = str(uuid.uuid4())
    
    if image_record.status.value in ["pending", "processing"]:
        if image_record.celery_task_id:
            result = AsyncResult(image_record.celery_task_id, app=celery_app)
            state = result.state
            meta = result.info or {}
            
            if state == "SUCCESS":
                return {"status": "processing"} 
            elif state == "FAILURE":
                 return JobStatus(
                    request_id=request_id,
                    task_id=image_record.celery_task_id,
                    status="failed",
                    error=str(meta),
                )
        return JobStatus(request_id=request_id, task_id=image_record.celery_task_id or "", status=image_record.status.value)
    
    confidence_data = {
        "score": image_record.confidence or 0.0,
        "color": "green" if image_record.confidence and image_record.confidence > 80 else "yellow",
        "label": "High" if image_record.confidence and image_record.confidence > 80 else "Medium",
        "uncertainty_flag": image_record.uncertainty_flag,
    }
    
    # Map to frontend structure
    findings = []
    if image_record.classification:
        for k, v in image_record.classification_probs.items() if image_record.classification_probs else {image_record.classification: image_record.confidence}:
            findings.append({"label": k, "confidence": v, "severity": "Moderate" if v > 50 else "Mild"})

    return {
        "request_id": request_id,
        "medical_disclaimer": MEDICAL_DISCLAIMER,
        "timestamp": datetime.now(UTC).isoformat(),
        "analysis_id": str(analysis_id),
        "id": str(analysis_id),
        "status": image_record.status.value,
        "type": image_record.image_type.value,
        "impression": image_record.explanation or "",
        "explanation": image_record.explanation or "",
        "findings": findings,
        "roi": image_record.roi_metadata or [],
        "confidence": confidence_data,
        "uncertainty": image_record.uncertainty_flag,
        "anomaly_detected": image_record.anomaly_detected,
        "citations": image_record.source_citations or [],
        "sources": image_record.source_citations or [],
        "gradcam": {
            "heatmap_url": image_record.gradcam_path,
            "top_regions": image_record.gradcam_regions or [],
        },
        "segmentation": {
            "mask_url": image_record.segmentation_mask_path,
            "overlay_url": image_record.segmentation_overlay,
            "roi_bounding_box": image_record.roi_metadata,
            "confidence": image_record.segmentation_confidence or 0.0
        },
        "classification": {
            "label": image_record.classification or "unknown",
            "probabilities": image_record.classification_probs or {},
            "top_class": image_record.classification or "unknown",
            "top_confidence": image_record.confidence or 0.0
        }
    }


@router.get("/{task_id}")
async def get_image_combined(task_id: str):
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
