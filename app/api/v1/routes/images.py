import uuid
from pathlib import Path

import aiofiles
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status, Depends
from celery.result import AsyncResult
from datetime import timezone, datetime
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import get_current_user
from app.db.session import get_db

UTC = timezone.utc

from app.schemas.image import ImageAsyncResponse, AnalysisType
from app.schemas.job import JobStatus
from app.workers.pipeline_tasks import analyze_image_task
from app.workers.celery_app import celery_app
from app.core.disclaimer import MEDICAL_DISCLAIMER

router = APIRouter()

VALID_IMAGE_TYPES = {t.value for t in AnalysisType}
IMAGE_TYPES = {"xray", "ct", "mri", "skin", "pathology"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".dcm"}
IMAGE_MAX_BYTES = {
    ".dcm": 50 * 1024 * 1024,
    ".jpg": 10 * 1024 * 1024,
    ".jpeg": 10 * 1024 * 1024,
    ".png": 10 * 1024 * 1024,
}
UPLOAD_ROOT = Path("uploads").resolve()


async def _save_image_upload(file: UploadFile, job_id: str) -> tuple[str, str]:
    ext = Path(file.filename or "").suffix.lower()
    if ext not in IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail={
                "code": "FILE_FORMAT_UNSUPPORTED",
                "message": f"Extension '{ext}' not allowed. Allowed: {sorted(IMAGE_EXTENSIONS)}",
            },
        )

    content = await file.read()
    max_bytes = IMAGE_MAX_BYTES[ext]
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={
                "code": "FILE_TOO_LARGE",
                "message": f"File exceeds {max_bytes // 1024 // 1024}MB limit",
            },
        )

    save_dir = UPLOAD_ROOT / "images"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{job_id}{ext}"
    async with aiofiles.open(save_path, "wb") as handle:
        await handle.write(content)
    return str(save_path), ext.lstrip(".")


@router.post("/{analysis_type}", status_code=status.HTTP_202_ACCEPTED)
async def submit_image_analysis(
    analysis_type: str,
    file: UploadFile = File(..., description="Medical image: DICOM, JPG, PNG"),
    patient_id: str = Form(None),
    clinical_context: str = Form(""),
    user=Depends(get_current_user),
):
    """
    Submit medical image for analysis.
    """
    if analysis_type not in VALID_IMAGE_TYPES and analysis_type not in IMAGE_TYPES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "code": "INVALID_IMAGE_TYPE",
                "message": f"image_type must be xray|ct|mri|skin|pathology, got '{analysis_type}'",
            },
        )

    job_id = str(uuid.uuid4())
    file_path, _extension = await _save_image_upload(file, job_id)

    # Dispatch Celery task
    task = analyze_image_task.apply_async(
        args=[file_path, analysis_type, patient_id, clinical_context],
        task_id=job_id,
        queue="images",
    )

    return {
        "job_id": task.id,
        "task_id": task.id,
        "status": "queued",
        "poll_url": f"/api/v1/jobs/{task.id}",
        "estimated_seconds": 30,
    }


@router.get("/status/{task_id}", response_model=JobStatus)
async def get_image_status(task_id: str, user=Depends(get_current_user)):
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
async def get_image_status_or_result(
    analysis_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    user=Depends(get_current_user),
):
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

    from app.services.fix_image_analysis import build_classification_findings, get_severity, get_severity_color

    score = image_record.confidence or 0.0
    severity = get_severity(score)
    
    confidence_data = {
        "score": score,
        "color": get_severity_color(severity),
        "label": severity,
        "uncertainty_flag": image_record.uncertainty_flag,
    }
    
    # Map to frontend structure
    findings = []
    if image_record.classification:
        raw_predictions = [
            {
                "label": k,
                "classification_prob": v,
                "detection_confidence": v,
            }
            for k, v in (image_record.classification_probs.items() if image_record.classification_probs else {image_record.classification: score}.items())
        ]
        findings = build_classification_findings(raw_predictions)
    yolo_detections = image_record.roi_metadata if isinstance(image_record.roi_metadata, list) else []

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
        "yolo": {
            "detections": yolo_detections,
            "annotated_path": image_record.segmentation_overlay,
            "model_used": "chest_xray_yolo" if yolo_detections else None,
        },
        "yolo_detections": yolo_detections,
        "yolo_annotated_path": image_record.segmentation_overlay,
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
            "yolo_overlay_url": image_record.segmentation_overlay,
            "roi_bounding_box": image_record.roi_metadata,
            "confidence": image_record.segmentation_confidence or 0.0
        },
        "classification": {
            "label": image_record.classification or "unknown",
            "probabilities": image_record.classification_probs or {},
            "top_class": image_record.classification or "unknown",
            "top_confidence": score,
            "classification_prob": score,
            "detection_confidence": score,
            "label_classification": "Class probability",
            "label_detection": "Detection confidence",
            "severity": severity,
            "severity_color": get_severity_color(severity),
        }
    }


@router.get("/{task_id}")
async def get_image_combined(task_id: str, user=Depends(get_current_user)):
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
