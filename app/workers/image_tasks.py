"""
VaidyaAI — Image Analysis Celery Tasks
Pipeline: image/DICOM → normalize → MedSAM → CheXNet → GradCAM → WHO RAG → structured report
"""

import logging
import time
from datetime import datetime, timezone

from celery import Task

from app.core.config import settings
from app.core.disclaimer import MEDICAL_DISCLAIMER
from app.ml.ml_prediction_engine import to_json_safe
from app.pipeline import run_image_pipeline
from app.workers.celery_app import celery_app
from app.workers.db_persist import mark_failed, persist_image_analysis

UTC = timezone.utc
logger = logging.getLogger(__name__)


class ImageAnalysisTask(Task):
    abstract = True

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        logger.error("Image task %s failed: %s", task_id, exc)


@celery_app.task(
    bind=True,
    base=ImageAnalysisTask,
    name="app.workers.image_tasks.analyze_image",
    soft_time_limit=settings.IMAGE_TASK_TIMEOUT,
    max_retries=2,
)
def analyze_image(self, analysis_id: str, file_path: str, image_type: str, file_format: str):
    """Run the production image pipeline and persist DB-compatible output."""
    start_time = time.time()

    try:
        self.update_state(state="PROGRESS", meta={"step": "pipeline", "pct": 10})
        response = run_image_pipeline(
            image_path=file_path,
            image_type=image_type,
            patient_id=None,
            job_id=analysis_id,
            clinical_context=f"format={file_format}",
        )
        payload = to_json_safe(response.model_dump())

        if payload.get("status") not in {"completed", "complete"}:
            raise ValueError(payload.get("error") or "Image pipeline did not complete")

        classification = payload.get("image_classification") or {}
        segmentation = payload.get("segmentation") or {}
        label = classification.get("label") or classification.get("top_class") or "unknown"
        confidence = payload.get("confidence_score") or classification.get("confidence") or 0.0
        if confidence and confidence <= 1:
            confidence = round(confidence * 100, 2)

        self.update_state(state="PROGRESS", meta={"step": "persist", "pct": 95})
        result = {
            "analysis_id": analysis_id,
            "image_type": image_type,
            "status": "complete",
            "dicom_metadata": {},
            "segmentation": {
                "mask_path": None,
                "overlay_path": payload.get("segmentation_overlay_path"),
                "roi_bounding_box": segmentation.get("bbox") or {},
                "confidence": segmentation.get("confidence", 0.0),
                "num_contours": segmentation.get("num_contours", 0),
                "mask_pixels": segmentation.get("mask_pixels", 0),
            },
            "classification": {
                "label": label,
                "top_class": label,
                "top_confidence": confidence,
                "probabilities": classification.get("probabilities", {}),
                "primary_finding": classification.get("primary_finding"),
                "icd10_code": (payload.get("label_metadata") or {}).get("icd10"),
                "urgency": (payload.get("label_metadata") or {}).get("urgency"),
                "body_region": (payload.get("label_metadata") or {}).get("body_region"),
            },
            "gradcam": {
                "heatmap_path": payload.get("gradcam_path"),
                "top_regions": (payload.get("who_structured_report") or {}).get("gradcam_regions", []),
            },
            "sources": payload.get("sources") or payload.get("radiology_evidence") or [],
            "radiology_evidence": payload.get("radiology_evidence", []),
            "who_report": payload.get("who_structured_report"),
            "explanation": payload.get("explanation") or "",
            "plain_language_summary": payload.get("plain_language_summary") or "",
            "confidence_score": confidence,
            "uncertainty_flag": payload.get("uncertainty_flag", True),
            "anomaly_detected": label not in {"No Finding", "No_Finding", "normal", "unknown"},
            "medical_disclaimer": MEDICAL_DISCLAIMER,
            "processing_time_ms": payload.get("processing_time_ms") or round((time.time() - start_time) * 1000, 2),
            "completed_at": payload.get("completed_at") or datetime.now(UTC).isoformat(),
        }

        persist_image_analysis(result)
        self.update_state(state="PROGRESS", meta={"step": "complete", "pct": 100})
        return result

    except Exception as exc:
        logger.error("analyze_image failed for %s: %s", analysis_id, exc)
        mark_failed("image_analyses", analysis_id, str(exc))
        raise self.retry(exc=exc, countdown=10)
