from pydantic import BaseModel
from uuid import UUID
from datetime import datetime
from app.models.image_analysis import ImageType
from app.core.config import settings


class ImageAnalysisResponse(BaseModel):
    id: UUID
    image_type: ImageType
    confidence: float | None = None
    classification: dict | None = None
    roi_metadata: dict | None = None
    segmentation_mask_path: str | None = None
    gradcam_path: str | None = None
    source_citations: list | None = None
    uncertainty_flag: bool
    celery_task_id: str | None = None
    medical_disclaimer: str = settings.MEDICAL_DISCLAIMER
    created_at: datetime

    model_config = {"from_attributes": True}


class ImageStatusResponse(BaseModel):
    id: UUID
    celery_task_id: str | None = None
    medical_disclaimer: str = settings.MEDICAL_DISCLAIMER
