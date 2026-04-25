from pydantic import BaseModel
from uuid import UUID
from datetime import datetime
from app.models.report import ReportType
from app.core.config import settings


class ReportResponse(BaseModel):
    id: UUID
    report_type: ReportType
    confidence: float | None = None
    parsed_data: dict | None = None
    analysis_result: dict | None = None
    source_citations: list | None = None
    uncertainty_flag: bool
    celery_task_id: str | None = None
    medical_disclaimer: str = settings.MEDICAL_DISCLAIMER
    created_at: datetime

    model_config = {"from_attributes": True}


class ReportStatusResponse(BaseModel):
    id: UUID
    celery_task_id: str | None = None
    medical_disclaimer: str = settings.MEDICAL_DISCLAIMER
