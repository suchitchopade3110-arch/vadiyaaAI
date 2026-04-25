from pydantic import BaseModel, Field
from uuid import UUID
from typing import Any
from datetime import datetime
from app.models.claim import ClaimStatus
from app.core.config import settings


class ClaimCreateRequest(BaseModel):
    claim_text: str = Field(..., min_length=10, max_length=5000)
    patient_id: UUID | None = None

    model_config = {"json_schema_extra": {
        "example": {
            "claim_text": "Patient has Type 2 Diabetes with HbA1c of 9.2% and hypertension.",
            "patient_id": None,
        }
    }}


class ClaimResponse(BaseModel):
    id: UUID
    status: ClaimStatus
    claim_text: str
    confidence: float | None = None
    extracted_entities: dict | None = None
    source_citations: list | None = None
    shap_values: dict | None = None
    uncertainty_flag: bool
    celery_task_id: str | None = None
    medical_disclaimer: str = settings.MEDICAL_DISCLAIMER
    created_at: datetime

    model_config = {"from_attributes": True}


class ClaimStatusResponse(BaseModel):
    id: UUID
    status: ClaimStatus
    celery_task_id: str | None = None
    medical_disclaimer: str = settings.MEDICAL_DISCLAIMER
