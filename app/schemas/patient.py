from pydantic import BaseModel, Field
from typing import Optional, Dict
from uuid import UUID
from datetime import datetime
from app.schemas.common import BaseResponse

class PatientCreate(BaseModel):
    patient_code: str = Field(..., description="Anonymized patient identifier")
    age: Optional[int] = Field(None, ge=0, le=150)
    gender: Optional[str] = None
    demographics: Dict = {}
    medical_history: Dict = {}

class PatientResponse(BaseResponse):
    id: UUID
    patient_code: str
    age: Optional[int]
    gender: Optional[str]
    created_at: datetime
