from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from uuid import UUID
from enum import Enum
from app.schemas.common import BaseResponse, ConfidenceSignal, SourceCitation, ExtractedEntities

class VerdictEnum(str, Enum):
    VERIFIED  = "verified"
    REFUTED   = "refuted"
    UNCERTAIN = "uncertain"

class ClaimRequest(BaseModel):
    claim_text: str = Field(..., min_length=10, max_length=5000,
                            description="Medical claim text to verify")
    patient_id: Optional[UUID] = None
    priority: str = Field(default="normal", description="normal | high")

class ClaimAsyncResponse(BaseResponse):
    claim_id: UUID
    task_id: str
    id: str  # Added for frontend compatibility
    status: str = "pending"
    poll_url: str
    estimated_seconds: int = 15

class ClaimResult(BaseResponse):
    claim_id: UUID
    claim_text: str
    extracted_entities: ExtractedEntities
    verdict: VerdictEnum
    explanation: str
    sources: List[SourceCitation]
    confidence: ConfidenceSignal
    hallucination_detected: bool
    hallucination_details: Optional[Dict] = None
    shap_values: Optional[Dict] = None
    processing_time_ms: Optional[float] = None
