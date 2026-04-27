from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from uuid import UUID
from enum import Enum
from app.schemas.common import BaseResponse, ConfidenceSignal, SourceCitation, ExtractedEntities

class ReportTypeEnum(str, Enum):
    LAB       = "lab"
    CLINICAL  = "clinical"
    DISCHARGE = "discharge"

class ReportAsyncResponse(BaseResponse):
    report_id: UUID
    task_id: str
    id: str  # Added for frontend compatibility
    status: str = "pending"
    report_type: ReportTypeEnum
    poll_url: str
    estimated_seconds: int = 20

class AnomalyFlag(BaseModel):
    parameter: str
    value: Any
    reference_range: str
    severity: str   # mild | moderate | severe

class ReportResult(BaseResponse):
    report_id: UUID
    report_type: ReportTypeEnum
    extracted_entities: ExtractedEntities
    raw_text_preview: Optional[str] = None
    risk_score: float = Field(..., ge=0, le=100)
    risk_factors: List[str]
    anomalies: List[AnomalyFlag]
    shap_values: Optional[Dict] = None
    explanation: str
    sources: List[SourceCitation]
    confidence: ConfidenceSignal
    processing_time_ms: Optional[float] = None
