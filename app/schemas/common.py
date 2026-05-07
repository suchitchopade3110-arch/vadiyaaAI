from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import timezone, datetime
UTC = timezone.utc

class BaseResponse(BaseModel):
    request_id: str
    medical_disclaimer: str = Field(
        default="⚠️ AI-ASSISTED ANALYSIS — NOT A MEDICAL DIAGNOSIS. "
                "Consult a qualified healthcare professional for medical decisions."
    )
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

class ConfidenceSignal(BaseModel):
    score: float = Field(..., ge=0, le=100, description="Platt-scaled 0–100")
    color: str   = Field(..., description="green | yellow | red")
    label: str   = Field(..., description="High | Medium | Low | Insufficient")
    uncertainty_flag: bool = False
    message: Optional[str] = None

    @classmethod
    def from_score(cls, score: float, source_count: int = 0) -> "ConfidenceSignal":
        if source_count < 2 or score < 30:
            return cls(score=score, color="red", label="Insufficient",
                      uncertainty_flag=True,
                      message="⚠️ INSUFFICIENT EVIDENCE: Fewer than 2 sources. HIGH uncertainty.")
        elif score >= 80:
            return cls(score=score, color="green", label="High")
        elif score >= 50:
            return cls(score=score, color="yellow", label="Medium")
        else:
            return cls(score=score, color="red", label="Low", uncertainty_flag=True)

class SourceCitation(BaseModel):
    source_id: str
    title: str
    excerpt: str
    relevance_score: float
    url: Optional[str] = None
    publication: Optional[str] = None

class ExtractedEntities(BaseModel):
    conditions:  List[str] = []
    medications: List[str] = []
    lab_values:  Dict[str, Any] = {}
    procedures:  List[str] = []
