from pydantic import BaseModel
from typing import Optional, List, Dict
from uuid import UUID
from enum import Enum
from app.schemas.common import BaseResponse, ConfidenceSignal, SourceCitation

class AnalysisType(str, Enum):
    XRAY      = "xray"
    CT        = "ct"
    MRI       = "mri"
    SKIN      = "skin"
    PATHOLOGY = "pathology"

class ImageAsyncResponse(BaseResponse):
    analysis_id: UUID
    task_id: str
    status: str = "pending"
    analysis_type: AnalysisType
    poll_url: str
    estimated_seconds: int = 45

class SegmentationResult(BaseModel):
    mask_url: Optional[str] = None
    overlay_url: Optional[str] = None
    roi_bounding_box: Optional[Dict] = None
    confidence: float

class ClassificationResult(BaseModel):
    label: str
    probabilities: Dict[str, float]
    top_class: str
    top_confidence: float

class GradCAMResult(BaseModel):
    heatmap_url: Optional[str] = None
    top_regions: List[Dict]

class ImageResult(BaseResponse):
    analysis_id: UUID
    image_type: AnalysisType
    segmentation: SegmentationResult
    classification: ClassificationResult
    gradcam: GradCAMResult
    explanation: str
    sources: List[SourceCitation]
    confidence: ConfidenceSignal
    anomaly_detected: bool
    dicom_metadata: Optional[Dict] = None
    processing_time_ms: Optional[float] = None
