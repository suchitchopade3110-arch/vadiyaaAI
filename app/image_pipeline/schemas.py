from pydantic import BaseModel
from typing import List, Dict, Optional

class SegmentationOutput(BaseModel):
    mask_path: str           # saved UUID path
    contour_points: List     # [[x,y], ...]
    roi_path: str            # cropped ROI saved path
    confidence: float        # 0.0-1.0
    bbox: List               # [x1, y1, x2, y2]

class ClassificationOutput(BaseModel):
    label: str               # "Pneumothorax", "Normal", etc.
    confidence: float        # Platt-scaled 0-1
    probabilities: Dict      # {"Pneumothorax": 0.82, ...}
    gradcam_path: str        # heatmap image UUID path

class ImagePipelineOutput(BaseModel):
    segmentation: SegmentationOutput
    classification: ClassificationOutput
    explanation: str                       # from LLM (RAG layer)
    sources: List                          # ChromaDB citations
    uncertainty_flag: bool
    disclaimer: str = "AI-assisted analysis. NOT a medical diagnosis."
