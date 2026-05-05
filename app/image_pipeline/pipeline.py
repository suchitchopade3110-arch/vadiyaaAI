import uuid
import os
import cv2
import numpy as np

from app.services.preprocessor import preprocess_image_file, preprocess_dicom_file
from .segmentor import run_pipeline as run_segmentation_pipeline, get_predictor, SegmentationResult
from .quality_gate import is_valid_medical_image
from .classifier import run_chexnet
from .gradcam import generate_gradcam
from .schemas import ImagePipelineOutput, SegmentationOutput, ClassificationOutput

# Mocking the RAG layer call as requested
def run_image_explanation_chain(classification_label: str, chexnet_confidence: float, gradcam_regions: list) -> str:
    from app.services.rag_pipeline import rag_pipeline
    # Adaptive call to existing RAG service
    return rag_pipeline.explain_image(
        image_type="Chest X-Ray", # Default or passed from image_type
        classification={"label": classification_label, "confidence": chexnet_confidence},
        segmentation={}, # Placeholder
        sources=[] # Placeholder for ChromaDB citations
    )

def run_image_pipeline(image_path: str, image_type: str) -> ImagePipelineOutput:
    """
    Master orchestrator for the VaidyaAI image analysis pipeline.
    """
    
    # STEP 1 — Quality gate
    valid, reason = is_valid_medical_image(image_path)
    if not valid:
        raise ValueError(f"Quality gate rejected: {reason}")

    # STEP 2 — Load + preprocess
    if image_path.lower().endswith('.dcm'):
        prep_out = preprocess_dicom_file(image_path)
        raw_image = prep_out.pixel_array
    else:
        prep_out = preprocess_image_file(image_path)
        raw_image = prep_out.normalized

    # STEP 3 & 4 — Full Segmentation Pipeline
    predictor = get_predictor()
    seg_result: SegmentationResult = run_segmentation_pipeline(raw_image, predictor, source_path=image_path)
    
    if seg_result.roi_crop is None:
        raise ValueError(f"Failed to extract ROI from image. Error: {seg_result.metadata.get('error', 'Unknown')}")

    mask = seg_result.segmentation_mask
    bbox = seg_result.bbox
    contours = seg_result.contours
    confidence = seg_result.confidence
    roi = seg_result.roi_crop
    
    # Simulate saving mask + roi → UUID paths
    # In real app, use app.core.file_storage
    mask_uuid = str(uuid.uuid4())
    roi_uuid = str(uuid.uuid4())
    mask_path = f"uploads/masks/{mask_uuid}.png"
    roi_path = f"uploads/rois/{roi_uuid}.png"
    
    # Ensure directories exist (simulated for now)
    os.makedirs("uploads/masks", exist_ok=True)
    os.makedirs("uploads/rois", exist_ok=True)
    
    # Save files (mocked or real save logic)
    cv2.imwrite(mask_path, mask * 255)
    cv2.imwrite(roi_path, cv2.cvtColor(roi, cv2.COLOR_RGB2BGR))

    # STEP 5 — Classify ROI (CheXNet)
    classification_res = run_chexnet(roi)

    # STEP 6 — GradCAM
    gradcam_path = generate_gradcam(roi, classification_res.label)
    # Update classification result with gradcam path if needed
    classification_res.gradcam_path = gradcam_path

    # STEP 7 — LLM explanation (call existing RAG layer)
    explanation = run_image_explanation_chain(
        classification_label=classification_res.label,
        chexnet_confidence=classification_res.confidence,
        gradcam_regions=[]   # pass highlighted regions if available
    )

    # Construct Final Output
    segmentation_out = SegmentationOutput(
        mask_path=mask_path,
        contour_points=contours,
        roi_path=roi_path,
        confidence=confidence,
        bbox=bbox
    )
    
    classification_out = ClassificationOutput(
        label=classification_res.label,
        confidence=classification_res.confidence,
        probabilities=classification_res.probabilities,
        gradcam_path=gradcam_path
    )

    return ImagePipelineOutput(
        segmentation=segmentation_out,
        classification=classification_out,
        explanation=explanation,
        sources=[], # TODO: ChromaDB citations
        uncertainty_flag=(confidence < 0.5 or classification_res.confidence < 0.5),
        disclaimer="AI-assisted analysis. NOT a medical diagnosis."
    )
