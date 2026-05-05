import logging
from typing import Any

from fastapi import HTTPException, UploadFile, status

from app.core.disclaimer import MEDICAL_DISCLAIMER
from app.services.preprocessor import run_ner
from app.services.file_upload import file_upload_service
from app.services.ml_service import ml_predictor
from app.services.rag_pipeline import rag_pipeline

logger = logging.getLogger(__name__)


def run_text_pipeline(query: str) -> dict[str, Any]:
    """
    Synchronous text integration for the simple /api/text contract.

    This connects the existing Phase 2-shaped services without changing the
    versioned async API:
    ClinicalBERT entities -> ML risk/SHAP -> RAG evidence/explanation.
    """
    ner_result = run_ner(query)
    feature_dict = {lv.name: {"value": lv.value, "unit": lv.unit} for lv in ner_result.lab_values}
    entities = {
        "conditions": ner_result.conditions,
        "medications": ner_result.medications,
        "lab_values": feature_dict,
        "procedures": [],
        "_source": "clinicalbert-ner"
    }
    prediction = ml_predictor.predict_risk(entities)
    anomalies = ml_predictor.detect_anomalies(entities.get("lab_values", {}))
    evidence = rag_pipeline.retrieve_evidence(query)

    explanation = rag_pipeline.explain_report(
        entities=entities,
        risk_score=prediction["risk_score"],
        risk_factors=prediction["risk_factors"],
        anomalies=anomalies,
        sources=evidence,
    )

    return {
        "diagnosis": _risk_label(prediction["risk_score"]),
        "confidence": prediction["risk_score"],
        "evidence": evidence,
        "explanation": explanation,
        "shap": prediction["shap_values"],
        "entities": entities,
        "risk_factors": prediction["risk_factors"],
        "anomalies": anomalies,
    }


async def run_image_pipeline(file: UploadFile) -> dict[str, Any]:
    """
    Synchronous image integration for the simple /api/image contract.

    The full production image workflow already exists as a Celery pipeline under
    /api/v1/analyze/image. This endpoint validates and stores the file, then
    returns the same Phase 2 response surface expected by the lightweight UI.
    """
    file_path, extension, pipeline_type = await file_upload_service.validate_and_save(
        file, sub_dir="images/direct"
    )

    if pipeline_type != "image":
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"File type '.{extension}' is not a medical image format.",
        )

    quality = file_upload_service.quality_score(file_path, extension)
    classification = {
        "label": "pending",
        "top_class": "pending",
        "probabilities": {},
        "top_confidence": 0.0,
        "_note": "CheXNet/ResNet integration point",
    }
    segmentation = {
        "mask_path": None,
        "overlay_path": None,
        "roi_bounding_box": {},
        "confidence": 0.0,
        "_note": "LiteMedSAM integration point",
    }
    evidence = rag_pipeline.retrieve_evidence(file.filename or "medical image")
    explanation = rag_pipeline.explain_image(
        image_type=extension,
        classification=classification,
        segmentation=segmentation,
        sources=evidence,
    )

    return {
        "diagnosis": classification["label"],
        "confidence": classification["top_confidence"],
        "heatmap": None,
        "mask": segmentation["mask_path"],
        "segmentation": segmentation,
        "classification": classification,
        "evidence": evidence,
        "explanation": explanation,
        "quality_score": quality,
        "stored_file_path": file_path,
        "medical_disclaimer": MEDICAL_DISCLAIMER,
    }


def _risk_label(score: float) -> str:
    if score >= 70:
        return "high_risk"
    if score >= 35:
        return "moderate_risk"
    return "low_risk"

