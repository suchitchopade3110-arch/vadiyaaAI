"""
pipeline.py - VaidyaAI Master Orchestrator
==========================================
Connects the PRD layers into unified image, report, claim, and smart-router
entry points.
"""

import json
import logging
import os
import time
import traceback
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from pydantic import BaseModel, Field
except Exception:  # pragma: no cover - only used in minimal self-test envs
    class BaseModel:
        def __init__(self, **kwargs):
            annotations = getattr(self.__class__, "__annotations__", {})
            for key, value in self.__class__.__dict__.items():
                if key.startswith("_") or callable(value) or isinstance(value, property):
                    continue
                if key not in kwargs and key in annotations:
                    setattr(self, key, value)
            for key, value in kwargs.items():
                setattr(self, key, value)

        def model_dump(self):
            return dict(self.__dict__)

    def Field(default_factory=None, default=None):
        if default_factory is not None:
            return default_factory()
        return default

from app.ml.ml_prediction_engine import (
    classify_image,
    ensemble_predict,
    generate_gradcam_overlay,
    predict_tabular,
    to_json_safe,
)

logger = logging.getLogger("vaidya.pipeline")


# ============================================================================
# CONFIGURATION
# ============================================================================

DISCLAIMER = (
    "AI-assisted analysis only. NOT A MEDICAL DIAGNOSIS. "
    "Consult a qualified healthcare professional for clinical decisions."
)

ARTIFACTS_DIR = Path(os.getenv("VAIDYA_ARTIFACTS_DIR", "/tmp/vaidya/artifacts"))
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# UNIFIED OUTPUT CONTRACT
# ============================================================================

class PipelineResponse(BaseModel):
    """Unified response from any pipeline. Backend serializes to API JSON."""

    job_id: str
    patient_id: Optional[str] = None
    pipeline_type: str

    status: str
    error: Optional[str] = None

    tabular_prediction: Optional[Dict[str, Any]] = None
    image_classification: Optional[Dict[str, Any]] = None
    ensemble: Optional[Dict[str, Any]] = None

    segmentation: Optional[Dict[str, Any]] = None

    extracted_entities: Optional[Dict[str, Any]] = None
    lab_values: List[Dict[str, Any]] = Field(default_factory=list)
    anomalies: List[Dict[str, Any]] = Field(default_factory=list)

    verdict: Optional[str] = None
    confidence_score: Optional[float] = None
    confidence_label: Optional[str] = None
    explanation: Optional[str] = None
    plain_language_summary: Optional[str] = None
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    uncertainty_flag: bool = False
    hallucination_flagged: bool = False

    shap_top_factors: List[Dict[str, Any]] = Field(default_factory=list)
    gradcam_path: Optional[str] = None
    segmentation_overlay_path: Optional[str] = None

    disclaimer: str = DISCLAIMER

    processing_time_ms: float = 0.0
    completed_at: str = ""


# ============================================================================
# CONTEXT + FALLBACK DATA SHAPES
# ============================================================================

@dataclass
class PipelineContext:
    """Shared context across all pipelines for tracing."""

    job_id: str
    patient_id: Optional[str]
    pipeline_type: str
    start_time: float

    @property
    def elapsed_ms(self) -> float:
        return round((time.time() - self.start_time) * 1000, 2)


@dataclass
class _ValidationResult:
    valid: bool = True
    reason: str = ""


@dataclass
class _NerResult:
    conditions: List[str]
    medications: List[str]
    dates: List[str]
    lab_values: List[Any]


def _make_context(
    pipeline_type: str,
    patient_id: Optional[str] = None,
    job_id: Optional[str] = None,
) -> PipelineContext:
    return PipelineContext(
        job_id=job_id or str(uuid.uuid4()),
        patient_id=patient_id,
        pipeline_type=pipeline_type,
        start_time=time.time(),
    )


def _build_error_response(
    ctx: PipelineContext,
    error: str,
    status: str = "failed",
) -> PipelineResponse:
    logger.error("[%s] %s pipeline %s: %s", ctx.job_id, ctx.pipeline_type, status, error)
    return PipelineResponse(
        job_id=ctx.job_id,
        patient_id=ctx.patient_id,
        pipeline_type=ctx.pipeline_type,
        status=status,
        error=error,
        processing_time_ms=ctx.elapsed_ms,
        completed_at=datetime.now(timezone.utc).isoformat(),
    )


def _confidence_label(score: float) -> str:
    if score > 75:
        return "HIGH"
    if score > 50:
        return "MEDIUM"
    return "LOW"


# ============================================================================
# LAZY LAYER WRAPPERS
# ============================================================================

def _validate_file_safe(file_path: str) -> _ValidationResult:
    try:
        from app.services.preprocessor import validate_file

        return validate_file(file_path)
    except Exception as exc:
        if os.path.exists(file_path):
            logger.warning("Validation fallback for %s: %s", file_path, exc)
            return _ValidationResult(valid=True, reason="")
        return _ValidationResult(valid=False, reason=f"File not found: {file_path}")


def _run_ocr_safe(file_path: str):
    try:
        from app.services.preprocessor import run_ocr

        return run_ocr(file_path)
    except Exception as exc:
        logger.warning("OCR unavailable for %s: %s", file_path, exc)
        return type("OcrFallback", (), {"raw_text": "", "confidence": 0.0, "pages": []})()


def _run_ner_safe(text: str) -> _NerResult:
    try:
        from app.services.preprocessor import run_ner

        return run_ner(text)
    except Exception as exc:
        logger.warning("NER fallback: %s", exc)
        return _NerResult(conditions=[], medications=[], dates=[], lab_values=[])


def _preprocess_image_safe(image_path: str):
    from app.services.preprocessor import preprocess_image_file

    return preprocess_image_file(image_path)


def _preprocess_dicom_safe(image_path: str):
    from app.services.preprocessor import preprocess_dicom_file

    return preprocess_dicom_file(image_path)


def _extract_lab_values_safe(raw_text: str, ner_result: Any, report_type: str) -> List[Any]:
    values = list(getattr(ner_result, "lab_values", []) or [])
    if values or report_type != "lab":
        return values
    try:
        from app.services.preprocessor import UniversalValueExtractor

        return UniversalValueExtractor().extract_all(raw_text)
    except Exception as exc:
        logger.warning("Structured lab extraction fallback: %s", exc)
        return values


def _run_segmentation_safe(image_array: Any, ctx: PipelineContext):
    try:
        from app.services.segmentation.pipeline_runner import pipeline as run_segmentation

        return run_segmentation(image_array, model_predictor=None, modality="X-Ray")
    except Exception as exc:
        logger.warning("[%s] Segmentation unavailable, using full image: %s", ctx.job_id, exc)
        return None


def _rag_service():
    try:
        from app.services.rag_pipeline import rag_pipeline

        return rag_pipeline
    except Exception as exc:
        logger.warning("RAG unavailable: %s", exc)
        return None


def _explain_report_safe(
    raw_text: str,
    entities: Dict[str, Any],
    risk_score: float,
    risk_factors: List[Dict[str, Any]],
    anomalies: List[Dict[str, Any]],
    report_type: str,
) -> Dict[str, Any]:
    rag = _rag_service()
    if rag is None:
        return {
            "verdict": "Uncertain",
            "confidence_score": 50.0,
            "explanation": "ML analysis completed. Evidence retrieval is currently unavailable.",
            "plain_language_summary": None,
            "sources": [],
            "uncertainty_flag": True,
            "hallucination_flagged": False,
        }
    try:
        sources = rag.retrieve_evidence(raw_text[:500])
        explanation = rag.explain_report(
            entities=entities,
            risk_score=risk_score,
            risk_factors=risk_factors,
            anomalies=anomalies,
            sources=sources,
            report_type=report_type,
        )
        return {
            "verdict": "Uncertain",
            "confidence_score": risk_score if risk_score else 50.0,
            "explanation": explanation,
            "plain_language_summary": explanation,
            "sources": sources,
            "uncertainty_flag": not bool(sources),
            "hallucination_flagged": False,
        }
    except Exception as exc:
        logger.warning("Report RAG fallback: %s", exc)
        return {
            "verdict": "Uncertain",
            "confidence_score": risk_score if risk_score else 50.0,
            "explanation": "Report analysis available but evidence retrieval is pending.",
            "plain_language_summary": None,
            "sources": [],
            "uncertainty_flag": True,
            "hallucination_flagged": False,
        }


def _verify_claim_safe(claim_text: str, entities: Dict[str, Any]) -> Dict[str, Any]:
    rag = _rag_service()
    if rag is None:
        return {
            "verdict": "Uncertain",
            "confidence_score": 50.0,
            "explanation": "Insufficient evidence for verification.",
            "sources": [],
            "uncertainty_flag": True,
            "hallucination_flagged": False,
        }
    try:
        sources = rag.retrieve_evidence(claim_text)
        verdict = rag.verify_claim(claim_text, entities, sources)
        return {
            "verdict": str(verdict.get("verdict", "Uncertain")).title(),
            "confidence_score": 50.0 if not sources else 70.0,
            "explanation": verdict.get("explanation"),
            "sources": sources,
            "uncertainty_flag": len(sources) == 0,
            "hallucination_flagged": bool(verdict.get("hallucination_detected", False)),
        }
    except Exception as exc:
        logger.warning("Claim RAG fallback: %s", exc)
        return {
            "verdict": "Uncertain",
            "confidence_score": 50.0,
            "explanation": "Insufficient evidence for verification.",
            "sources": [],
            "uncertainty_flag": True,
            "hallucination_flagged": False,
        }


def _ner_entities_dict(ner_result: Any) -> Dict[str, Any]:
    return {
        "conditions": list(getattr(ner_result, "conditions", []) or [])[:10],
        "medications": list(getattr(ner_result, "medications", []) or [])[:10],
        "dates": list(getattr(ner_result, "dates", []) or [])[:5],
    }


def _lab_value_to_row(lab_value: Any) -> Dict[str, Any]:
    name = getattr(lab_value, "name", None)
    value = getattr(lab_value, "value", None)
    unit = getattr(lab_value, "unit", "")
    if isinstance(lab_value, dict):
        name = lab_value.get("name") or lab_value.get("test")
        value = lab_value.get("value") if lab_value.get("value") is not None else lab_value.get("result")
        unit = lab_value.get("unit", "")
    return {
        "test": name,
        "result": value,
        "unit": unit,
        "reference": "--",
        "flag": "NORMAL",
    }


def _lab_values_to_dict(lab_values: List[Any], gender: str, age: int) -> Dict[str, Any]:
    lab_dict = {}
    for lab_value in lab_values:
        row = _lab_value_to_row(lab_value)
        if row["test"] is None or row["result"] is None:
            continue
        key = str(row["test"]).lower().replace(" ", "_")
        lab_dict[key] = row["result"]
    lab_dict["gender"] = 1 if gender.lower() == "male" else 0
    lab_dict["age"] = age
    return lab_dict


# ============================================================================
# PIPELINE 1 - IMAGE ANALYSIS
# ============================================================================

def run_image_pipeline(
    image_path: str,
    image_type: str = "xray",
    patient_id: Optional[str] = None,
    job_id: Optional[str] = None,
    clinical_context: str = "",
) -> PipelineResponse:
    """DICOM/JPG/PNG -> MedSAM -> CheXNet -> GradCAM -> optional RAG explanation."""
    ctx = _make_context("image", patient_id, job_id)
    logger.info("[%s] Starting image pipeline: %s", ctx.job_id, image_path)

    try:
        validation = _validate_file_safe(image_path)
        if not validation.valid:
            return _build_error_response(ctx, f"Validation failed: {validation.reason}")

        ext = Path(image_path).suffix.lower()
        if ext == ".dcm":
            dicom_out = _preprocess_dicom_safe(image_path)
            primary_image_array = dicom_out.pixel_array
        else:
            norm_out = _preprocess_image_safe(image_path)
            if getattr(norm_out, "rejected", False):
                return _build_error_response(ctx, f"Quality gate: {norm_out.reject_reason}", "low_quality")
            primary_image_array = norm_out.normalized

        from PIL import Image

        segmentation = None
        roi_pil = None
        seg = _run_segmentation_safe(primary_image_array, ctx)
        if seg is not None:
            segmentation = {
                "bbox": list(getattr(seg, "bbox", []) or []),
                "confidence": float(getattr(seg, "confidence", 0.0) or 0.0),
                "mask_pixels": int(getattr(seg, "segmentation_mask", 0).sum())
                if hasattr(getattr(seg, "segmentation_mask", None), "sum")
                else 0,
                "num_contours": len(getattr(seg, "contours", []) or []),
            }
            roi_crop = getattr(seg, "roi_crop", None)
            if roi_crop is not None and getattr(roi_crop, "size", 0):
                roi_pil = Image.fromarray(roi_crop).convert("RGB")

        if roi_pil is None:
            roi_pil = Image.fromarray(primary_image_array).convert("RGB")

        classification = classify_image(roi_pil)
        gradcam_path = None
        try:
            gradcam_path = generate_gradcam_overlay(roi_pil, job_id=ctx.job_id)["gradcam_path"]
            classification.gradcam_path = gradcam_path
        except Exception as exc:
            logger.warning("[%s] GradCAM failed: %s", ctx.job_id, exc)

        rag = _rag_service()
        explanation = (
            f"Image classified as {classification.label} with "
            f"{classification.confidence:.2%} confidence."
        )
        sources = []
        uncertainty_flag = classification.confidence < 0.6
        if rag is not None:
            try:
                sources = rag.retrieve_evidence(f"{image_type} {classification.label} {clinical_context}".strip())
                explanation = rag.explain_image(
                    image_type=image_type,
                    classification=to_json_safe(classification),
                    segmentation=segmentation or {},
                    sources=sources,
                )
                uncertainty_flag = uncertainty_flag or not bool(sources)
            except Exception as exc:
                logger.warning("[%s] Image RAG fallback: %s", ctx.job_id, exc)

        return PipelineResponse(
            job_id=ctx.job_id,
            patient_id=ctx.patient_id,
            pipeline_type="image",
            status="completed",
            image_classification=to_json_safe(classification),
            segmentation=segmentation,
            verdict="Uncertain",
            confidence_score=round(classification.confidence * 100, 2),
            confidence_label=_confidence_label(classification.confidence * 100),
            explanation=explanation,
            plain_language_summary=explanation,
            sources=sources,
            uncertainty_flag=uncertainty_flag,
            gradcam_path=gradcam_path,
            processing_time_ms=ctx.elapsed_ms,
            completed_at=datetime.now(timezone.utc).isoformat(),
        )
    except Exception as exc:
        logger.error("[%s] Image pipeline crashed: %s", ctx.job_id, traceback.format_exc())
        return _build_error_response(ctx, str(exc))


# ============================================================================
# PIPELINE 2 - REPORT ANALYSIS
# ============================================================================

def run_report_pipeline(
    file_path: str,
    patient_id: Optional[str] = None,
    report_type: str = "lab",
    job_id: Optional[str] = None,
    gender: str = "male",
    age: int = 40,
) -> PipelineResponse:
    """PDF/CSV report -> OCR -> NER -> optional XGBoost/SHAP -> optional RAG."""
    ctx = _make_context("report", patient_id, job_id)
    logger.info("[%s] Starting report pipeline: %s type=%s", ctx.job_id, file_path, report_type)

    try:
        validation = _validate_file_safe(file_path)
        if not validation.valid:
            return _build_error_response(ctx, f"Validation failed: {validation.reason}")

        ocr_result = _run_ocr_safe(file_path)
        raw_text = getattr(ocr_result, "raw_text", "") or ""
        ner_result = _run_ner_safe(raw_text)
        lab_values = _extract_lab_values_safe(raw_text, ner_result, report_type)
        lab_rows = [_lab_value_to_row(value) for value in lab_values]
        lab_dict = _lab_values_to_dict(lab_values, gender=gender, age=age)

        entities = _ner_entities_dict(ner_result)
        tabular_prediction = None
        anomalies = []
        risk_score = 50.0
        risk_factors = []

        if report_type == "lab" and lab_values:
            try:
                tabular = predict_tabular(lab_dict, raw_text)
                tabular_prediction = to_json_safe(tabular)
                risk_score = float(tabular_prediction.get("risk_score", 0.5)) * 100
                risk_factors = tabular_prediction.get("top_contributors", [])
                for test_name, status in tabular_prediction.get("anomalies", {}).items():
                    if status == "ABNORMAL":
                        anomalies.append(
                            {
                                "test": test_name,
                                "value": lab_dict.get(test_name),
                                "status": "ABNORMAL",
                                "severity": "HIGH",
                            }
                        )
            except Exception as exc:
                logger.warning("[%s] XGBoost skipped/failed: %s", ctx.job_id, exc)

        rag_result = _explain_report_safe(
            raw_text=raw_text,
            entities=entities,
            risk_score=risk_score,
            risk_factors=risk_factors,
            anomalies=anomalies,
            report_type=report_type,
        )

        confidence_label = (
            tabular_prediction.get("confidence", "MEDIUM")
            if tabular_prediction
            else _confidence_label(float(rag_result.get("confidence_score", 50.0)))
        )

        return PipelineResponse(
            job_id=ctx.job_id,
            patient_id=ctx.patient_id,
            pipeline_type="report",
            status="completed",
            tabular_prediction=tabular_prediction,
            extracted_entities=entities,
            lab_values=lab_rows,
            anomalies=anomalies,
            verdict=rag_result["verdict"],
            confidence_score=rag_result["confidence_score"],
            confidence_label=confidence_label,
            explanation=rag_result["explanation"],
            plain_language_summary=rag_result["plain_language_summary"],
            sources=rag_result["sources"],
            uncertainty_flag=rag_result["uncertainty_flag"],
            hallucination_flagged=rag_result["hallucination_flagged"],
            shap_top_factors=risk_factors,
            processing_time_ms=ctx.elapsed_ms,
            completed_at=datetime.now(timezone.utc).isoformat(),
        )
    except Exception as exc:
        logger.error("[%s] Report pipeline crashed: %s", ctx.job_id, traceback.format_exc())
        return _build_error_response(ctx, str(exc))


# ============================================================================
# PIPELINE 3 - CLAIM VERIFICATION
# ============================================================================

def run_claim_pipeline(
    claim_text: str,
    patient_id: Optional[str] = None,
    job_id: Optional[str] = None,
) -> PipelineResponse:
    """Claim text -> NER -> RAG verdict with graceful fallback."""
    ctx = _make_context("claim", patient_id, job_id)
    logger.info("[%s] Starting claim pipeline", ctx.job_id)

    try:
        if not claim_text or len(claim_text.strip()) < 10:
            return _build_error_response(ctx, "Claim text too short (min 10 chars)")

        ner_result = _run_ner_safe(claim_text)
        entities = {
            "conditions": list(getattr(ner_result, "conditions", []) or []),
            "medications": list(getattr(ner_result, "medications", []) or []),
        }
        verdict = _verify_claim_safe(claim_text, entities)

        return PipelineResponse(
            job_id=ctx.job_id,
            patient_id=ctx.patient_id,
            pipeline_type="claim",
            status="completed",
            extracted_entities=entities,
            verdict=verdict["verdict"],
            confidence_score=verdict["confidence_score"],
            confidence_label=_confidence_label(float(verdict["confidence_score"])),
            explanation=verdict["explanation"],
            sources=verdict["sources"],
            uncertainty_flag=verdict["uncertainty_flag"],
            hallucination_flagged=verdict["hallucination_flagged"],
            processing_time_ms=ctx.elapsed_ms,
            completed_at=datetime.now(timezone.utc).isoformat(),
        )
    except Exception as exc:
        logger.error("[%s] Claim pipeline crashed: %s", ctx.job_id, traceback.format_exc())
        return _build_error_response(ctx, str(exc))


# ============================================================================
# UNIFIED ENTRY POINT
# ============================================================================

def run_pipeline(input_data: Dict[str, Any], job_id: Optional[str] = None) -> PipelineResponse:
    """Smart router. Dispatches to image, report, or claim pipeline."""
    pipeline_type = input_data.get("type")
    patient_id = input_data.get("patient_id")

    if pipeline_type == "image":
        return run_image_pipeline(
            image_path=input_data["file_path"],
            image_type=input_data.get("image_type", "xray"),
            patient_id=patient_id,
            job_id=job_id,
            clinical_context=input_data.get("clinical_context", ""),
        )
    if pipeline_type == "report":
        return run_report_pipeline(
            file_path=input_data["file_path"],
            patient_id=patient_id,
            report_type=input_data.get("report_type", "lab"),
            job_id=job_id,
            gender=input_data.get("gender", "male"),
            age=input_data.get("age", 40),
        )
    if pipeline_type == "claim":
        return run_claim_pipeline(
            claim_text=input_data["claim_text"],
            patient_id=patient_id,
            job_id=job_id,
        )

    ctx = _make_context("unknown", patient_id, job_id)
    return _build_error_response(ctx, f"Unknown pipeline type: {pipeline_type}")


# ============================================================================
# STANDALONE TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("VaidyaAI Master Pipeline - Self Test")
    print("=" * 60)

    result = run_claim_pipeline(
        claim_text="Aspirin reduces the risk of myocardial infarction by 50%.",
        patient_id="test-001",
    )
    print(json.dumps(to_json_safe(result.model_dump()), indent=2))

    result2 = run_pipeline(
        {
            "type": "claim",
            "claim_text": "Vitamin D deficiency causes osteoporosis.",
            "patient_id": "test-002",
        }
    )
    print(f"Status: {result2.status}")
    print(f"Verdict: {result2.verdict}")
    print(f"Time: {result2.processing_time_ms}ms")

    print("\nPipeline orchestrator ready")
