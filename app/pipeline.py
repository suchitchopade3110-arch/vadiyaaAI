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

from app.image_pipeline.classifier_v2 import classify_image
from app.ml.ml_prediction_engine import (
    ensemble_predict,
    generate_gradcam_overlay,
    predict_tabular,
    to_json_safe,
)
from clinical_reference import analyze_blood_report, enrich_lab_with_who_ranges, clean_test_name

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
    ensemble_details: Optional[Dict[str, Any]] = None
    image_classification: Optional[Dict[str, Any]] = None
    label_metadata: Optional[Dict[str, Any]] = None
    radiology_evidence: List[Dict[str, Any]] = Field(default_factory=list)
    who_structured_report: Optional[Dict[str, Any]] = None
    ensemble: Optional[Dict[str, Any]] = None

    segmentation: Optional[Dict[str, Any]] = None

    extracted_entities: Optional[Dict[str, Any]] = None
    lab_values: List[Dict[str, Any]] = Field(default_factory=list)
    anomalies: List[Dict[str, Any]] = Field(default_factory=list)

    verdict: Optional[str] = None
    risk_score: Optional[float] = None
    risk_label: Optional[str] = None
    differential_diagnosis: Optional[Dict[str, Any]] = None
    explanation_mode: str = "brief"
    explanation_brief: Optional[str] = None
    explanation_full: Optional[str] = None
    rag_explanation: Optional[str] = None
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
    patient_history: List[Dict[str, Any]] = Field(default_factory=list)
    history_comparison: Optional[Dict[str, Any]] = None
    ocr_engine: Optional[str] = None
    ocr_confidence: Optional[float] = None
    ner_engine: Optional[str] = None
    qr_token: Optional[str] = None
    qr_available: bool = False

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


def _normalize_lab_key(name: Any) -> str:
    return str(name or "").lower().strip()


def _fallback_risk_from_anomalies(anomalies: List[Dict[str, Any]]) -> float:
    if not anomalies:
        return 0.0
    critical = sum(1 for item in anomalies if str(item.get("severity", "")).upper() == "CRITICAL")
    high = sum(1 for item in anomalies if str(item.get("severity", "")).upper() == "HIGH")
    score = min(95.0, 15.0 + (critical * 25.0) + (high * 15.0))
    return round(score, 1)


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
    if report_type != "lab":
        return values
    try:
        from app.services.preprocessor import extract_all_lab_values, deduplicate_lab_values

        layered = extract_all_lab_values(raw_text)
        if values:
            return deduplicate_lab_values(values + layered)
        return layered
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


def _lab_value_to_row(lab_value: Any, gender: str = "male", age: int = 40) -> Dict[str, Any]:
    name = getattr(lab_value, "name", None)
    value = getattr(lab_value, "value", None)
    unit = getattr(lab_value, "unit", "")
    ref_low = getattr(lab_value, "ref_low", None)
    ref_high = getattr(lab_value, "ref_high", None)
    reference = getattr(lab_value, "reference", None)
    if isinstance(lab_value, dict):
        name = lab_value.get("name") or lab_value.get("test")
        value = lab_value.get("value") if lab_value.get("value") is not None else lab_value.get("result")
        unit = lab_value.get("unit", "")
        ref_low = lab_value.get("ref_low")
        ref_high = lab_value.get("ref_high")
        reference = lab_value.get("reference") or lab_value.get("reference_range")

    if name:
        name = clean_test_name(str(name))

    if ref_low is not None and ref_high is not None:
        reference = f"{ref_low} - {ref_high}"
    elif not reference and name:
        try:
            from app.services.preprocessor import resolve_reference

            low, high, _ = resolve_reference(str(name), gender=gender, age=age, pdf_ref_low=ref_low, pdf_ref_high=ref_high)
            if low is not None and high is not None:
                reference = f"{low} - {high}"
        except Exception:
            reference = None

    return {
        "field": name,
        "test": name,
        "result": value,
        "unit": unit,
        "reference": reference or "--",
        "flag": "NORMAL",
        "ref_low": ref_low,
        "ref_high": ref_high,
    }


def _lab_values_to_dict(lab_values: List[Any], gender: str, age: int) -> Dict[str, Any]:
    lab_dict = {}
    for lab_value in lab_values:
        row = _lab_value_to_row(lab_value, gender=gender, age=age)
        if row["test"] is None or row["result"] is None:
            continue
        key = _normalize_lab_key(row["test"]).replace(" ", "_")
        lab_dict[key] = row["result"]
    lab_dict["gender"] = 1 if gender.lower() == "male" else 0
    lab_dict["age"] = age
    return lab_dict


def _engine_anomalies_to_rows(anomalies: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for key, item in (anomalies or {}).items():
        if not isinstance(item, dict) or item.get("status") != "ABNORMAL":
            continue
        rows.append(
            {
                "field": item.get("field") or key,
                "test": item.get("field") or key,
                "value": item.get("value"),
                "unit": item.get("unit", ""),
                "reference": item.get("reference", "--"),
                "direction": item.get("direction", "ABNORMAL"),
                "severity": item.get("severity", "HIGH"),
                "flag": item.get("flag", "⚠️ ABNORMAL"),
                "clinical_meaning": item.get("clinical_meaning", ""),
            }
        )
    return rows


def _deduplicate_anomalies(anomalies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: Dict[str, Dict[str, Any]] = {}
    for item in anomalies or []:
        if not isinstance(item, dict):
            continue
        key = _normalize_lab_key(item.get("field") or item.get("test") or item.get("label"))
        if not key:
            continue
        existing = seen.get(key)
        if existing is None or len(str(item.get("clinical_meaning") or item.get("explanation") or "")) > len(
            str(existing.get("clinical_meaning") or existing.get("explanation") or "")
        ):
            seen[key] = item
    return list(seen.values())


def _urine_entities_to_rows(entities: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    dipstick = entities.get("dipstick") or {}
    microscopy = entities.get("microscopy") or {}
    quantitative = entities.get("quantitative") or {}

    for section_name, section in [
        ("dipstick", dipstick),
        ("microscopy", microscopy),
        ("quantitative", quantitative),
    ]:
        if not isinstance(section, dict):
            continue
        for key, value in section.items():
            if key == "casts" and isinstance(value, dict):
                for cast_name, cast_value in value.items():
                    rows.append(
                        {
                            "field": cast_name.replace("_", " ").title(),
                            "test": cast_name,
                            "result": cast_value,
                            "value": cast_value,
                            "unit": "",
                            "reference": "absent",
                            "flag": "ABNORMAL" if str(cast_value).lower() == "present" else "NORMAL",
                            "section": section_name,
                        }
                    )
                continue
            if not isinstance(value, dict):
                continue
            rows.append(
                {
                    "field": key.replace("_", " ").title(),
                    "test": key,
                    "result": value.get("value", ""),
                    "value": value.get("value", ""),
                    "unit": value.get("unit", ""),
                    "reference": value.get("reference", "--"),
                    "flag": str(value.get("flag", "normal")).upper(),
                    "section": section_name,
                }
            )
    return rows


def _urine_pattern_anomalies(patterns: List[str], evidence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    evidence_by_pattern = {item.get("pattern"): item for item in evidence or []}
    urgency_to_severity = {
        "critical": "CRITICAL",
        "high": "HIGH",
        "medium": "MODERATE",
        "low": "LOW",
        "none": "NORMAL",
    }
    rows = []
    for pattern in patterns or []:
        matched = next(
            (item for item in evidence_by_pattern.values() if pattern.lower().replace("_", " ") in str(item.get("pattern", "")).lower()),
            {},
        )
        severity = urgency_to_severity.get(str(matched.get("urgency", "medium")).lower(), "MODERATE")
        rows.append(
            {
                "field": pattern.replace("_", " "),
                "test": pattern,
                "value": "detected",
                "unit": "",
                "reference": "not detected",
                "direction": "ABNORMAL",
                "severity": severity,
                "flag": f"⚠️ {severity}",
                "clinical_meaning": matched.get("text", ""),
                "status": "ABNORMAL",
            }
        )
    return rows


def _urine_risk_from_patterns(patterns: List[str]) -> float:
    score_map = {
        "DKA": 95.0,
        "Pre_eclampsia": 95.0,
        "Nephritic": 75.0,
        "Nephrotic": 75.0,
        "UTI": 65.0,
        "Renal_TB_suspected": 75.0,
        "Obstructive_jaundice": 70.0,
        "Hemolytic": 55.0,
    }
    return max([score_map.get(pattern, 35.0) for pattern in patterns] or [20.0])


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

        who_outputs = {}
        try:
            from app.services.image_analysis_service import build_who_image_outputs

            who_outputs = build_who_image_outputs(
                label=classification.label,
                confidence=classification.confidence,
                segmentation=segmentation or {},
                patient_id=ctx.patient_id,
            )
            structured = who_outputs.get("structured_explanation") or {}
            explanation = structured.get("clinical_interpretation") or explanation
            uncertainty_flag = uncertainty_flag or bool(structured.get("uncertainty_flag"))
        except Exception as exc:
            logger.warning("[%s] WHO image enrichment fallback: %s", ctx.job_id, exc)

        evidence_sources = who_outputs.get("radiology_evidence") or []
        return PipelineResponse(
            job_id=ctx.job_id,
            patient_id=ctx.patient_id,
            pipeline_type="image",
            status="completed",
            image_classification=to_json_safe(classification),
            label_metadata=who_outputs.get("label_metadata"),
            radiology_evidence=evidence_sources,
            who_structured_report=who_outputs.get("who_structured_report"),
            segmentation=segmentation,
            verdict=(who_outputs.get("structured_explanation") or {}).get("verdict", "Uncertain"),
            risk_score=round(classification.confidence * 100, 2),
            risk_label=_confidence_label(classification.confidence * 100),
            confidence_score=round(classification.confidence * 100, 2),
            confidence_label=_confidence_label(classification.confidence * 100),
            explanation=explanation,
            plain_language_summary=(who_outputs.get("structured_explanation") or {}).get("patient_friendly_summary", explanation),
            sources=evidence_sources or sources,
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
    explanation_mode: str = "brief",
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
        urine_result = None
        try:
            from app.services.clinical_ner_service import detect_report_type, run_clinical_pipeline

            if detect_report_type(raw_text) == "urinalysis":
                urine_result = run_clinical_pipeline(raw_text)
        except Exception as exc:
            logger.warning("[%s] Urinalysis NER enrichment skipped: %s", ctx.job_id, exc)
        ner_result = _NerResult(conditions=[], medications=[], dates=[], lab_values=[]) if urine_result else _run_ner_safe(raw_text)

        lab_values = _extract_lab_values_safe(raw_text, ner_result, report_type)
        if report_type == "lab":
            lab_values = enrich_lab_with_who_ranges(lab_values, gender=gender)
        lab_rows = [_lab_value_to_row(value) for value in lab_values]
        lab_dict = _lab_values_to_dict(lab_values, gender=gender, age=age)
        clinical_analysis = analyze_blood_report(lab_dict, gender=gender, age=age) if report_type == "lab" else None

        entities = _ner_entities_dict(ner_result)
        if clinical_analysis is not None:
            entities["conditions"] = clinical_analysis["active_conditions"]
        tabular_prediction = None
        anomalies = []
        risk_score = 50.0
        risk_factors = []
        if urine_result:
            entities = urine_result.get("entities") or {}
            entities["_source"] = urine_result.get("model_used")
            lab_rows = _urine_entities_to_rows(entities)
            lab_dict = urine_result.get("features") or {}
            anomalies = _urine_pattern_anomalies(
                entities.get("detected_patterns") or [],
                urine_result.get("rag_evidence") or [],
            )
            risk_score = _urine_risk_from_patterns(entities.get("detected_patterns") or [])
            risk_factors = [
                {"feature": pattern, "shap": 0.0}
                for pattern in (entities.get("detected_patterns") or [])
            ]
            tabular_prediction = {
                "report_type": "urinalysis",
                "features": urine_result.get("features") or {},
                "rag_evidence": urine_result.get("rag_evidence") or [],
                "model_used": urine_result.get("model_used"),
            }

        if report_type == "lab" and lab_values and not urine_result:
            try:
                tabular = predict_tabular(lab_dict, raw_text)
                tabular_prediction = to_json_safe(tabular)
                risk_factors = tabular_prediction.get("top_contributors", [])
                if clinical_analysis is not None:
                    row_lookup = {
                        _normalize_lab_key(row.get("field") or row.get("test")): row
                        for row in lab_rows
                    }
                    anomalies = []
                    for key, item in clinical_analysis["anomalies"].items():
                        if not isinstance(item, dict) or item.get("status") != "ABNORMAL":
                            continue
                        row = row_lookup.get(_normalize_lab_key(key), {})
                        anomalies.append(
                            {
                                "field": clean_test_name(item.get("field") or key),
                                "test": clean_test_name(key),
                                "value": item.get("value", row.get("result")),
                                "unit": item.get("unit", row.get("unit", "")),
                                "reference": item.get("reference", row.get("reference", "--")),
                                "direction": item.get("direction", "ABNORMAL"),
                                "severity": item.get("severity", "HIGH"),
                                "flag": item.get("flag", "⚠️ ABNORMAL"),
                                "clinical_meaning": item.get("clinical_meaning", ""),
                                "status": "ABNORMAL",
                            }
                        )
                    risk_map = {"normal": 0.0, "moderate": 30.0, "high": 70.0, "critical": 95.0}
                    risk_score = risk_map.get(str(clinical_analysis["risk_level"]).lower(), 30.0 if anomalies else 0.0)
                    risk_factors = clinical_analysis["conditions"] or risk_factors
                    for row in lab_rows:
                        key = _normalize_lab_key(row.get("field") or row.get("test"))
                        item = clinical_analysis["anomalies"].get(key)
                        if isinstance(item, dict):
                            row["reference"] = item.get("reference", row.get("reference", "--"))
                            row["flag"] = item.get("flag", row.get("flag", "NORMAL"))
                            row["clinical_meaning"] = item.get("clinical_meaning", "")
                else:
                    risk_score = float(tabular_prediction.get("risk_score", 0.5)) * 100
                    anomaly_map = tabular_prediction.get("anomalies", {}) or {}
                    row_lookup = {
                        _normalize_lab_key(row.get("field") or row.get("test")): row
                        for row in lab_rows
                    }
                    for test_name, details in anomaly_map.items():
                        if not isinstance(details, dict) or details.get("status") != "ABNORMAL":
                            continue
                        row = row_lookup.get(_normalize_lab_key(test_name), {})
                        anomalies.append(
                            {
                                "field": clean_test_name(row.get("field") or details.get("field") or test_name),
                                "test": clean_test_name(test_name),
                                "value": row.get("result", details.get("value", lab_dict.get(test_name))),
                                "unit": row.get("unit", details.get("unit", "")),
                                "reference": row.get("reference", details.get("reference", "--")),
                                "direction": details.get("direction", "ABNORMAL"),
                                "severity": details.get("severity", "HIGH"),
                                "flag": details.get("flag", "⚠️ ABNORMAL"),
                                "clinical_meaning": details.get("clinical_meaning", ""),
                                "status": "ABNORMAL",
                            }
                        )
            except Exception as exc:
                logger.warning("[%s] XGBoost skipped/failed: %s", ctx.job_id, exc)

        if report_type == "lab" and anomalies and risk_score <= 0.05:
            risk_score = max(_fallback_risk_from_anomalies(anomalies), 30.0)
            if risk_score <= 0.05:
                critical = sum(1 for a in anomalies if str(a.get("severity", "")).upper() == "CRITICAL")
                high = sum(1 for a in anomalies if str(a.get("severity", "")).upper() == "HIGH")
                risk_score = min(95.0, 15.0 + (critical * 25.0) + (high * 15.0))

        ensemble_details = None
        if urine_result:
            risk_label = _confidence_label(risk_score)
            ensemble_details = {
                "ensemble_risk_score": risk_score,
                "risk_label": risk_label,
                "model_scores": {"urinalysis_rules": risk_score},
                "model_weights": {"urinalysis_rules": 1.0},
                "model_agreement": "rules",
                "agreement_std_dev": 0.0,
                "shap_values": {},
                "top_factors": risk_factors,
                "anomalies": anomalies,
                "uncertainty_flag": not bool(anomalies),
                "confidence": risk_score,
                "models_used": ["urinalysis_rules", "urinalysis_rag"],
            }
        else:
            try:
                from app.services.ensemble_predictor import ensemble_predict

                ensemble_details = ensemble_predict(
                    ml_features=lab_dict,
                    lab_values=lab_rows,
                    clinical_risk_level=clinical_analysis.get("risk_level") if clinical_analysis else None,
                )
                risk_score = float(ensemble_details.get("ensemble_risk_score", risk_score))
                risk_label = ensemble_details.get("risk_label")
                ensemble_anomalies = ensemble_details.get("anomalies") or []
                if ensemble_anomalies:
                    anomalies = _deduplicate_anomalies(anomalies + ensemble_anomalies)
                if ensemble_details.get("top_factors"):
                    risk_factors = ensemble_details["top_factors"]
            except Exception as exc:
                logger.warning("[%s] Ensemble predictor skipped: %s", ctx.job_id, exc)
                risk_label = _confidence_label(risk_score)

        try:
            from app.services.differential_diagnosis import generate_differential, generate_explanation

            differential_diagnosis = generate_differential(
                entities=entities,
                anomalies=anomalies,
                risk_score=risk_score,
                report_type=report_type,
            )
            explanation_mode = "full" if str(explanation_mode).lower() == "full" else "brief"
            generated_explanation = generate_explanation(
                entities=entities,
                anomalies=anomalies,
                risk_score=risk_score,
                risk_label=risk_label,
                mode=explanation_mode,
            )
            explanation_brief = (
                generated_explanation
                if explanation_mode == "brief"
                else generate_explanation(entities, anomalies, risk_score, risk_label, mode="brief")
            )
            explanation_full = generated_explanation if explanation_mode == "full" else None
        except Exception as exc:
            logger.warning("[%s] Differential diagnosis skipped: %s", ctx.job_id, exc)
            differential_diagnosis = None
            explanation_mode = "brief"
            explanation_brief = None
            explanation_full = None

        rag_result = _explain_report_safe(
            raw_text=raw_text,
            entities=entities,
            risk_score=risk_score,
            risk_factors=risk_factors,
            anomalies=anomalies,
            report_type=report_type,
        )

        patient_history: List[Dict[str, Any]] = []
        history_comparison = None
        if patient_id:
            try:
                from app.services.patient_history_service import compare_with_latest_history, get_patient_history_sync

                patient_history = get_patient_history_sync(patient_id, limit=5)
                current_for_compare = {
                    "report_id": ctx.job_id,
                    "risk_score": risk_score,
                    "lab_values": lab_rows,
                    "extracted_entities": entities,
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                }
                history_comparison = compare_with_latest_history(current_for_compare, patient_history)
            except Exception as exc:
                logger.warning("[%s] Patient history comparison skipped: %s", ctx.job_id, exc)

        confidence_label = (
            tabular_prediction.get("confidence", "MEDIUM")
            if tabular_prediction
            else risk_label or _confidence_label(float(rag_result.get("confidence_score", 50.0)))
        )

        return PipelineResponse(
            job_id=ctx.job_id,
            patient_id=ctx.patient_id,
            pipeline_type="report",
            status="completed",
            tabular_prediction=tabular_prediction,
            ensemble_details=ensemble_details,
            extracted_entities=entities,
            lab_values=lab_rows,
            anomalies=anomalies,
            verdict=rag_result["verdict"],
            risk_score=risk_score,
            risk_label=risk_label or confidence_label,
            differential_diagnosis=differential_diagnosis,
            explanation_mode=explanation_mode,
            explanation_brief=explanation_brief,
            explanation_full=explanation_full,
            rag_explanation=rag_result["explanation"],
            confidence_score=rag_result["confidence_score"],
            confidence_label=confidence_label,
            explanation=explanation_brief or rag_result["explanation"],
            plain_language_summary=explanation_brief or rag_result["plain_language_summary"],
            sources=rag_result["sources"],
            uncertainty_flag=rag_result["uncertainty_flag"] or bool((ensemble_details or {}).get("uncertainty_flag")),
            hallucination_flagged=rag_result["hallucination_flagged"],
            shap_top_factors=risk_factors,
            patient_history=patient_history[:3],
            history_comparison=history_comparison,
            ocr_engine="paddle",
            ocr_confidence=getattr(ocr_result, "confidence", None),
            ner_engine="groq-llama3-ner" if os.getenv("GROQ_API_KEY") else "clinicalbert-or-regex-ner",
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
            risk_score=verdict["confidence_score"],
            risk_label=_confidence_label(float(verdict["confidence_score"])),
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
