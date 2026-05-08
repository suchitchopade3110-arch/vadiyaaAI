"""WHO-enhanced image analysis helpers for radiology outputs."""

from __future__ import annotations

from typing import Any

from app.core.disclaimer import MEDICAL_DISCLAIMER
from app.services.ingest_radiology_kb import (
    get_label_metadata,
    get_report_template,
)
from app.services.fix_image_analysis import (
    filter_keyword_fallback_sources,
    get_severity,
    retrieve_image_evidence_from_main_kb,
)

WHO_IMAGE_EXPLAIN_SYSTEM_PROMPT = """
You are a clinical radiology AI assistant. Generate structured clinical
interpretation of medical imaging findings grounded in the provided radiology
pattern evidence.

Rules:
- Use only provided classification, GradCAM/ROI details, and retrieved evidence.
- Map findings to a specific radiographic pattern when possible.
- Include ICD-10 code and urgency level for the primary finding.
- Use plain English for patient summary.
- If evidence is insufficient, set uncertainty_flag=true and verdict=Uncertain.
- Always include the medical disclaimer.
"""


WHO_IMAGE_EXPLAIN_USER_PROMPT = """
Classification Label : {label}
ICD-10 Code          : {icd10}
Urgency Level        : {urgency}
Body Region          : {body_region}
CheXNet Confidence   : {confidence}/100
GradCAM/ROI Regions  : {gradcam_regions}

Radiology Evidence:
{evidence}

Report Template Sections:
{report_sections}
"""


def _to_percent(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return round(number * 100, 1) if 0 <= number <= 1 else round(number, 1)


def _regions_from_segmentation(segmentation: dict[str, Any] | None) -> list[str]:
    if not segmentation:
        return []
    regions = []
    bbox = segmentation.get("bbox")
    if bbox:
        regions.append(f"ROI bbox {bbox}")
    if segmentation.get("num_contours"):
        regions.append(f"{segmentation['num_contours']} segmented contour(s)")
    return regions


def build_image_explanation_prompt(
    label: str,
    chexnet_confidence: float,
    gradcam_regions: list[str] | None = None,
    body_region: str = "lung",
    top_k: int = 3,
) -> dict[str, Any]:
    """Build prompt context for a WHO-grounded image explanation."""
    label_meta = get_label_metadata(label)
    resolved_region = label_meta.get("body_region") or body_region
    evidence_results = []
    try:
        evidence_results = retrieve_image_evidence_from_main_kb(
            label,
            gradcam_regions=gradcam_regions or [resolved_region],
            top_k=top_k,
        )
    except Exception:
        evidence_results = []
    evidence_results = filter_keyword_fallback_sources(evidence_results)
    template = get_report_template(resolved_region)

    evidence_text = "\n".join(
        (
            f"[{index + 1}] Source: {item.get('source') or item.get('title') or 'WHO Imaging Standards'} "
            f"| Topic: {item.get('topic', '')}\n"
            f"    Description: {str(item.get('text', ''))[:260]}"
        )
        for index, item in enumerate(evidence_results)
    ) or "No specific radiology pattern evidence retrieved."

    user_prompt = WHO_IMAGE_EXPLAIN_USER_PROMPT.format(
        label=label,
        icd10=label_meta.get("icd10", "R91.8"),
        urgency=label_meta.get("urgency", "unknown"),
        body_region=resolved_region,
        confidence=_to_percent(chexnet_confidence),
        gradcam_regions=", ".join(gradcam_regions or []) or "Not provided",
        evidence=evidence_text,
        report_sections=", ".join(template["sections"]),
    )

    return {
        "system_prompt": WHO_IMAGE_EXPLAIN_SYSTEM_PROMPT.strip(),
        "user_prompt": user_prompt.strip(),
        "label_meta": label_meta,
        "evidence": evidence_results,
        "template": template,
    }


def default_image_explanation(
    label: str,
    confidence: float,
    prompt_data: dict[str, Any],
) -> dict[str, Any]:
    """Deterministic structured explanation used when no LLM JSON is available."""
    evidence = prompt_data.get("evidence") or []
    label_meta = prompt_data.get("label_meta") or {}
    primary_pattern = (evidence[0].get("pattern") or evidence[0].get("title")) if evidence else label
    confidence_pct = _to_percent(confidence)
    urgency = label_meta.get("urgency", "unknown")
    severity = get_severity(confidence_pct)
    uncertainty = confidence_pct < 50 or not evidence

    interpretation = (
        f"The leading AI finding is {label} with {confidence_pct}% confidence. "
        f"The closest radiology pattern match is {primary_pattern}. "
        f"Confidence severity is {severity}; urgency is classified as {urgency}. "
        "Correlate with clinical history and formal radiology review."
    )

    return {
        "classification_label": label,
        "icd10_code": label_meta.get("icd10", "R91.8"),
        "urgency_level": urgency,
        "confidence_score": confidence_pct,
        "classification_prob": confidence_pct,
        "detection_confidence": confidence_pct,
        "label_classification": "Class probability",
        "label_detection": "Detection confidence",
        "severity": severity,
        "radiographic_pattern": primary_pattern,
        "clinical_interpretation": interpretation,
        "patient_friendly_summary": (
            f"The image analysis found a possible {label.replace('_', ' ').lower()} pattern. "
            "Please have a qualified clinician or radiologist review this result."
        ),
        "report_sections": {},
        "recommendations": [
            "Review with a qualified radiologist or treating clinician.",
            "Correlate with symptoms, exam findings, and prior imaging.",
        ],
        "uncertainty_flag": uncertainty,
        "verdict": "Uncertain" if uncertainty else "Verified",
        "disclaimer": MEDICAL_DISCLAIMER,
    }


def generate_who_structured_report(
    label: str,
    chexnet_confidence: float,
    gradcam_regions: list[str] | None = None,
    patient_info: dict[str, Any] | None = None,
    llm_explanation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate a WHO-format structured radiology report object."""
    label_meta = get_label_metadata(label)
    body_region = label_meta.get("body_region", "lung")
    template = get_report_template(body_region)
    explanation = llm_explanation or {}
    urgency = label_meta.get("urgency", "medium")
    is_urgent = urgency == "critical" or label in template.get("urgency_flags", [])

    provided_sections = explanation.get("report_sections") or {}
    return {
        "report_type": "AI-Assisted Radiology Report (WHO Format)",
        "patient": patient_info or {"id": "anonymous"},
        "primary_finding": {
            "label": label,
            "icd10_code": label_meta.get("icd10", "R91.8"),
            "confidence": _to_percent(chexnet_confidence),
            "urgency": urgency,
            "body_region": body_region,
        },
        "gradcam_regions": gradcam_regions or [],
        "is_urgent": is_urgent,
        "urgency_alert": f"URGENT FINDING: {label}" if is_urgent else None,
        "report_sections": {
            section: provided_sections.get(section, "No focal abnormality described by the AI pipeline.")
            for section in template["sections"]
        },
        "standard_observations": {
            observation: "See clinical interpretation"
            for observation in template["standard_observations"]
        },
        "clinical_interpretation": explanation.get("clinical_interpretation", ""),
        "patient_friendly_summary": explanation.get("patient_friendly_summary", ""),
        "recommendations": explanation.get("recommendations", []),
        "radiographic_pattern": explanation.get("radiographic_pattern", label),
        "uncertainty_flag": explanation.get("uncertainty_flag", _to_percent(chexnet_confidence) < 50),
        "verdict": explanation.get("verdict", "Uncertain"),
        "disclaimer": explanation.get("disclaimer", MEDICAL_DISCLAIMER),
        "report_format": "WHO Diagnostic Imaging Protocol",
        "model_used": "CheXNet + GradCAM + WHO RAG",
    }


def build_who_image_outputs(
    *,
    label: str,
    confidence: float,
    segmentation: dict[str, Any] | None = None,
    patient_id: str | None = None,
    gradcam_regions: list[str] | None = None,
) -> dict[str, Any]:
    regions = list(gradcam_regions or []) + _regions_from_segmentation(segmentation)
    prompt_data = build_image_explanation_prompt(label, confidence, regions)
    explanation = default_image_explanation(label, confidence, prompt_data)
    report = generate_who_structured_report(
        label=label,
        chexnet_confidence=confidence,
        gradcam_regions=regions,
        patient_info={"id": patient_id or "anonymous"},
        llm_explanation=explanation,
    )
    return {
        "prompt": prompt_data,
        "label_metadata": prompt_data["label_meta"],
        "radiology_evidence": prompt_data["evidence"],
        "structured_explanation": explanation,
        "who_structured_report": report,
    }
