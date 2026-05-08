"""
classifier_v2.py - VaidyaAI 14-Class CheXNet
============================================
14-class pathology detection for chest X-rays.

Primary path:
    torchxrayvision DenseNet121 NIH weights

Fallback path:
    legacy CheXNet wrapper from app.image_pipeline.classifier

This module keeps backward-compatible fields (label, confidence,
probabilities) while exposing the richer 14-class contract.
"""

from __future__ import annotations

import logging
import re
import uuid
from functools import lru_cache
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image

try:
    from pydantic import BaseModel, Field
except Exception:  # pragma: no cover - minimal fallback for stripped envs
    class BaseModel:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

        def model_dump(self):
            return dict(self.__dict__)

    def Field(default_factory=None, default=None):
        if default_factory is not None:
            return default_factory()
        return default


logger = logging.getLogger("vaidya.classifier_v2")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NIH_LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion",
    "Emphysema",
    "Fibrosis",
    "Hernia",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pleural_Thickening",
    "Pneumonia",
    "Pneumothorax",
]

CLINICAL_SEVERITY = {
    "Pneumothorax": "CRITICAL",
    "Cardiomegaly": "HIGH",
    "Edema": "HIGH",
    "Consolidation": "HIGH",
    "Mass": "HIGH",
    "Nodule": "MODERATE",
    "Pneumonia": "HIGH",
    "Effusion": "HIGH",
    "Atelectasis": "MODERATE",
    "Emphysema": "MODERATE",
    "Fibrosis": "MODERATE",
    "Infiltration": "MODERATE",
    "Pleural_Thickening": "LOW",
    "Hernia": "LOW",
}

CLINICAL_MEANINGS = {
    "Atelectasis": "Partial collapse of lung - reduces oxygen exchange",
    "Cardiomegaly": "Enlarged heart - may indicate heart disease",
    "Consolidation": "Lung tissue filled with fluid - possible pneumonia",
    "Edema": "Fluid in lungs - heart failure or infection",
    "Effusion": "Fluid around lungs - infection or heart/kidney issue",
    "Emphysema": "Air trapping in lungs - COPD pattern",
    "Fibrosis": "Lung scarring - chronic lung disease",
    "Hernia": "Abdominal contents in chest - diaphragmatic hernia",
    "Infiltration": "Non-specific opacity - infection or inflammation",
    "Mass": "Dense opacity - requires further evaluation",
    "Nodule": "Small round opacity - follow-up CT recommended",
    "Pleural_Thickening": "Thickened pleura - past infection or exposure",
    "Pneumonia": "Lung infection - bacterial or viral pneumonia",
    "Pneumothorax": "Air in pleural space - collapsed lung (URGENT)",
}

DETECTION_THRESHOLDS = {label: 0.50 for label in NIH_LABELS}


def _normalize_label(label: str) -> str:
    return re.sub(r"[\s_/-]+", "", label or "").lower()


class PathologyFinding(BaseModel):
    label: str
    probability: float
    detected: bool
    severity: str
    clinical_meaning: str


class ImageClassification14(BaseModel):
    primary_finding: str
    primary_confidence: float
    all_findings: List[PathologyFinding]
    detected_pathologies: List[str]
    no_finding: bool
    gradcam_path: str = ""
    model_version: str = "torchxrayvision-densenet121-nih"

    # Backward-compatible fields used by the rest of the app.
    label: str = ""
    confidence: float = 0.0
    probabilities: Dict[str, float] = Field(default_factory=dict)
    top_class: str = ""
    top_confidence: float = 0.0


@lru_cache(maxsize=1)
def get_xrv_model():
    """Load the NIH DenseNet121 model from torchxrayvision."""
    import torchxrayvision as xrv

    logger.info("Loading torchxrayvision NIH DenseNet121...")
    model = xrv.models.DenseNet(weights="densenet121-res224-nih")
    model.to(DEVICE)
    model.eval()
    logger.info("torchxrayvision model loaded on %s", DEVICE)
    return model


def _preprocess_for_xrv(img_pil: Image.Image) -> torch.Tensor:
    img_gray = img_pil.convert("L").resize((224, 224), Image.LANCZOS)
    img_np = np.array(img_gray, dtype=np.float32)
    img_np = (img_np / 255.0) * 2048.0 - 1024.0
    return torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).to(DEVICE)


def _build_findings(probabilities: Dict[str, float]) -> List[PathologyFinding]:
    from app.services.fix_image_analysis import get_severity

    findings: List[PathologyFinding] = []
    for label in NIH_LABELS:
        prob = float(probabilities.get(label, 0.0))
        threshold = DETECTION_THRESHOLDS.get(label, 0.15)
        findings.append(
            PathologyFinding(
                label=label,
                probability=round(prob, 4),
                detected=prob >= threshold,
                severity=get_severity(prob * 100),
                clinical_meaning=CLINICAL_MEANINGS.get(label, ""),
            )
        )
    findings.sort(key=lambda item: item.probability, reverse=True)
    return findings


def _legacy_fallback(img_pil: Image.Image) -> ImageClassification14:
    from app.image_pipeline.classifier import run_chexnet

    legacy = run_chexnet(np.array(img_pil.convert("RGB")))
    legacy_probs = getattr(legacy, "probabilities", {}) or {}
    probabilities = {label: float(legacy_probs.get(label, 0.0)) for label in NIH_LABELS}
    findings = _build_findings(probabilities)
    detected_pathologies = [item.label for item in findings if item.detected]

    primary = getattr(legacy, "label", "No Finding") or "No Finding"
    primary_conf = float(getattr(legacy, "confidence", 0.0) or 0.0)
    no_finding = primary.lower() in {"normal", "no finding"}
    legacy_confidence = 0.0 if no_finding else primary_conf

    return ImageClassification14(
        primary_finding="No Finding" if no_finding else primary,
        primary_confidence=round(primary_conf, 4),
        all_findings=findings,
        detected_pathologies=detected_pathologies,
        no_finding=no_finding,
        model_version="legacy-chexnet-fallback",
        label="No Finding" if no_finding else primary,
        confidence=round(legacy_confidence, 4),
        probabilities=probabilities,
        top_class="No Finding" if no_finding else primary,
        top_confidence=round(legacy_confidence, 4),
    )


def classify_image_14class(
    img_pil: Image.Image,
    detection_threshold: float = 0.15,
) -> ImageClassification14:
    """
    14-class chest X-ray pathology detection.

    Keeps backward-compatible fields:
        label, confidence, probabilities, top_class, top_confidence
    """
    if not isinstance(img_pil, Image.Image):
        raise TypeError(f"Expected PIL.Image, got {type(img_pil)}")

    try:
        model = get_xrv_model()
    except Exception as exc:
        logger.warning("torchxrayvision unavailable, using legacy CheXNet fallback: %s", exc)
        return _legacy_fallback(img_pil)

    try:
        img_tensor = _preprocess_for_xrv(img_pil)
        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.sigmoid(output).squeeze(0).detach().cpu().numpy()

        xrv_labels = [str(label) for label in getattr(model, "pathologies", NIH_LABELS)]
        prob_lookup = {_normalize_label(label): float(prob) for label, prob in zip(xrv_labels, probs)}
        probabilities = {label: prob_lookup.get(_normalize_label(label), 0.0) for label in NIH_LABELS}
        all_findings = _build_findings(probabilities)
        detected_pathologies = [item.label for item in all_findings if item.detected]

        if detected_pathologies:
            primary = next(item for item in all_findings if item.detected)
            primary_finding = primary.label
            primary_confidence = primary.probability
            no_finding = False
            legacy_confidence = primary_confidence
        else:
            primary_finding = "No Finding"
            primary_confidence = float(1.0 - max((item.probability for item in all_findings), default=0.0))
            no_finding = True
            legacy_confidence = 0.0

        return ImageClassification14(
            primary_finding=primary_finding,
            primary_confidence=round(primary_confidence, 4),
            all_findings=all_findings,
            detected_pathologies=detected_pathologies,
            no_finding=no_finding,
            label=primary_finding,
            confidence=round(legacy_confidence, 4),
            probabilities=probabilities,
            top_class=primary_finding,
            top_confidence=round(legacy_confidence, 4),
        )
    except Exception as exc:
        logger.warning("14-class classifier failed, using legacy fallback: %s", exc)
        return _legacy_fallback(img_pil)


def classify_image(img_pil: Image.Image) -> ImageClassification14:
    """Backward-compatible entry point."""
    return classify_image_14class(img_pil)


def generate_gradcam_14class(
    img_pil: Image.Image,
    target_class: Optional[str] = None,
    job_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compatibility wrapper for notebook parity.

    The live pipeline already owns GradCAM rendering, so we return a
    structured placeholder rather than duplicating the heatmap logic.
    """
    if job_id is None:
        job_id = str(uuid.uuid4())
    return {
        "predicted_class": target_class or "Unknown",
        "confidence": 0.0,
        "gradcam_path": "",
        "job_id": job_id,
    }
