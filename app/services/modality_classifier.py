"""Modality-specific image classification router.

The main image pipeline still uses the existing CheXNet classifier for chest
X-rays. Non-X-ray modalities are routed to purpose-built classifiers when their
optional model dependencies and weights are available, with safe placeholders
when they are not.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    label: str
    top_class: str
    confidence: float
    probabilities: dict[str, float] = field(default_factory=dict)
    primary_finding: str = ""
    severity: str = "MODERATE"
    icd10: str = "Z03.89"
    body_region: str = "unknown"
    modality: str = "unknown"
    model_used: str = "unknown"
    gradcam_path: str | None = None
    all_findings: list[dict[str, Any]] = field(default_factory=list)


CHEST_MODALITIES = {"xray", "x-ray", "x_ray", "chest", "cxr"}
MRI_MODALITIES = {"mri", "brain", "brain_mri"}
CT_MODALITIES = {"ct", "ct_scan", "computed_tomography"}
SKIN_MODALITIES = {"skin", "dermatology", "derm"}
PATH_MODALITIES = {"pathology", "histology", "cytology", "wsi"}

MRI_MODEL = os.getenv("MRI_CLASSIFIER_MODEL", "itistamtran/vit_brain_tumor_multiclass_v2")
MRI_MODEL_CHAIN = [
    item.strip()
    for item in os.getenv(
        "MRI_CLASSIFIER_MODEL_CHAIN",
        ",".join(
            [
                MRI_MODEL,
                "Andrewwango/vit-base-patch16-224-finetuned-brain-mri-v3",
                "Devarshi/Brain_Tumor_MRI_Images_classification",
                "seriousfish/brain-tumor-classification",
            ]
        ),
    ).split(",")
    if item.strip()
]
SKIN_MODEL = os.getenv("SKIN_CLASSIFIER_MODEL", "NightBaron/skin-cancer-detection-model")
SKIN_FALLBACK_MODEL = os.getenv("SKIN_CLASSIFIER_FALLBACK_MODEL", "Anwarkh/skin-lesion-classifier")
PATHOLOGY_MODEL = os.getenv("PATHOLOGY_CLASSIFIER_MODEL", "1aurent/swin_tiny_patch4_window7_224.CRC_100K_Random")
HF_CLASSIFIER_LOCAL_ONLY = os.getenv("HF_CLASSIFIER_LOCAL_ONLY", "true").lower() not in {"0", "false", "no"}

_mri_pipe = None
_mri_model_id = None
_skin_pipe = None
_pathology_pipe = None

MRI_ICD10 = {
    "glioma_tumor": "C71.9",
    "glioma": "C71.9",
    "meningioma_tumor": "D32.9",
    "meningioma": "D32.9",
    "pituitary_tumor": "D35.2",
    "pituitary": "D35.2",
    "no_tumor": "Z03.89",
    "notumor": "Z03.89",
    "unknown": "Z03.89",
}

MRI_LABELS = {
    "glioma_tumor": "Glioma",
    "glioma": "Glioma",
    "meningioma_tumor": "Meningioma",
    "meningioma": "Meningioma",
    "pituitary_tumor": "Pituitary Tumor",
    "pituitary": "Pituitary Tumor",
    "no_tumor": "No Tumor Detected",
    "notumor": "No Tumor Detected",
    "unknown": "Unknown / Out-of-Distribution",
}

MRI_CLINICAL = {
    "glioma_tumor": "Brain tumor arising from glial cells. Neurosurgical and oncology review is recommended.",
    "glioma": "Brain tumor arising from glial cells. Neurosurgical and oncology review is recommended.",
    "meningioma_tumor": "Tumor arising from the meninges. Specialist review is recommended.",
    "meningioma": "Tumor arising from the meninges. Specialist review is recommended.",
    "pituitary_tumor": "Pituitary region mass. Endocrine workup and neurosurgical assessment are recommended.",
    "pituitary": "Pituitary region mass. Endocrine workup and neurosurgical assessment are recommended.",
    "no_tumor": "No tumor pattern detected by the AI model. Correlate with formal radiology review.",
    "notumor": "No tumor pattern detected by the AI model. Correlate with formal radiology review.",
    "unknown": "The model identified the image as outside the core tumor classes or uncertain. Formal radiology review is required.",
}

SKIN_ICD10 = {
    "melanoma": "C43.9",
    "basal cell carcinoma": "C44.91",
    "basal_cell_carcinoma": "C44.91",
    "squamous cell carcinoma": "C44.92",
    "squamous_cell_carcinoma": "C44.92",
    "actinic keratosis": "L57.0",
    "actinic_keratosis": "L57.0",
    "benign keratosis": "L82.1",
    "benign_keratosis": "L82.1",
    "dermatofibroma": "D23.9",
    "vascular lesion": "L98.9",
    "vascular_lesion": "L98.9",
    "nevus": "D22.9",
}

WHO_TIERS = {
    "benign": {"tier": 2, "label": "Benign / Negative for Malignancy", "action": "Routine follow-up"},
    "normal": {"tier": 2, "label": "Benign / Negative for Malignancy", "action": "Routine follow-up"},
    "atypical": {"tier": 3, "label": "Atypical", "action": "Repeat sampling or ancillary testing"},
    "suspicious": {"tier": 5, "label": "Suspicious for Malignancy", "action": "Core needle biopsy or tumor board review"},
    "malignant": {"tier": 6, "label": "Malignant", "action": "Immediate oncological staging"},
    "tumor": {"tier": 5, "label": "Suspicious for Malignancy", "action": "Core needle biopsy or tumor board review"},
    "cancer": {"tier": 6, "label": "Malignant", "action": "Immediate oncological staging"},
}


def _severity(conf_pct: float, label: str = "") -> str:
    label_key = _normalize_label(label)
    if any(term in label_key for term in ("no_tumor", "notumor", "normal", "benign", "negative", "no_finding", "unknown")):
        return "LOW"
    if conf_pct >= 80:
        return "HIGH"
    if conf_pct >= 50:
        return "MODERATE"
    return "LOW"


def _platt_scale(confidence: float, temperature: float = 3.2, bias: float = 0.0) -> float:
    """Monotonic confidence calibration for overconfident softmax outputs."""
    confidence = max(1e-6, min(1.0 - 1e-6, float(confidence or 0.0)))
    logit = math.log(confidence / (1.0 - confidence))
    calibrated = 1.0 / (1.0 + math.exp(-((logit / temperature) + bias)))
    return round(max(0.0, min(1.0, calibrated)), 4)


def _tta_predict(pipe, image, top_k: int = 4) -> list[dict[str, Any]]:
    """Average predictions across simple test-time augmentations."""
    from PIL import Image

    augments = [
        image,
        image.transpose(Image.Transpose.FLIP_LEFT_RIGHT),
        image.rotate(10),
        image.rotate(-10),
    ]
    label_scores: dict[str, float] = {}
    used = 0
    for augmented in augments:
        try:
            preds = pipe(augmented, top_k=top_k)
            used += 1
            for pred in preds:
                label = str(pred.get("label", "unknown"))
                label_scores[label] = label_scores.get(label, 0.0) + float(pred.get("score", 0.0))
        except Exception as exc:
            logger.debug("TTA prediction skipped: %s", exc)
    divisor = used or 1
    results = [
        {"label": label, "score": round(score / divisor, 4)}
        for label, score in label_scores.items()
    ]
    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


def _calibrated_probability(score: float) -> float:
    return _platt_scale(float(score or 0.0))


def _normalize_label(label: str) -> str:
    return str(label or "").strip().lower().replace(" ", "_").replace("-", "_")


def _display_label(raw_label: str, mapping: dict[str, str] | None = None) -> str:
    key = _normalize_label(raw_label)
    if mapping and key in mapping:
        return mapping[key]
    return key.replace("_", " ").title()


def _placeholder(
    *,
    label: str,
    modality: str,
    body_region: str,
    model_used: str = "none",
    primary_finding: str | None = None,
    icd10: str = "Z03.89",
) -> ClassificationResult:
    return ClassificationResult(
        label=label,
        top_class=f"{modality}_unclassified",
        confidence=0.0,
        primary_finding=primary_finding or f"{label} classifier unavailable",
        severity="LOW",
        icd10=icd10,
        body_region=body_region,
        modality=modality,
        model_used=model_used,
    )


def _load_hf_pipeline(kind: str, model_name: str):
    if HF_CLASSIFIER_LOCAL_ONLY:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from transformers import pipeline as hf_pipeline

    return hf_pipeline("image-classification", model=model_name, device=-1)


def _load_mri_pipe():
    global _mri_pipe, _mri_model_id
    if _mri_pipe is not None:
        return _mri_pipe
    for model_name in MRI_MODEL_CHAIN:
        try:
            _mri_pipe = _load_hf_pipeline("mri", model_name)
            _mri_model_id = model_name
            logger.info("[MRI] Classifier loaded: %s", model_name)
            return _mri_pipe
        except Exception as exc:
            logger.warning("[MRI] Model unavailable (%s): %s", model_name, exc)
    logger.error("[MRI] All models in fallback chain failed")
    return None


def _load_skin_pipe():
    global _skin_pipe
    if _skin_pipe is not None:
        return _skin_pipe
    for model_name in (SKIN_MODEL, SKIN_FALLBACK_MODEL):
        try:
            _skin_pipe = _load_hf_pipeline("skin", model_name)
            logger.info("[SKIN] Classifier loaded: %s", model_name)
            return _skin_pipe
        except Exception as exc:
            logger.warning("[SKIN] Model unavailable (%s): %s", model_name, exc)
    return None


def _load_pathology_pipe():
    global _pathology_pipe
    if _pathology_pipe is not None:
        return _pathology_pipe
    try:
        _pathology_pipe = _load_hf_pipeline("pathology", PATHOLOGY_MODEL)
        logger.info("[PATHOLOGY] Classifier loaded: %s", PATHOLOGY_MODEL)
        return _pathology_pipe
    except Exception as exc:
        logger.warning("[PATHOLOGY] Classifier unavailable: %s", exc)
        return None


def classify_mri(image_path: str) -> ClassificationResult:
    from PIL import Image

    pipe = _load_mri_pipe()
    if pipe is None:
        return _placeholder(
            label="MRI Study",
            modality="mri",
            body_region="brain",
            primary_finding="MRI classifier unavailable. Install or cache the configured HuggingFace model.",
        )

    try:
        results = _tta_predict(pipe, Image.open(image_path).convert("RGB"), top_k=5)
        if not results:
            raise RuntimeError("MRI classifier returned no predictions")
        top = results[0]
        raw_label = _normalize_label(top["label"])
        raw_confidence = float(top["score"])
        confidence = _calibrated_probability(raw_confidence)
        display = _display_label(raw_label, MRI_LABELS)
        all_findings = [
            {
                "label": _display_label(item["label"], MRI_LABELS),
                "probability": _calibrated_probability(float(item["score"])),
                "detected": float(item["score"]) > 0.2,
                "severity": _severity(_calibrated_probability(float(item["score"])) * 100, item["label"]),
                "clinical_meaning": MRI_CLINICAL.get(_normalize_label(item["label"]), ""),
                "icd10": MRI_ICD10.get(_normalize_label(item["label"]), "Z03.89"),
            }
            for item in results
        ]
        return ClassificationResult(
            label=display,
            top_class=raw_label,
            confidence=round(confidence, 4),
            probabilities={_display_label(r["label"], MRI_LABELS): _calibrated_probability(float(r["score"])) for r in results},
            primary_finding=display,
            severity=_severity(confidence * 100, raw_label),
            icd10=MRI_ICD10.get(raw_label, "Z03.89"),
            body_region="brain",
            modality="mri",
            model_used=_mri_model_id or MRI_MODEL,
            all_findings=all_findings,
        )
    except Exception as exc:
        logger.error("[MRI] Inference failed: %s", exc)
        return _placeholder(label="MRI Analysis Failed", modality="mri", body_region="brain", model_used="error")


def classify_ct(image_path: str) -> ClassificationResult:
    try:
        import numpy as np
        import torch
        import torchxrayvision as xrv
        from PIL import Image

        image = Image.open(image_path).convert("L").resize((224, 224))
        image_np = np.asarray(image, dtype=np.float32)
        image_np = xrv.datasets.normalize(image_np, 255)[None, ...]
        tensor = torch.from_numpy(image_np[None, ...]).float()

        model = xrv.models.DenseNet(weights="densenet121-res224-nih")
        model.eval()
        with torch.no_grad():
            preds = model(tensor).detach().cpu().numpy()[0]

        labels = [str(label) for label in getattr(model, "pathologies", [])]
        scores = sorted(zip(labels, preds), key=lambda item: float(item[1]), reverse=True)
        top_label, top_score = scores[0] if scores else ("CT Study", 0.0)
        raw_confidence = float(top_score)
        confidence = _calibrated_probability(raw_confidence)
        return ClassificationResult(
            label=top_label,
            top_class=_normalize_label(top_label),
            confidence=round(confidence, 4),
            probabilities={label: _calibrated_probability(float(score)) for label, score in scores[:8]},
            primary_finding=top_label,
            severity=_severity(confidence * 100, top_label),
            icd10="R91.8",
            body_region="chest",
            modality="ct",
            model_used="torchxrayvision-densenet121-res224-nih",
            all_findings=[
                {
                    "label": label,
                    "probability": _calibrated_probability(float(score)),
                    "detected": float(score) > 0.3,
                    "severity": _severity(_calibrated_probability(float(score)) * 100, label),
                    "clinical_meaning": "",
                    "icd10": "R91.8",
                }
                for label, score in scores[:8]
            ],
        )
    except Exception as exc:
        logger.warning("[CT] Classifier unavailable: %s", exc)
        return _placeholder(
            label="CT Study",
            modality="ct",
            body_region="chest",
            primary_finding="CT classifier unavailable. TorchXRayVision fallback could not run.",
            icd10="R91.8",
        )


def classify_skin(image_path: str) -> ClassificationResult:
    from PIL import Image

    pipe = _load_skin_pipe()
    if pipe is None:
        return _placeholder(
            label="Skin Lesion",
            modality="skin",
            body_region="skin",
            primary_finding="Skin classifier unavailable. Install or cache the configured dermatology model.",
            icd10="L98.9",
        )

    try:
        results = _tta_predict(pipe, Image.open(image_path).convert("RGB"), top_k=5)
        if not results:
            raise RuntimeError("Skin classifier returned no predictions")
        top = results[0]
        raw_label = _normalize_label(top["label"])
        raw_confidence = float(top["score"])
        confidence = _calibrated_probability(raw_confidence)
        display = _display_label(top["label"])
        return ClassificationResult(
            label=display,
            top_class=raw_label,
            confidence=round(confidence, 4),
            probabilities={_display_label(r["label"]): _calibrated_probability(float(r["score"])) for r in results},
            primary_finding=display,
            severity=_severity(confidence * 100, raw_label),
            icd10=SKIN_ICD10.get(raw_label, "L98.9"),
            body_region="skin",
            modality="skin",
            model_used=SKIN_MODEL,
            all_findings=[
                {
                    "label": _display_label(item["label"]),
                    "probability": _calibrated_probability(float(item["score"])),
                    "detected": float(item["score"]) > 0.15,
                    "severity": _severity(_calibrated_probability(float(item["score"])) * 100, item["label"]),
                    "clinical_meaning": "",
                    "icd10": SKIN_ICD10.get(_normalize_label(item["label"]), "L98.9"),
                }
                for item in results
            ],
        )
    except Exception as exc:
        logger.error("[SKIN] Inference failed: %s", exc)
        return _placeholder(label="Skin Analysis Failed", modality="skin", body_region="skin", model_used="error", icd10="L98.9")


def classify_pathology(image_path: str) -> ClassificationResult:
    from PIL import Image

    pipe = _load_pathology_pipe()
    if pipe is None:
        return _placeholder(
            label="Pathology Study",
            modality="pathology",
            body_region="tissue",
            primary_finding="Pathology classifier unavailable. Install or cache the configured pathology model.",
            icd10="R89.9",
        )

    try:
        results = pipe(Image.open(image_path).convert("RGB"), top_k=5)
        top = results[0]
        raw_label = _normalize_label(top["label"])
        tier_key = next((key for key in WHO_TIERS if key in raw_label), "atypical")
        tier = WHO_TIERS[tier_key]
        raw_confidence = float(top["score"])
        confidence = _calibrated_probability(raw_confidence)
        primary = f"WHO Tier {tier['tier']}: {tier['label']}"
        return ClassificationResult(
            label=tier["label"],
            top_class=tier_key,
            confidence=round(confidence, 4),
            probabilities={_display_label(r["label"]): _calibrated_probability(float(r["score"])) for r in results},
            primary_finding=primary,
            severity=_severity(confidence * 100, tier["label"]),
            icd10="R89.9",
            body_region="tissue",
            modality="pathology",
            model_used=PATHOLOGY_MODEL,
            all_findings=[
                {
                    "label": _display_label(item["label"]),
                    "probability": _calibrated_probability(float(item["score"])),
                    "detected": float(item["score"]) > 0.1,
                    "severity": _severity(_calibrated_probability(float(item["score"])) * 100, item["label"]),
                    "clinical_meaning": WHO_TIERS.get(tier_key, {}).get("action", ""),
                    "icd10": "R89.9",
                }
                for item in results
            ],
        )
    except Exception as exc:
        logger.error("[PATHOLOGY] Inference failed: %s", exc)
        return _placeholder(label="Pathology Analysis Failed", modality="pathology", body_region="tissue", model_used="error", icd10="R89.9")


def classify_by_modality(
    image_path: str,
    modality: str,
    fallback_classify_fn: Callable[[Any], Any] | None = None,
    fallback_image: Any | None = None,
) -> Any:
    """Route image classification to the modality-appropriate model."""
    normalized = str(modality or "xray").lower().strip()

    if normalized in CHEST_MODALITIES:
        if fallback_classify_fn is None:
            return _placeholder(label="Chest X-Ray", modality="xray", body_region="chest", icd10="R91.8")
        try:
            if fallback_image is None:
                from PIL import Image

                fallback_image = Image.open(image_path).convert("RGB")
            result = fallback_classify_fn(fallback_image)
            try:
                setattr(result, "modality", "xray")
                setattr(result, "body_region", "chest")
                setattr(result, "model_used", getattr(result, "model_version", "chexnet"))
            except Exception:
                pass
            return result
        except Exception as exc:
            logger.warning("[ROUTER] CheXNet fallback failed: %s", exc)
            return _placeholder(label="Chest X-Ray", modality="xray", body_region="chest", model_used="error", icd10="R91.8")

    if normalized in MRI_MODALITIES:
        return classify_mri(image_path)
    if normalized in CT_MODALITIES:
        return classify_ct(image_path)
    if normalized in SKIN_MODALITIES:
        return classify_skin(image_path)
    if normalized in PATH_MODALITIES:
        return classify_pathology(image_path)

    logger.warning("[ROUTER] Unknown image modality '%s'", modality)
    return _placeholder(
        label=f"{str(modality or 'Image').upper()} Study",
        modality=normalized or "unknown",
        body_region="unknown",
        primary_finding=f"Modality '{modality}' classifier is not configured.",
    )
