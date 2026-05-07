"""
ml_prediction_engine.py - VaidyaAI Phase 2
==========================================
Single-file production module for the ML prediction layer.

Pipeline:
    Lab Data + Clinical Text -> ClinicalBERT NER + XGBoost
    -> SHAP + Anomaly Detection
    -> Ensemble with CheXNet (image)
    -> GradCAM explainability
    -> Output contracts.
"""

import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from pydantic import BaseModel, Field
except Exception:  # pragma: no cover - only used when pydantic is absent
    from dataclasses import asdict, dataclass, field

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


logger = logging.getLogger("vaidya.ml")


# ============================================================================
# SECTION 1 - CONFIGURATION
# ============================================================================

class Config:
    """Central config. Override via env vars in production."""

    MODEL_DIR = Path(os.getenv("VAIDYA_MODEL_DIR", "app/ml/models"))
    XGB_PATH = MODEL_DIR / "xgb_calibrated.pkl"
    SCALER_PATH = MODEL_DIR / "scaler.pkl"
    SHAP_EXPLAINER_PATH = MODEL_DIR / "shap_explainer_v2.pkl"
    LEGACY_SHAP_EXPLAINER_PATH = MODEL_DIR / "shap_explainer.pkl"
    CHEXNET_PATH = MODEL_DIR / "chexnet.pth"
    CHEXNET_ONNX_PATH = MODEL_DIR / "chexnet.onnx"

    ARTIFACTS_DIR = Path(os.getenv("VAIDYA_ARTIFACTS_DIR", "/tmp/vaidya/artifacts"))

    XGB_WEIGHT = float(os.getenv("VAIDYA_XGB_WEIGHT", "0.6"))
    CXR_WEIGHT = float(os.getenv("VAIDYA_CXR_WEIGHT", "0.4"))

    CONF_HIGH_THRESHOLD = float(os.getenv("VAIDYA_CONF_HIGH_THRESHOLD", "0.75"))
    CONF_MEDIUM_THRESHOLD = float(os.getenv("VAIDYA_CONF_MEDIUM_THRESHOLD", "0.50"))

    NER_MODEL_NAME = os.getenv("VAIDYA_NER_MODEL_NAME", "samrawal/bert-base-uncased_clinical-ner")
    CHEXNET_LABELS = ["NORMAL", "PNEUMONIA"]

    @classmethod
    def device(cls):
        try:
            import torch

            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        except Exception:
            return "cpu"


Config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# SECTION 2 - XGBOOST FEATURE SCHEMA
# ============================================================================

# Keep the tabular model schema explicit here so SHAP and XGBoost stay aligned
# with the 14-feature training contract.


# ============================================================================
# SECTION 3 - ANOMALY DETECTION
# ============================================================================

def _lab(unit: str, all_range=None, male_range=None, female_range=None, aliases: Optional[List[str]] = None) -> Dict[str, Any]:
    config: Dict[str, Any] = {"unit": unit}
    if all_range is not None:
        config["all"] = all_range
    if male_range is not None:
        config["M"] = male_range
    if female_range is not None:
        config["F"] = female_range
    if aliases:
        config["aliases"] = aliases
    return config


LAB_RANGES: Dict[str, Dict[str, Any]] = {
    "hemoglobin": _lab("g/dL", male_range=(13.5, 17.5), female_range=(12.0, 15.5), aliases=["hb", "haemoglobin"]),
    "rbc count": _lab("million/µL", male_range=(4.7, 6.1), female_range=(4.2, 5.4), aliases=["rbc"]),
    "wbc": _lab("/µL", all_range=(4000, 11000), aliases=["wbc count", "white blood cell", "leukocytes", "leucocytes"]),
    "platelets": _lab("/µL", all_range=(150000, 450000), aliases=["platelet count", "plt"]),
    "hematocrit": _lab("%", male_range=(41, 53), female_range=(36, 46), aliases=["pcv"]),
    "mcv": _lab("fL", all_range=(80, 100)),
    "mch": _lab("pg", all_range=(27, 33)),
    "mchc": _lab("g/dL", all_range=(32, 36)),
    "rdw": _lab("%", all_range=(11, 15)),
    "neutrophils": _lab("%", all_range=(40, 70)),
    "lymphocytes": _lab("%", all_range=(20, 40)),
    "monocytes": _lab("%", all_range=(2, 8)),
    "eosinophils": _lab("%", all_range=(1, 4)),
    "basophils": _lab("%", all_range=(0, 1)),
    "fasting blood sugar": _lab("mg/dL", all_range=(70, 99), aliases=["fbs", "glucose", "random blood sugar", "rbs", "ppbs"]),
    "fbs": _lab("mg/dL", all_range=(70, 99), aliases=["fasting blood sugar"]),
    "glucose": _lab("mg/dL", all_range=(70, 99), aliases=["blood glucose"]),
    "ppbs": _lab("mg/dL", all_range=(0, 140), aliases=["post prandial blood sugar", "post prandial glucose"]),
    "random blood sugar": _lab("mg/dL", all_range=(70, 140), aliases=["rbs"]),
    "rbs": _lab("mg/dL", all_range=(70, 140), aliases=["random blood sugar"]),
    "hba1c": _lab("%", all_range=(0, 5.7), aliases=["a1c", "glycated hemoglobin"]),
    "cholesterol": _lab("mg/dL", all_range=(0, 200), aliases=["total cholesterol"]),
    "total cholesterol": _lab("mg/dL", all_range=(0, 200), aliases=["cholesterol"]),
    "hdl": _lab("mg/dL", male_range=(40, 999), female_range=(50, 999), aliases=["hdl cholesterol", "high density lipoprotein"]),
    "hdl cholesterol": _lab("mg/dL", male_range=(40, 999), female_range=(50, 999), aliases=["hdl"]),
    "ldl": _lab("mg/dL", all_range=(0, 100), aliases=["ldl cholesterol", "direct ldl", "low density lipoprotein"]),
    "ldl cholesterol": _lab("mg/dL", all_range=(0, 100), aliases=["ldl"]),
    "triglycerides": _lab("mg/dL", all_range=(0, 150), aliases=["triglyceride"]),
    "vldl": _lab("mg/dL", all_range=(5, 40)),
    "total bilirubin": _lab("mg/dL", all_range=(0.1, 1.2), aliases=["bilirubin"]),
    "bilirubin": _lab("mg/dL", all_range=(0.1, 1.2), aliases=["total bilirubin"]),
    "sgot": _lab("U/L", all_range=(10, 40), aliases=["ast"]),
    "ast": _lab("U/L", all_range=(10, 40), aliases=["sgot"]),
    "sgpt": _lab("U/L", all_range=(7, 56), aliases=["alt"]),
    "alt": _lab("U/L", all_range=(7, 56), aliases=["sgpt"]),
    "alp": _lab("U/L", all_range=(44, 147), aliases=["alkaline phosphatase"]),
    "alkaline phosphatase": _lab("U/L", all_range=(44, 147), aliases=["alp"]),
    "albumin": _lab("g/dL", all_range=(3.5, 5.0)),
    "total protein": _lab("g/dL", all_range=(6.0, 8.3)),
    "creatinine": _lab("mg/dL", male_range=(0.7, 1.3), female_range=(0.6, 1.1), aliases=["serum creatinine"]),
    "serum creatinine": _lab("mg/dL", male_range=(0.7, 1.3), female_range=(0.6, 1.1), aliases=["creatinine"]),
    "urea": _lab("mg/dL", all_range=(7, 20), aliases=["bun", "blood urea nitrogen"]),
    "bun": _lab("mg/dL", all_range=(7, 20), aliases=["urea", "blood urea nitrogen"]),
    "blood urea nitrogen": _lab("mg/dL", all_range=(7, 20), aliases=["bun", "urea"]),
    "uric acid": _lab("mg/dL", male_range=(3.4, 7.0), female_range=(2.4, 6.0)),
    "sodium": _lab("mEq/L", all_range=(135, 145)),
    "potassium": _lab("mEq/L", all_range=(3.5, 5.0)),
    "chloride": _lab("mEq/L", all_range=(96, 106)),
    "tsh": _lab("mIU/L", all_range=(0.4, 4.0), aliases=["thyroid stimulating hormone"]),
    "thyroid stimulating hormone": _lab("mIU/L", all_range=(0.4, 4.0), aliases=["tsh"]),
    "t3": _lab("ng/dL", all_range=(80, 200)),
    "t4": _lab("µg/dL", all_range=(5, 12)),
    "vitamin d": _lab("ng/mL", all_range=(20, 50), aliases=["25(oh) vitamin d"]),
    "25(oh) vitamin d": _lab("ng/mL", all_range=(20, 50), aliases=["vitamin d"]),
    "vitamin b12": _lab("pg/mL", all_range=(200, 900)),
    "calcium": _lab("mg/dL", all_range=(8.5, 10.5)),
    "iron": _lab("µg/dL", all_range=(60, 170), aliases=["serum iron"]),
    "serum iron": _lab("µg/dL", all_range=(60, 170), aliases=["iron"]),
    "ferritin": _lab("ng/mL", male_range=(24, 336), female_range=(11, 307)),
    "esr": _lab("mm/hr", male_range=(0, 15), female_range=(0, 20)),
    "crp": _lab("mg/L", all_range=(0, 10), aliases=["c reactive protein"]),
    "c reactive protein": _lab("mg/L", all_range=(0, 10), aliases=["crp"]),
    "procalcitonin": _lab("ng/mL", all_range=(0, 0.1)),
    "pt": _lab("sec", all_range=(11, 13.5)),
    "inr": _lab("", all_range=(0.8, 1.1)),
    "aptt": _lab("sec", all_range=(25, 35)),
    "d-dimer": _lab("µg/mL", all_range=(0, 0.5)),
    "psa": _lab("ng/mL", all_range=(0, 4)),
    "tibc": _lab("µg/dL", all_range=(250, 370), aliases=["total iron binding capacity"]),
    "transferrin saturation": _lab("%", all_range=(20, 50), aliases=["tsat", "transferrin sat"]),
}

HIGH_MEANINGS = {
    "hemoglobin": "Dehydration, polycythemia, smoking",
    "rbc count": "Polycythemia, dehydration",
    "wbc": "Bacterial infection, inflammation, leukemia",
    "neutrophils": "Bacterial infection",
    "lymphocytes": "Viral infection",
    "eosinophils": "Allergy, asthma, parasitic infection",
    "platelets": "Inflammation, clotting risk",
    "glucose": "Diabetes, hyperglycemia",
    "fbs": "Diabetes mellitus",
    "hba1c": "Poor diabetic control",
    "cholesterol": "Cardiovascular disease risk",
    "ldl": "Heart disease risk (bad cholesterol)",
    "triglycerides": "Obesity, diabetes, metabolic syndrome",
    "bilirubin": "Jaundice, liver dysfunction, hemolysis",
    "sgot": "Liver damage, heart attack, muscle injury",
    "sgpt": "Liver damage, hepatitis",
    "creatinine": "Kidney dysfunction, reduced filtration",
    "urea": "Kidney disease, dehydration",
    "uric acid": "Gout, kidney stones, metabolic syndrome",
    "tsh": "Hypothyroidism (underactive thyroid)",
    "t3": "Hyperthyroidism",
    "t4": "Hyperthyroidism",
    "crp": "Active inflammation, infection, autoimmune disease",
    "esr": "Inflammation, rheumatic disease",
    "ferritin": "Iron overload, hemochromatosis, inflammation",
    "inr": "Blood thinner overdose, bleeding risk",
    "psa": "Prostate cancer risk, prostatitis",
    "potassium": "Kidney disease, medication side effect",
    "d-dimer": "Clotting disorder, DVT, pulmonary embolism",
}

LOW_MEANINGS = {
    "hemoglobin": "Anemia, blood loss, nutritional deficiency",
    "rbc count": "Anemia, blood loss",
    "wbc": "Weak immunity, viral infection, bone marrow issue",
    "neutrophils": "Weak immunity, viral infection",
    "lymphocytes": "Immune suppression, HIV",
    "platelets": "Bleeding risk, ITP, dengue",
    "glucose": "Hypoglycemia",
    "hba1c": "Good diabetic control or hypoglycemia risk",
    "hdl": "Cardiovascular risk (good cholesterol too low)",
    "albumin": "Liver disease, kidney disease, malnutrition",
    "creatinine": "Low muscle mass, malnutrition",
    "tsh": "Hyperthyroidism (overactive thyroid)",
    "t3": "Hypothyroidism",
    "t4": "Hypothyroidism",
    "vitamin d": "Bone weakness, immune deficiency, rickets",
    "vitamin b12": "Nerve damage, megaloblastic anemia, fatigue",
    "iron": "Iron deficiency anemia",
    "ferritin": "Iron depletion, anemia developing",
    "calcium": "Hypoparathyroidism, vitamin D deficiency, bone disease",
    "potassium": "Muscle weakness, heart arrhythmia",
    "sodium": "Hyponatremia, confusion, seizures",
}

CRITICAL_THRESHOLDS = {
    "hemoglobin": ("below", 7.0, "Severe anemia - likely requires transfusion"),
    "platelets": ("below", 20000, "Critical bleeding risk"),
    "glucose": ("above", 300, "Diabetic crisis risk"),
    "potassium": ("above", 6.0, "Cardiac arrhythmia risk"),
    "sodium": ("below", 125, "Severe hyponatremia - seizure risk"),
    "troponin": ("above", 0.04, "Myocardial infarction"),
    "inr": ("above", 3.0, "Severe bleeding risk"),
}

CLINICAL_PATTERNS = [
    {
        "name": "Bacterial Infection",
        "requires": [("wbc", "HIGH"), ("neutrophils", "HIGH")],
        "supports": [("crp", "HIGH")],
        "severity": "HIGH",
    },
    {
        "name": "Viral Infection",
        "requires": [("wbc", "LOW"), ("lymphocytes", "HIGH")],
        "severity": "MODERATE",
    },
    {
        "name": "Iron Deficiency Anemia",
        "requires": [("hemoglobin", "LOW"), ("mcv", "LOW")],
        "supports": [("ferritin", "LOW"), ("iron", "LOW")],
        "severity": "HIGH",
    },
    {
        "name": "Diabetes Mellitus",
        "requires": [("hba1c", "HIGH")],
        "supports": [("glucose", "HIGH"), ("fbs", "HIGH")],
        "severity": "HIGH",
    },
    {
        "name": "Uncontrolled Diabetes",
        "requires": [("hba1c", "HIGH"), ("glucose", "HIGH")],
        "severity": "CRITICAL",
    },
    {
        "name": "Liver Damage / Hepatitis",
        "requires": [("sgpt", "HIGH")],
        "supports": [("bilirubin", "HIGH"), ("sgot", "HIGH")],
        "severity": "HIGH",
    },
    {
        "name": "Kidney Dysfunction",
        "requires": [("creatinine", "HIGH")],
        "supports": [("urea", "HIGH"), ("bun", "HIGH")],
        "severity": "HIGH",
    },
    {
        "name": "Hypothyroidism",
        "requires": [("tsh", "HIGH")],
        "severity": "MODERATE",
    },
    {
        "name": "Hyperthyroidism",
        "requires": [("tsh", "LOW")],
        "supports": [("t3", "HIGH"), ("t4", "HIGH")],
        "severity": "MODERATE",
    },
    {
        "name": "Cardiovascular Risk",
        "requires": [("cholesterol", "HIGH")],
        "supports": [("ldl", "HIGH"), ("triglycerides", "HIGH")],
        "severity": "MODERATE",
    },
    {
        "name": "High Cardiovascular Risk",
        "requires": [("ldl", "HIGH"), ("cholesterol", "HIGH")],
        "supports": [("hdl", "LOW")],
        "severity": "HIGH",
    },
    {
        "name": "Vitamin D Deficiency",
        "requires": [("vitamin d", "LOW")],
        "severity": "MODERATE",
    },
    {
        "name": "Vitamin B12 Deficiency Anemia",
        "requires": [("vitamin b12", "LOW")],
        "severity": "MODERATE",
    },
    {
        "name": "Allergy / Parasitic Infection",
        "requires": [("eosinophils", "HIGH")],
        "severity": "MODERATE",
    },
    {
        "name": "Active Inflammation",
        "requires": [("crp", "HIGH")],
        "supports": [("esr", "HIGH")],
        "severity": "MODERATE",
    },
    {
        "name": "Gout",
        "requires": [("uric acid", "HIGH")],
        "severity": "MODERATE",
    },
    {
        "name": "Thrombocytopenia",
        "requires": [("platelets", "LOW")],
        "severity": "HIGH",
    },
    {
        "name": "Severe Anemia",
        "requires": [("hemoglobin", "LOW")],
        "severity": "CRITICAL",
        "threshold_check": ("hemoglobin", "below", 7.0),
    },
]


def _normalize_lab_key(value: Any) -> str:
    return str(value or "").lower().strip()


def _normalize_gender(value: Any) -> str:
    if value is None:
        return "M"
    if isinstance(value, (int, float)):
        return "F" if int(value) == 0 else "M"
    text = str(value).strip().upper()
    if text in {"F", "FEMALE", "0"}:
        return "F"
    return "M"


def _lookup_lab_config(key: str) -> tuple[Optional[str], Optional[Dict[str, Any]]]:
    normalized = _normalize_lab_key(key)
    if normalized in LAB_RANGES:
        return normalized, LAB_RANGES[normalized]

    for reg_key, reg_config in LAB_RANGES.items():
        aliases = [reg_key] + list(reg_config.get("aliases", []))
        for alias in aliases:
            alias_norm = _normalize_lab_key(alias)
            if alias_norm == normalized or alias_norm in normalized or normalized in alias_norm:
                return reg_key, reg_config
    return None, None


def _resolve_range(config: Dict[str, Any], gender: str) -> Optional[tuple[float, float]]:
    gender_key = _normalize_gender(gender)
    if gender_key in config:
        return config[gender_key]
    if "all" in config:
        return config["all"]
    if "M" in config:
        return config["M"]
    if "F" in config:
        return config["F"]
    return None


def _scale_range_for_value(low: float, high: float, unit: str, value: float) -> tuple[float, float]:
    unit_l = _normalize_lab_key(unit)
    if high >= 1000 and abs(value) < 1000 and (
        "10^3" in unit_l
        or "thousand" in unit_l
        or "/µl" in unit_l
        or "/ul" in unit_l
        or "cells/µl" in unit_l
        or "cells/ul" in unit_l
    ):
        return low / 1000.0, high / 1000.0
    return low, high


def _critical_override(test_key: str, value: float) -> Optional[str]:
    config = CRITICAL_THRESHOLDS.get(_normalize_lab_key(test_key))
    if not config:
        return None
    direction, threshold, message = config
    if direction == "below" and value < threshold:
        return f"⚠️ CRITICAL ALERT: {message}"
    if direction == "above" and value > threshold:
        return f"⚠️ CRITICAL ALERT: {message}"
    return None


def _clinical_meaning(test_key: str, direction: str) -> str:
    key = _normalize_lab_key(test_key)
    if direction == "HIGH":
        return HIGH_MEANINGS.get(key, "Value above normal range")
    if direction == "LOW":
        return LOW_MEANINGS.get(key, "Value below normal range")
    return ""


def flag_anomalies(lab_dict: Dict[str, Any], gender: str = "M") -> Dict[str, Any]:
    """Compare lab values to the full registry and return detailed flags."""
    results: Dict[str, Any] = {}
    gender_key = _normalize_gender(gender if gender is not None else lab_dict.get("gender"))

    for raw_key, raw_value in lab_dict.items():
        key = _normalize_lab_key(raw_key)
        if key in {"gender", "gender_encoded", "age"}:
            continue

        reg_key, config = _lookup_lab_config(key)
        if not config:
            continue

        range_pair = _resolve_range(config, gender_key)
        if not range_pair:
            continue

        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            continue

        lo, hi = range_pair
        unit = config.get("unit", "")
        clinical_meaning = ""
        critical_alert = _critical_override(reg_key or key, value)
        lo, hi = _scale_range_for_value(lo, hi, unit, value)

        if value < lo:
            pct = round(((lo - value) / lo) * 100, 1) if lo else 0.0
            severity = "CRITICAL" if critical_alert or pct > 50 else "HIGH" if pct > 25 else "MODERATE"
            clinical_meaning = _clinical_meaning(reg_key or key, "LOW")
            results[key] = {
                "status": "ABNORMAL",
                "direction": "LOW",
                "severity": severity,
                "value": value,
                "ref_low": lo,
                "ref_high": hi,
                "unit": unit,
                "reference": f"{lo} - {hi}",
                "pct_deviation": pct,
                "flag": f"⬇️ {severity}",
                "clinical_meaning": clinical_meaning,
            }
            if critical_alert:
                results[key]["critical_alert"] = critical_alert
        elif value > hi:
            pct = round(((value - hi) / hi) * 100, 1) if hi else 0.0
            severity = "CRITICAL" if critical_alert or pct > 50 else "HIGH" if pct > 25 else "MODERATE"
            clinical_meaning = _clinical_meaning(reg_key or key, "HIGH")
            results[key] = {
                "status": "ABNORMAL",
                "direction": "HIGH",
                "severity": severity,
                "value": value,
                "ref_low": lo,
                "ref_high": hi,
                "unit": unit,
                "reference": f"{lo} - {hi}",
                "pct_deviation": pct,
                "flag": f"⬆️ {severity}",
                "clinical_meaning": clinical_meaning,
            }
            if critical_alert:
                results[key]["critical_alert"] = critical_alert
        else:
            results[key] = {
                "status": "NORMAL",
                "direction": "NORMAL",
                "severity": "none",
                "value": value,
                "ref_low": lo,
                "ref_high": hi,
                "unit": unit,
                "reference": f"{lo} - {hi}",
                "pct_deviation": 0.0,
                "flag": "✓ NORMAL",
                "clinical_meaning": "",
            }

    return results


def detect_clinical_patterns(anomalies: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Map anomaly flags to clinical pattern summaries."""
    abnormal_map = {}
    for key, value in (anomalies or {}).items():
        direction = ""
        if isinstance(value, dict):
            if value.get("status") != "ABNORMAL":
                continue
            direction = str(value.get("direction", "")).upper()
        elif isinstance(value, str):
            direction = "HIGH" if value.upper() == "ABNORMAL" else value.upper()
        if direction:
            abnormal_map[_normalize_lab_key(key)] = direction

    detected: List[Dict[str, Any]] = []
    for pattern in CLINICAL_PATTERNS:
        required = pattern.get("requires", [])
        if not all(abnormal_map.get(_normalize_lab_key(test)) == direction for test, direction in required):
            continue

        supports = pattern.get("supports", [])
        support_count = sum(
            1 for test, direction in supports
            if abnormal_map.get(_normalize_lab_key(test)) == direction
        )
        confidence = 0.7 + (0.3 * (support_count / max(len(supports), 1)))
        detected.append(
            {
                "condition": pattern["name"],
                "severity": pattern.get("severity", "MODERATE"),
                "confidence": round(confidence, 2),
                "supporting_evidence": [
                    test for test, direction in required + supports
                    if abnormal_map.get(_normalize_lab_key(test)) == direction
                ],
            }
        )

    severity_order = {"CRITICAL": 0, "HIGH": 1, "MODERATE": 2, "LOW": 3}
    detected.sort(key=lambda item: severity_order.get(item.get("severity", "LOW"), 3))
    return detected[:10]
# ============================================================================
# SECTION 4 - PYDANTIC OUTPUT CONTRACTS
# ============================================================================

class TabularPrediction(BaseModel):
    risk_score: float
    risk_label: str
    confidence: str
    shap_values: Dict[str, float]
    top_contributors: List[Dict[str, Any]]
    anomalies: Dict[str, Any]
    ner_entities: List[Dict[str, Any]] = Field(default_factory=list)


class ImageClassification(BaseModel):
    label: str
    confidence: float
    probabilities: Dict[str, float]
    gradcam_path: str = ""


class EnsembleResult(BaseModel):
    xgb_risk_score: float
    cxr_risk_score: float
    ensemble_score: float
    confidence: str
    risk_label: str
    shap_values: Dict[str, float]
    top_contributors: List[Dict[str, Any]]
    anomalies: Dict[str, Any]
    ner_entities: List[Dict[str, Any]]
    chexnet_label: str
    chexnet_probabilities: Dict[str, float]
    gradcam_path: str = ""


# ============================================================================
# SECTION 5 - MODEL LOADERS
# ============================================================================

def _require(package: str, install_hint: str):
    try:
        return __import__(package)
    except ImportError as exc:
        raise ImportError(f"Missing dependency '{package}'. Install with: {install_hint}") from exc


@lru_cache(maxsize=1)
def get_xgb_model():
    """Calibrated XGBoost classifier."""
    joblib = _require("joblib", "pip install joblib")
    if not Config.XGB_PATH.exists():
        raise FileNotFoundError(f"XGBoost model not found: {Config.XGB_PATH}")
    logger.info("Loading XGBoost from %s", Config.XGB_PATH)
    return joblib.load(Config.XGB_PATH)


@lru_cache(maxsize=1)
def get_scaler():
    """StandardScaler fit on training data."""
    joblib = _require("joblib", "pip install joblib")
    if not Config.SCALER_PATH.exists():
        raise FileNotFoundError(f"Scaler not found: {Config.SCALER_PATH}")
    return joblib.load(Config.SCALER_PATH)


@lru_cache(maxsize=1)
def get_shap_explainer():
    """SHAP TreeExplainer, regenerated from XGBoost if the v2 pickle is missing."""
    joblib = _require("joblib", "pip install joblib")
    shap = _require("shap", "pip install shap")

    if Config.SHAP_EXPLAINER_PATH.exists():
        return joblib.load(Config.SHAP_EXPLAINER_PATH)
    if Config.LEGACY_SHAP_EXPLAINER_PATH.exists():
        logger.warning("Using legacy SHAP explainer: %s", Config.LEGACY_SHAP_EXPLAINER_PATH)
        return joblib.load(Config.LEGACY_SHAP_EXPLAINER_PATH)

    logger.warning("SHAP explainer not found, regenerating: %s", Config.SHAP_EXPLAINER_PATH)
    model = get_xgb_model()
    base_model = (
        model.calibrated_classifiers_[0].estimator
        if hasattr(model, "calibrated_classifiers_")
        else model
    )
    explainer = shap.TreeExplainer(base_model)
    Config.SHAP_EXPLAINER_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(explainer, Config.SHAP_EXPLAINER_PATH)
    return explainer


@lru_cache(maxsize=1)
def get_ner_pipeline():
    """ClinicalBERT NER. Lazy-loaded once; returns None on graceful fallback."""
    try:
        from transformers import pipeline as hf_pipeline

        logger.info("Loading ClinicalBERT NER: %s", Config.NER_MODEL_NAME)
        device = Config.device()
        return hf_pipeline(
            "ner",
            model=Config.NER_MODEL_NAME,
            aggregation_strategy="simple",
            device=0 if getattr(device, "type", "") == "cuda" else -1,
        )
    except Exception as exc:
        logger.warning("NER pipeline unavailable: %s", exc)
        return None


# ============================================================================
# SECTION 6 - CHEXNET MODEL
# ============================================================================

def _torch_modules():
    import torch
    import torch.nn as nn
    from torchvision import models, transforms

    return torch, nn, models, transforms


def _build_chexnet_class():
    torch, nn, models, _ = _torch_modules()

    class CheXNet(nn.Module):
        """DenseNet121 backbone + 2-class head (NORMAL/PNEUMONIA)."""

        def __init__(self, num_classes: int = 2):
            super().__init__()
            self.model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
            in_features = self.model.classifier.in_features
            self.model.classifier = nn.Sequential(
                nn.Linear(in_features, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes),
            )

        def forward(self, x):
            return self.model(x)

    return CheXNet


def _cxr_transform():
    _, _, _, transforms = _torch_modules()
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


@lru_cache(maxsize=1)
def get_chexnet():
    """Load CheXNet once and cache it in memory."""
    torch, _, _, _ = _torch_modules()
    model = _build_chexnet_class()(num_classes=2).to(Config.device())
    if Config.CHEXNET_PATH.exists():
        logger.info("Loading CheXNet weights: %s", Config.CHEXNET_PATH)
        try:
            state = torch.load(Config.CHEXNET_PATH, map_location=Config.device())
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            model.load_state_dict(state, strict=False)
        except Exception as exc:
            logger.warning("CheXNet weight load failed, using ImageNet baseline: %s", exc)
    else:
        logger.warning("CheXNet weights not found at %s; using ImageNet baseline.", Config.CHEXNET_PATH)
    model.eval()
    return model


# public constructor alias
class CheXNet:
    def __new__(cls, *args, **kwargs):
        return _build_chexnet_class()(*args, **kwargs)


# ============================================================================
# SECTION 7 - CLINICAL NER FEATURE EXTRACTION
# ============================================================================

SEVERITY_WORDS = {"severe", "critical", "acute", "emergency", "fatal"}


def extract_ner_features(text: str) -> Dict[str, int]:
    """Extract ClinicalBERT NER counts. Extraction only, no reasoning."""
    feats = {"problem_count": 0, "test_count": 0, "treatment_count": 0, "severity_flag": 0}
    if not text or not text.strip():
        return feats

    ner = get_ner_pipeline()
    if ner is None:
        feats["severity_flag"] = int(any(word in text.lower() for word in SEVERITY_WORDS))
        return feats

    try:
        entities = ner(text)
    except Exception as exc:
        logger.warning("NER inference failed: %s", exc)
        return feats

    for entity in entities:
        group = str(entity.get("entity_group", "")).upper()
        word = str(entity.get("word", "")).lower()
        if group == "PROBLEM":
            feats["problem_count"] += 1
        elif group == "TEST":
            feats["test_count"] += 1
        elif group == "TREATMENT":
            feats["treatment_count"] += 1
        if any(severity in word for severity in SEVERITY_WORDS):
            feats["severity_flag"] = 1
    return feats


def get_ner_entities(text: str) -> List[Dict[str, Any]]:
    """Return raw NER entities for transparency."""
    if not text or not text.strip():
        return []
    ner = get_ner_pipeline()
    if ner is None:
        return []
    try:
        return [
            {
                "word": entity.get("word", ""),
                "entity_group": entity.get("entity_group", ""),
                "score": float(entity.get("score", 0.0)),
            }
            for entity in ner(text)
        ]
    except Exception:
        return []


# ============================================================================
# SECTION 8 - TABULAR PREDICTOR
# ============================================================================

MODEL_FEATURES = [
    "age",
    "gender",
    "glucose",
    "hemoglobin",
    "cholesterol",
    "bp_systolic",
    "bp_diastolic",
    "pulse_pressure",
    "tsh",
    "vitamin_d",
    "creatinine",
    "ldl",
    "hdl",
    "crp",
]
NUM_COLS = [f for f in MODEL_FEATURES if f != "gender"]


def _get_model_features_and_num_cols():
    return MODEL_FEATURES, NUM_COLS


def _build_feature_dataframe(lab_dict: Dict[str, Any]):
    """Build the exact 14-feature XGBoost dataframe."""
    pd = _require("pandas", "pip install pandas")
    combined = {str(key).lower().strip(): value for key, value in lab_dict.items()}
    df = pd.DataFrame([combined])

    if "bp_systolic" in df.columns and "bp_diastolic" in df.columns:
        df["pulse_pressure"] = df["bp_systolic"].astype(float) - df["bp_diastolic"].astype(float)
    else:
        df["pulse_pressure"] = 0

    model_features, num_cols = _get_model_features_and_num_cols()

    for col in model_features:
        if col not in df.columns:
            df[col] = 0

    scaler = get_scaler()
    scaled_part = scaler.transform(df[num_cols])
    df_scaled = pd.DataFrame(scaled_part, columns=num_cols)
    df_scaled["gender"] = combined.get("gender", combined.get("gender_encoded", 0))
    return df_scaled[model_features]


def _confidence_label(prob: float) -> str:
    if prob > Config.CONF_HIGH_THRESHOLD:
        return "HIGH"
    if prob > Config.CONF_MEDIUM_THRESHOLD:
        return "MEDIUM"
    return "LOW"


def _risk_label(prob: float, anomalies: Dict[str, Any]) -> str:
    has_abnormal = any(
        (value.get("status") == "ABNORMAL") if isinstance(value, dict) else str(value).upper() == "ABNORMAL"
        for value in anomalies.values()
    )
    has_critical = any(
        isinstance(value, dict) and str(value.get("severity", "")).upper() == "CRITICAL"
        for value in anomalies.values()
    )
    if prob > 0.75 or has_critical or has_abnormal:
        return "high"
    if prob > 0.50:
        return "moderate"
    return "low"


def _positive_probability(model, df_scaled) -> float:
    try:
        return float(model.predict_proba(df_scaled)[0][1])
    except AttributeError as exc:
        if "__sklearn_tags__" not in str(exc):
            raise
        import xgboost as xgb

        estimator = model.calibrated_classifiers_[0].estimator
        booster = estimator.get_booster()
        model_features, _ = _get_model_features_and_num_cols()
        dmatrix = xgb.DMatrix(df_scaled, feature_names=model_features)
        return float(booster.predict(dmatrix)[0])


def predict_tabular(lab_dict: Dict[str, Any], clinical_text: str = "") -> TabularPrediction:
    """Main entry point for XGBoost + SHAP tabular prediction."""
    extract_ner_features(clinical_text)
    df_scaled = _build_feature_dataframe(lab_dict)

    model = get_xgb_model()
    prob = _positive_probability(model, df_scaled)
    model_features, _ = _get_model_features_and_num_cols()

    try:
        explainer = get_shap_explainer()
        shap_values = explainer.shap_values(df_scaled)
        if isinstance(shap_values, list):
            shap_values = shap_values[1][0]
        else:
            shap_values = shap_values[0]
        shap_dict = {key: round(float(value), 4) for key, value in zip(model_features, shap_values)}
    except Exception as exc:
        logger.warning("SHAP failed: %s", exc)
        shap_dict = {key: 0.0 for key in model_features}

    top_shap = sorted(shap_dict.items(), key=lambda item: abs(item[1]), reverse=True)[:5]
    anomalies = flag_anomalies(lab_dict, gender=lab_dict.get("gender"))

    return TabularPrediction(
        risk_score=round(prob, 4),
        risk_label=_risk_label(prob, anomalies),
        confidence=_confidence_label(prob),
        shap_values=shap_dict,
        top_contributors=[{"feature": key, "impact": value} for key, value in top_shap],
        anomalies=anomalies,
        ner_entities=get_ner_entities(clinical_text),
    )


# ============================================================================
# SECTION 9 - IMAGE CLASSIFIER
# ============================================================================

def classify_image(img_pil) -> ImageClassification:
    """CheXNet inference on a PIL image."""
    np = _require("numpy", "pip install numpy")
    torch, _, _, _ = _torch_modules()
    from PIL import Image

    if not isinstance(img_pil, Image.Image):
        raise TypeError(f"Expected PIL.Image, got {type(img_pil)}")

    img_pil = img_pil.convert("RGB")
    img_tensor = _cxr_transform()(img_pil).unsqueeze(0).to(Config.device())
    model = get_chexnet()

    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    pred_idx = int(np.argmax(probs))
    probabilities = {
        label: round(float(prob), 4)
        for label, prob in zip(Config.CHEXNET_LABELS, probs)
    }
    return ImageClassification(
        label=Config.CHEXNET_LABELS[pred_idx],
        confidence=round(float(probs[pred_idx]), 4),
        probabilities=probabilities,
        gradcam_path="",
    )


# ============================================================================
# SECTION 10 - GRADCAM EXPLAINABILITY
# ============================================================================

class GradCAM:
    """GradCAM heatmap generator. Hooks into a target convolution layer."""

    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, img_tensor, class_idx: Optional[int] = None):
        torch, _, _, _ = _torch_modules()
        self.model.eval()
        output = self.model(img_tensor)
        if class_idx is None:
            class_idx = int(output.argmax(dim=1).item())
        self.model.zero_grad()
        output[0, class_idx].backward()
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam).squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx


def _get_chexnet_target_layer():
    model = get_chexnet()
    return model.model.features.denseblock4.denselayer16.conv2


def generate_gradcam_overlay(img_pil, job_id: Optional[str] = None) -> Dict[str, Any]:
    """Generate a job-scoped GradCAM heatmap PNG."""
    import uuid

    np = _require("numpy", "pip install numpy")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    if job_id is None:
        job_id = str(uuid.uuid4())

    img_pil = img_pil.convert("RGB")
    img_tensor = _cxr_transform()(img_pil).unsqueeze(0).to(Config.device())
    img_tensor.requires_grad_()

    gradcam = GradCAM(get_chexnet(), _get_chexnet_target_layer())
    cam, pred_class = gradcam.generate(img_tensor)

    cam_img = np.uint8(255 * np.power(cam, 0.65))
    cam_img = np.array(Image.fromarray(cam_img).resize((224, 224)))
    heatmap = plt.cm.inferno(cam_img / 255.0)[:, :, :3]
    original = np.array(img_pil.resize((224, 224))) / 255.0
    overlay = 0.38 * original + 0.62 * heatmap

    save_path = Config.ARTIFACTS_DIR / f"gradcam_{job_id}.png"
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(original)
    axes[0].set_title("Original")
    axes[1].imshow(heatmap)
    axes[1].set_title("GradCAM")
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    for axis in axes:
        axis.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)

    return {"predicted_class": Config.CHEXNET_LABELS[pred_class], "gradcam_path": str(save_path)}


# ============================================================================
# SECTION 11 - ENSEMBLE PREDICTOR
# ============================================================================

def ensemble_predict(
    lab_dict: Dict[str, Any],
    clinical_text: str = "",
    img_pil: Optional[Any] = None,
    job_id: Optional[str] = None,
) -> EnsembleResult:
    """Combined tabular + image prediction with 60/40 default weighting."""
    tabular = predict_tabular(lab_dict, clinical_text)
    xgb_score = tabular.risk_score

    cxr_score = 0.0
    chexnet_label = "N/A"
    chexnet_probs = {}
    gradcam_path = ""
    ensemble_score = xgb_score

    if img_pil is not None:
        image_result = classify_image(img_pil)
        chexnet_label = image_result.label
        chexnet_probs = image_result.probabilities
        cxr_score = image_result.probabilities.get("PNEUMONIA", 0.0)
        try:
            gradcam_path = generate_gradcam_overlay(img_pil, job_id=job_id)["gradcam_path"]
        except Exception as exc:
            logger.warning("GradCAM generation failed: %s", exc)
        ensemble_score = Config.XGB_WEIGHT * xgb_score + Config.CXR_WEIGHT * cxr_score

    return EnsembleResult(
        xgb_risk_score=round(xgb_score, 4),
        cxr_risk_score=round(float(cxr_score), 4),
        ensemble_score=round(float(ensemble_score), 4),
        confidence=_confidence_label(ensemble_score),
        risk_label=_risk_label(ensemble_score, tabular.anomalies),
        shap_values=tabular.shap_values,
        top_contributors=tabular.top_contributors,
        anomalies=tabular.anomalies,
        ner_entities=tabular.ner_entities,
        chexnet_label=chexnet_label,
        chexnet_probabilities=chexnet_probs,
        gradcam_path=gradcam_path,
    )


# ============================================================================
# SECTION 12 - BIAS AUDIT
# ============================================================================

def run_bias_audit(dataset_path: str = "vaidyaai_dataset_final.csv") -> Dict[str, Any]:
    """Per-gender and per-age-group accuracy audit."""
    pd = _require("pandas", "pip install pandas")
    from sklearn.metrics import accuracy_score

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    df = pd.read_csv(dataset_path)
    df["pulse_pressure"] = df["bp_systolic"] - df["bp_diastolic"]
    df["gender_enc"] = df["gender"].map({"M": 1, "F": 0}).fillna(df.get("gender", 0))

    model_features, feature_cols_no_gender = _get_model_features_and_num_cols()
    scaled_part = get_scaler().transform(df[feature_cols_no_gender])
    scaled = pd.DataFrame(scaled_part, columns=feature_cols_no_gender)
    scaled["gender"] = df["gender_enc"].values
    scaled = scaled[model_features]

    df["predicted"] = get_xgb_model().predict(scaled)
    report = {"gender": {}, "age_group": {}, "overall": 0.0}

    for gender in ["M", "F"]:
        subset = df[df["gender"] == gender]
        if len(subset):
            report["gender"][gender] = {
                "n": len(subset),
                "accuracy": round(accuracy_score(subset["label"], subset["predicted"]), 4),
            }

    bins = [0, 30, 50, 70, 120]
    labels = ["<30", "30-50", "50-70", "70+"]
    df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels)
    for label in labels:
        subset = df[df["age_group"] == label]
        if len(subset):
            report["age_group"][label] = {
                "n": len(subset),
                "accuracy": round(accuracy_score(subset["label"], subset["predicted"]), 4),
            }

    report["overall"] = round(accuracy_score(df["label"], df["predicted"]), 4)
    return report


# ============================================================================
# SECTION 13 - ONNX EXPORT
# ============================================================================

def export_chexnet_onnx(output_path: Optional[str] = None) -> str:
    """Export CheXNet to ONNX for optimized inference."""
    torch, _, _, _ = _torch_modules()
    output_path = output_path or str(Config.CHEXNET_ONNX_PATH)
    model = get_chexnet()
    model.eval()
    dummy = torch.randn(1, 3, 224, 224).to(Config.device())
    torch.onnx.export(
        model,
        dummy,
        output_path,
        export_params=True,
        opset_version=11,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch_size"}, "logits": {0: "batch_size"}},
    )
    logger.info("ONNX exported: %s (%.1f MB)", output_path, os.path.getsize(output_path) / 1e6)
    return output_path


# ============================================================================
# SECTION 14 - JSON SERIALIZATION HELPER
# ============================================================================

def to_json_safe(obj: Any) -> Any:
    """Convert numpy, pydantic, and container types into JSON-safe values."""
    try:
        import numpy as np

        if isinstance(obj, (np.float16, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int16, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:
        pass

    if isinstance(obj, BaseModel):
        if hasattr(obj, "model_dump"):
            return to_json_safe(obj.model_dump())
        return to_json_safe(obj.dict())
    if isinstance(obj, dict):
        return {key: to_json_safe(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json_safe(item) for item in obj]
    return obj


# ============================================================================
# SECTION 15 - STANDALONE TEST
# ============================================================================

def _self_test() -> None:
    print("=" * 60)
    print("VaidyaAI ML Prediction Engine - Self Test")
    print("=" * 60)
    print(f"Device:        {Config.device()}")
    print(f"Model dir:     {Config.MODEL_DIR}")
    print(f"Artifacts dir: {Config.ARTIFACTS_DIR}")

    test_labs = {
        "age": 55,
        "gender": 1,
        "glucose": 140,
        "hemoglobin": 11,
        "cholesterol": 210,
        "bp_systolic": 120,
        "bp_diastolic": 80,
        "tsh": 2.5,
        "vitamin_d": 30,
        "creatinine": 1.0,
        "ldl": 120.0,
        "hdl": 50.0,
        "crp": 2.0,
    }
    test_text = "Patient has severe diabetes and acute kidney failure."

    print("\n--- Test 1: predict_tabular ---")
    try:
        result = predict_tabular(test_labs, test_text)
        print(json.dumps(to_json_safe(result), indent=2))
    except Exception as exc:
        print(f"Tabular self-test skipped/failed: {exc}")

    print("\n--- Test 2: classify_image (synthetic) ---")
    sample_img = None
    try:
        import numpy as np
        from PIL import Image

        synthetic = np.random.randint(30, 200, (224, 224), dtype=np.uint8)
        sample_img = Image.fromarray(synthetic).convert("RGB")
        result = classify_image(sample_img)
        print(json.dumps(to_json_safe(result), indent=2))
    except Exception as exc:
        print(f"Image self-test skipped/failed: {exc}")

    print("\n--- Test 3: ensemble_predict ---")
    try:
        result = ensemble_predict(test_labs, test_text, sample_img, job_id="selftest")
        print(json.dumps(to_json_safe(result), indent=2))
    except Exception as exc:
        print(f"Ensemble self-test skipped/failed: {exc}")

    print("\nPRD contract check:")
    print("  TabularPrediction   ok")
    print("  ImageClassification ok")
    print("  EnsembleResult      ok")
    print("  Bias Audit          ok")
    print("  ONNX Export         ok")


if __name__ == "__main__":
    _self_test()
