"""Phase 3 ensemble predictor for report risk scoring."""

from __future__ import annotations

import logging
import math
from typing import Any

from app.ml.anomaly.lab_flagger import flag_anomalies
from app.ml.predictor import predict_safe


log = logging.getLogger(__name__)


RISK_THRESHOLDS = {
    "low": (0, 30),
    "moderate": (30, 60),
    "high": (60, 80),
    "critical": (80, 101),
}


def _risk_label(score: float) -> str:
    for label, (low, high) in RISK_THRESHOLDS.items():
        if low <= score < high:
            return label
    return "critical"


def _platt_scale(raw: float) -> float:
    try:
        return round(1 / (1 + math.exp(-0.1 * (float(raw) - 50))) * 100, 1)
    except Exception:
        return 50.0


def _anomaly_score(anomalies: list[dict[str, Any]]) -> float:
    critical = sum(1 for item in anomalies if str(item.get("severity", "")).upper() == "CRITICAL")
    high = sum(1 for item in anomalies if str(item.get("severity", "")).upper() == "HIGH")
    moderate = sum(1 for item in anomalies if str(item.get("severity", "")).upper() == "MODERATE")
    return min(95.0, critical * 30.0 + high * 15.0 + moderate * 5.0)


def _clinical_score(level: str | None) -> float | None:
    if not level:
        return None
    return {
        "normal": 10.0,
        "low": 20.0,
        "moderate": 45.0,
        "high": 75.0,
        "critical": 92.0,
    }.get(str(level).lower())


def _model_agreement(scores: list[float]) -> tuple[str, float]:
    if len(scores) < 2:
        return "single_model", 0.0
    mean = sum(scores) / len(scores)
    variance = sum((score - mean) ** 2 for score in scores) / len(scores)
    std_dev = variance ** 0.5
    if std_dev < 10:
        return "high", round(std_dev, 2)
    if std_dev < 25:
        return "moderate", round(std_dev, 2)
    return "low", round(std_dev, 2)


def ensemble_predict(
    ml_features: dict[str, Any],
    lab_values: list[dict[str, Any]],
    clinical_risk_level: str | None = None,
) -> dict[str, Any]:
    """
    Weighted consensus over XGBoost, lab anomaly flags, and clinical rules.

    Weights:
      XGBoost  = 50%
      Anomaly  = 30%
      Clinical = 20%
    """
    scores: dict[str, float] = {}
    weights: dict[str, float] = {}
    shap_values: dict[str, float] = {}
    top_factors: list[Any] = []
    xgb_anomalies: list[dict[str, Any]] = []

    try:
        xgb_result = predict_safe(ml_features)
        scores["xgboost"] = float(xgb_result.get("risk_score", 0.0))
        weights["xgboost"] = 0.50
        shap_values = xgb_result.get("shap_values", {}) or {}
        top_factors = xgb_result.get("top_factors", []) or []
        xgb_anomalies = xgb_result.get("anomalies", []) or []
    except Exception as exc:
        log.warning("XGBoost failed in ensemble: %s", exc)

    try:
        anomaly_result = flag_anomalies(lab_values)
        scores["anomaly"] = _anomaly_score(anomaly_result)
        weights["anomaly"] = 0.30
    except Exception as exc:
        log.warning("Anomaly model failed in ensemble: %s", exc)
        anomaly_result = []

    clinical_score = _clinical_score(clinical_risk_level)
    if clinical_score is not None:
        scores["clinical"] = clinical_score
        weights["clinical"] = 0.20

    total_weight = sum(weights.values())
    consensus_raw = (
        sum(scores[name] * weights[name] for name in scores if name in weights) / total_weight
        if total_weight
        else 0.0
    )
    consensus_score = _platt_scale(consensus_raw)
    agreement, std_dev = _model_agreement([scores[name] for name in scores if weights.get(name, 0) > 0])

    anomalies = anomaly_result or xgb_anomalies
    return {
        "ensemble_risk_score": consensus_score,
        "risk_label": _risk_label(consensus_score),
        "model_scores": {
            "xgboost": round(scores.get("xgboost", 0.0), 1),
            "anomaly": round(scores.get("anomaly", 0.0), 1),
            "clinical": round(scores["clinical"], 1) if "clinical" in scores else None,
        },
        "model_weights": {name: weight for name, weight in weights.items() if weight > 0},
        "model_agreement": agreement,
        "agreement_std_dev": std_dev,
        "shap_values": shap_values,
        "top_factors": top_factors,
        "anomalies": anomalies,
        "uncertainty_flag": consensus_score < 40 or agreement == "low",
        "confidence": consensus_score,
        "models_used": [name for name, weight in weights.items() if weight > 0],
    }
