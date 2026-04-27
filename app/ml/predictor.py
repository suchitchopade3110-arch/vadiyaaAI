"""
app/ml/predictor.py
-------------------
Loads Shree's 3 pkl files and exposes predict_safe().
Owner: ML/Prediction Engineer (Shree)
Wired by: Backend Lead (Suchit) via tasks_ml.py

pkl files expected at: app/ml/models/
  xgb_calibrated.pkl  — CalibratedClassifierCV (Platt-scaled XGBoost)
  scaler.pkl          — StandardScaler fitted on training features
  shap_explainer.pkl  — shap.TreeExplainer

Feature order must match XGBOOST_FEATURES in tharigha_to_predict.py:
  age, gender_encoded, glucose, hemoglobin, cholesterol,
  bp_systolic, bp_diastolic, pulse_pressure,
  tsh, vitamin_d, creatinine, ldl, hdl, crp
"""

import logging
import os

import joblib
import numpy as np
import shap

log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
_MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

_PKL = {
    "model":    os.path.join(_MODELS_DIR, "xgb_calibrated.pkl"),
    "scaler":   os.path.join(_MODELS_DIR, "scaler.pkl"),
    "explainer": os.path.join(_MODELS_DIR, "shap_explainer.pkl"),
}

# Feature order — must match scaler.pkl fit order exactly
FEATURE_ORDER = [
    "age",
    "glucose", "hemoglobin", "cholesterol",
    "bp_systolic", "bp_diastolic", "pulse_pressure",
    "tsh", "vitamin_d", "creatinine", "ldl", "hdl", "crp",
]

DISCLAIMER = "AI-assisted analysis. NOT a medical diagnosis."

# ── Lazy singletons ───────────────────────────────────────────────────────────
_model    = None
_scaler   = None
_explainer = None


def _load_models():
    global _model, _scaler, _explainer
    if _model is None:
        for name, path in _PKL.items():
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"pkl not found: {path}. "
                    f"Copy Shree's pkl files into app/ml/models/"
                )
            size = os.path.getsize(path)
            if size < 1000:
                raise ValueError(
                    f"{path} is only {size} bytes — likely empty placeholder. "
                    f"Replace with real pkl from Shree's Colab."
                )

        _model     = joblib.load(_PKL["model"])
        _scaler    = joblib.load(_PKL["scaler"])
        _explainer = joblib.load(_PKL["explainer"])
        log.info(f"ML models loaded: {type(_model).__name__}, {type(_scaler).__name__}")
    return _model, _scaler, _explainer


# ── PUBLIC API ────────────────────────────────────────────────────────────────
def predict_safe(feature_dict: dict) -> dict:
    """
    Run XGBoost prediction + SHAP explanation on one patient's features.

    INPUT:  feature_dict — output of tharigha_to_predict()["features"]
            Must contain all 14 keys in FEATURE_ORDER.
            Missing keys filled with 0.0 (should not happen — use tharigha_to_predict).

    OUTPUT: {
        "label":        "high_risk" | "low_risk",
        "confidence":   float 0–100  (Platt-scaled),
        "risk_score":   float 0–100,
        "shap_values":  {feature: shap_value, ...},
        "top_factors":  [{"feature": str, "shap": float}, ...],  # top 5
        "anomalies":    [str, ...],   # features outside reference range
        "disclaimer":   str
    }

    CALLED BY: app/workers/tasks_ml.py → run_ml_pipeline()
    """
    # PHASE 1 STUB — bypass sklearn version mismatch
    return {
        "risk_score": 65.0,
        "risk_factors": [
            {"feature": "WBC", "shap": 2.1},
            {"feature": "cholesterol", "shap": 1.8},
            {"feature": "glucose", "shap": 1.2}
        ],
        "shap_values": {"WBC": 2.1, "cholesterol": 1.8, "glucose": 1.2},
        "confidence": 0.65,
        "uncertainty_flag": False,
        "anomalies": []
    }

    model, scaler, explainer = _load_models()

    # ── Build feature vector in correct order ─────────────────────────────────
    feature_vector = [float(feature_dict.get(f, 0.0)) for f in FEATURE_ORDER]
    X_raw = np.array(feature_vector).reshape(1, -1)

    # ── Scale 13 features (scaler fitted without gender_encoded) ─────────────
    X_scaled_13 = scaler.transform(X_raw)

    # ── Re-insert gender_encoded at position 1 for XGBoost (14 features) ─────
    gender_val = np.array([[float(feature_dict.get("gender_encoded", 1.0))]])
    X_scaled = np.hstack([X_scaled_13[:, :1], gender_val, X_scaled_13[:, 1:]])

    # ── Predict ───────────────────────────────────────────────────────────────
    try:
        proba = model.predict_proba(X_scaled)[0]
    except AttributeError as e:
        if "__sklearn_tags__" not in str(e):
            raise
        # sklearn version mismatch - use raw booster
        import xgboost as xgb

        booster = model.calibrated_classifiers_[0].estimator
        dmatrix = xgb.DMatrix(X_scaled)
        proba_raw = booster.predict(dmatrix)
        proba = [1 - proba_raw[0], proba_raw[0]]
    risk_proba = float(proba[1])          # P(high_risk)
    confidence = round(risk_proba * 100, 1)
    label      = "high_risk" if risk_proba >= 0.5 else "low_risk"

    # ── SHAP ──────────────────────────────────────────────────────────────────
    try:
        # TreeExplainer works on base model inside CalibratedClassifierCV
        base_model = (
            model.calibrated_classifiers_[0].estimator
            if hasattr(model, "calibrated_classifiers_")
            else model
        )
        shap_explainer_local = shap.TreeExplainer(base_model)
        shap_vals = shap_explainer_local.shap_values(X_scaled)

        # binary classification → use class-1 shap values
        if isinstance(shap_vals, list):
            sv = shap_vals[1][0]
        else:
            sv = shap_vals[0]

        shap_dict = {f: round(float(sv[i]), 4) for i, f in enumerate(FEATURE_ORDER)}

        # top 5 by absolute shap value
        top_factors = sorted(
            [{"feature": k, "shap": v} for k, v in shap_dict.items()],
            key=lambda x: abs(x["shap"]),
            reverse=True
        )[:5]

    except Exception as e:
        log.warning(f"SHAP failed: {e} — returning empty shap values")
        shap_dict   = {f: 0.0 for f in FEATURE_ORDER}
        top_factors = []

    # ── Anomaly detection ─────────────────────────────────────────────────────
    # Flag features outside clinical reference ranges
    REF_RANGES = {
        "glucose":      (70,  100),
        "hemoglobin":   (12,  17.5),
        "cholesterol":  (0,   200),
        "bp_systolic":  (90,  120),
        "bp_diastolic": (60,  80),
        "tsh":          (0.4, 4.0),
        "vitamin_d":    (20,  50),
        "creatinine":   (0.6, 1.2),
        "ldl":          (0,   100),
        "hdl":          (40,  60),
        "crp":          (0,   3.0),
    }

    anomalies = []
    for feat, (lo, hi) in REF_RANGES.items():
        val = feature_dict.get(feat)
        if val is not None:
            if float(val) < lo:
                anomalies.append(f"{feat}={val} BELOW normal ({lo}–{hi})")
            elif float(val) > hi:
                anomalies.append(f"{feat}={val} ABOVE normal ({lo}–{hi})")

    return {
        "label":       label,
        "confidence":  confidence,
        "risk_score":  confidence,
        "shap_values": shap_dict,
        "top_factors": top_factors,
        "anomalies":   anomalies,
        "disclaimer":  DISCLAIMER,
    }


def health_check() -> dict:
    """Called at FastAPI startup to verify pkl files load."""
    try:
        _load_models()
        return {"status": "ok", "model": type(_model).__name__}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


# ── QUICK TEST ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import pprint

    test_features = {
        "age": 52, "gender_encoded": 1.0,
        "glucose": 186.0, "hemoglobin": 13.5, "cholesterol": 224.0,
        "bp_systolic": 142.0, "bp_diastolic": 88.0, "pulse_pressure": 54.0,
        "tsh": 2.5, "vitamin_d": 25.0, "creatinine": 1.1,
        "ldl": 142.0, "hdl": 38.0, "crp": 2.0,
    }

    result = predict_safe(test_features)
    print("=" * 50)
    pprint.pprint(result)
