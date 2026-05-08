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

# FEATURE_ORDER will be determined dynamically from the model
FEATURE_ORDER = []

DISCLAIMER = "AI-assisted analysis. NOT a medical diagnosis."

# ── Lazy singletons ───────────────────────────────────────────────────────────
_model    = None
_scaler   = None
_explainer = None


def _base_model_for_shap(model):
    """TreeExplainer works on the base estimator inside CalibratedClassifierCV."""
    return (
        model.calibrated_classifiers_[0].estimator
        if hasattr(model, "calibrated_classifiers_")
        else model
    )


def load_shap_explainer(model):
    """Load SHAP explainer safely; regenerate and cache it if pickle is incompatible."""
    path = _PKL["explainer"]
    if os.path.exists(path):
        try:
            explainer = joblib.load(path)
            log.info("[SHAP] Loaded explainer from %s", path)
            return explainer
        except Exception as exc:
            log.info("[SHAP] Pickle incompatible (%s); regenerating", exc)

    explainer = shap.TreeExplainer(_base_model_for_shap(model))
    try:
        joblib.dump(explainer, path)
        log.info("[SHAP] Regenerated and saved fresh explainer to %s", path)
    except Exception as exc:
        log.warning("[SHAP] Regenerated explainer but could not save cache: %s", exc)
    return explainer


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
        _explainer = load_shap_explainer(_model)
        log.info(f"ML models loaded: {type(_model).__name__}, {type(_scaler).__name__}")
        
        # Issue 2 Fix: Extract features exactly as model expects them
        global FEATURE_ORDER
        try:
            FEATURE_ORDER = list(_model.feature_names_in_)
        except AttributeError:
            try:
                FEATURE_ORDER = list(_model.calibrated_classifiers_[0].estimator.feature_names_in_)
            except AttributeError:
                FEATURE_ORDER = ["age", "gender", "glucose", "hemoglobin", "cholesterol", "bp_systolic", "bp_diastolic", "pulse_pressure", "tsh", "vitamin_d", "creatinine", "ldl", "hdl", "crp"]
                
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
    # PHASE 1 STUB — removed to enable XGBoost and real SHAP
    pass

    model, scaler, explainer = _load_models()

    # ── Normalize features and apply Fix 3 (SHAP Glucose Deviation) ───────────
    normalized_features = {}
    import re
    for k, v in feature_dict.items():
        val = v.get("value") if isinstance(v, dict) else v
        match = re.search(r"[\d.]+", str(val))
        if match:
            normalized_features[k.lower()] = float(match.group())

    num_cols = [f for f in FEATURE_ORDER if f != "gender" and f != "gender_encoded"]
    
    feature_vector = []
    for f in num_cols:
        val = normalized_features.get(f, 0.0)
        # Fix 3: Glucose feature encoding -> deviation from normal
        if f == "glucose":
            val = max(0.0, val - 110.0)
        feature_vector.append(val)
        
    X_raw = np.array(feature_vector).reshape(1, -1)

    # ── Scale features (scaler fitted without gender) ─────────────
    X_scaled_num = scaler.transform(X_raw)

    # ── Re-insert gender and other categorical features in exact FEATURE_ORDER ─────
    final_scaled_vector = []
    num_idx = 0
    for f in FEATURE_ORDER:
        if f == "gender" or f == "gender_encoded":
            final_scaled_vector.append(normalized_features.get("gender_encoded", 1.0))
        else:
            final_scaled_vector.append(X_scaled_num[0, num_idx])
            num_idx += 1
            
    X_scaled = np.array(final_scaled_vector).reshape(1, -1)

    # ── Predict ───────────────────────────────────────────────────────────────
    try:
        proba = model.predict_proba(X_scaled)[0]
    except AttributeError as e:
        if "__sklearn_tags__" not in str(e):
            raise
        import xgboost as xgb

        xgb_classifier = model.calibrated_classifiers_[0].estimator
        raw_booster = xgb_classifier.get_booster()
        
        # Booster was trained with explicit feature names
        xgb_features = FEATURE_ORDER
        dmatrix = xgb.DMatrix(X_scaled, feature_names=xgb_features)
        
        # Raw booster returns 1D array of P(y=1)
        proba_raw = raw_booster.predict(dmatrix)
        proba = [1 - proba_raw[0], proba_raw[0]]
    risk_proba = float(proba[1])          # P(high_risk)
    confidence = round(risk_proba * 100, 1)
    label      = "high_risk" if risk_proba >= 0.5 else "low_risk"

    # ── SHAP ──────────────────────────────────────────────────────────────────
    try:
        shap_vals = explainer.shap_values(X_scaled)

        # binary classification → use class-1 shap values
        if isinstance(shap_vals, list):
            sv = shap_vals[1][0]
        else:
            sv = shap_vals[0]

        actual_feature_count = X_scaled.shape[1]
        if len(FEATURE_ORDER) == actual_feature_count:
            feat_names = list(FEATURE_ORDER)
        else:
            feat_names = list(FEATURE_ORDER[:actual_feature_count])
            while len(feat_names) < actual_feature_count:
                feat_names.append(f"feature_{len(feat_names)}")
        shap_dict = {f: round(float(sv[i]), 4) for i, f in enumerate(feat_names)}

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

    # ── Anomaly detection & Reference Ranges (Fix 1, 2, 4) ────────────────────
    REF_RANGES = {
        "glucose":      (70,  110, "mg/dL"),
        "hemoglobin":   (12,  17.5, "g/dL"),
        "cholesterol":  (0,   200,  "mg/dL"),
        "bp_systolic":  (90,  120,  "mmHg"),
        "bp_diastolic": (60,  80,   "mmHg"),
        "tsh":          (0.4, 4.0,  "mIU/L"),
        "vitamin_d":    (20,  50,   "ng/mL"),
        "creatinine":   (0.6, 1.2,  "mg/dL"),
        "ldl":          (0,   100,  "mg/dL"),
        "hdl":          (40,  60,   "mg/dL"),
        "crp":          (0,   3.0,  "mg/L"),
    }

    anomalies = []
    
    # Mutate original feature_dict to expose reference/flags to frontend
    for k, v in feature_dict.items():
        k_lower = k.lower()
        if k_lower in REF_RANGES and k_lower in normalized_features:
            lo, hi, unit = REF_RANGES[k_lower]
            num_val = normalized_features[k_lower]
            
            # Decide flag
            if num_val > hi:
                flag = "HIGH"
            elif num_val < lo:
                flag = "LOW"
            else:
                flag = "NORMAL"

            # Mutate to feed frontend Fix 2
            if isinstance(v, dict):
                v["reference"] = f"{lo}\u2013{hi}"
                v["unit"] = unit
                v["flag"] = flag
                v["name"] = k.title()
                v["value"] = num_val

            if flag != "NORMAL":
                severity = "mild"
                if num_val > hi * 1.5 or num_val < lo * 0.5:
                    severity = "severe"
                elif num_val > hi * 1.2 or num_val < lo * 0.8:
                    severity = "moderate"
                
                anomalies.append({
                    "field": k.title(),
                    "value": num_val,
                    "unit": unit,
                    "reference": f"{lo}\u2013{hi}",
                    "severity": severity,
                    "explanation": f"{k.title()} is {flag} (normal range: {lo}\u2013{hi}). Possible clinical risk."
                })

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
