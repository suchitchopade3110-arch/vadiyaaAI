"""
tharigha_to_predict.py
----------------------
Converts Thaariha's NER output (ner_df row / dict) → Shree's XGBoost feature dict.

OWNER: Thaariha (preprocess) hands this to Suchit (backend) for Celery wiring.
USED BY: tasks_ml.py → predictor.py → predict_safe()

Pipeline position:
  process_pdf() → ner_df row
      ↓
  tharigha_to_predict()      ← THIS FILE
      ↓
  predict_safe(feature_dict) → {label, confidence, shap, anomalies}
      ↓
  /analyze/report route → API response
"""

import json
import logging
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)

# ── Feature schema ────────────────────────────────────────────────────────────
# Must match FEATURES list in predictor.py + scaler.pkl fit order exactly.
# Source: Shree's Colab — feature_schema shared Week 1 Day 5.
XGBOOST_FEATURES = [
    "age", "gender_encoded",
    "glucose", "hemoglobin", "cholesterol",
    "bp_systolic", "bp_diastolic", "pulse_pressure",
    "tsh", "vitamin_d", "creatinine", "ldl", "hdl", "crp",
]

# Population medians — used when field is None/missing.
# DO NOT use 0.0 — zero is a valid clinical value for some tests.
# Source: merged Kaggle dataset medians from Shree's EDA.
POPULATION_MEDIANS: dict = {
    "age":          45.0,
    "gender_encoded": 1.0,   # 1=M, 0=F, 1 = dataset median
    "glucose":      90.0,
    "hemoglobin":   13.5,
    "cholesterol":  190.0,
    "bp_systolic":  120.0,
    "bp_diastolic": 80.0,
    "pulse_pressure": 40.0,  # derived: systolic - diastolic
    "tsh":          2.5,
    "vitamin_d":    25.0,
    "creatinine":   0.9,
    "ldl":          120.0,
    "hdl":          50.0,
    "crp":          2.0,
}

# Lab value reference ranges for validator.
# Format: (min_possible, max_possible) — physiologically impossible outside these.
HARD_LIMITS: dict = {
    "age":          (0,    130),
    "glucose":      (1,    2000),
    "hemoglobin":   (1,    25),
    "cholesterol":  (50,   800),
    "bp_systolic":  (50,   300),
    "bp_diastolic": (20,   200),
    "tsh":          (0.001, 100),
    "vitamin_d":    (1,    200),
    "creatinine":   (0.1,  30),
    "ldl":          (10,   600),
    "hdl":          (5,    200),
    "crp":          (0,    500),
}

DISCLAIMER = "AI-assisted analysis. NOT a medical diagnosis."


# ── Field name aliases ────────────────────────────────────────────────────────
# Maps Thaariha's ner_df column names → canonical XGBoost feature names.
_ALIASES: dict = {
    # glucose variants
    "glucose":               "glucose",
    "blood_sugar":           "glucose",
    "fasting_glucose":       "glucose",
    "blood glucose":         "glucose",
    # hemoglobin variants
    "hemoglobin":            "hemoglobin",
    "hb":                    "hemoglobin",
    "haemoglobin":           "hemoglobin",
    # cholesterol
    "cholesterol":           "cholesterol",
    "total_cholesterol":     "cholesterol",
    # BP
    "bp_systolic":           "bp_systolic",
    "systolic":              "bp_systolic",
    "systolic_bp":           "bp_systolic",
    "bp_diastolic":          "bp_diastolic",
    "diastolic":             "bp_diastolic",
    "diastolic_bp":          "bp_diastolic",
    # other labs
    "tsh":                   "tsh",
    "vitamin_d":             "vitamin_d",
    "vit_d":                 "vitamin_d",
    "creatinine":            "creatinine",
    "ldl":                   "ldl",
    "hdl":                   "hdl",
    "crp":                   "crp",
    "c_reactive_protein":    "crp",
    # demographics
    "age":                   "age",
    "gender":                "gender",  # handled separately → gender_encoded
    "sex":                   "gender",
}


# ── Validator ─────────────────────────────────────────────────────────────────
def _validate_value(field: str, value: float) -> tuple[float, Optional[str]]:
    """
    Returns (value, warning_msg).
    Clamps impossible values + returns warning string for anomaly log.
    """
    if field not in HARD_LIMITS:
        return value, None
    lo, hi = HARD_LIMITS[field]
    if value < lo or value > hi:
        warn = f"INVALID {field}={value} outside [{lo},{hi}] → replaced with median"
        log.warning(warn)
        return POPULATION_MEDIANS.get(field, value), warn
    return value, None


def _encode_gender(raw: Optional[str]) -> float:
    """M/Male/m/1 → 1.0 | F/Female/f/0 → 0.0 | None → median 1.0"""
    if raw is None:
        return POPULATION_MEDIANS["gender_encoded"]
    s = str(raw).strip().lower()
    if s in ("m", "male", "1", "1.0"):
        return 1.0
    if s in ("f", "female", "0", "0.0"):
        return 0.0
    log.warning(f"Unknown gender value '{raw}' → using median 1.0")
    return POPULATION_MEDIANS["gender_encoded"]


# ── bert_entities parser ──────────────────────────────────────────────────────
def _parse_bert_entities(entities_raw) -> dict:
    """
    Extract conditions/medications from bert_entities column.
    ner_df stores this as JSON string (from to_csv) or list of dicts.
    Returns: {"conditions": [...], "medications": [...]}
    """
    if entities_raw is None:
        return {"conditions": [], "medications": []}

    # CSV reload gives string — deserialize
    if isinstance(entities_raw, str):
        try:
            entities_raw = json.loads(entities_raw)
        except (json.JSONDecodeError, ValueError):
            return {"conditions": [], "medications": []}

    if not isinstance(entities_raw, list):
        return {"conditions": [], "medications": []}

    conditions, medications = [], []
    DISEASE_LABELS = {"DISEASE", "DISORDER", "CONDITION", "DIAGNOSIS", "DiseaseClass", "Condition"}
    MED_LABELS     = {"MEDICATION", "DRUG", "CHEMICAL", "TREATMENT", "MedicalProcedure"}

    for ent in entities_raw:
        if not isinstance(ent, dict):
            continue
        label = ent.get("label", "").upper()
        text  = ent.get("text", "").strip()
        if not text:
            continue
        if label in DISEASE_LABELS:
            conditions.append(text)
        elif label in MED_LABELS:
            medications.append(text)

    return {"conditions": list(set(conditions)), "medications": list(set(medications))}


# ── MAIN BRIDGE ───────────────────────────────────────────────────────────────
def tharigha_to_predict(ner_row: dict | pd.Series) -> dict:
    """
    Convert one Thaariha NER output row → Shree's XGBoost feature dict.

    INPUT:  ner_df.iloc[i]  or  dict with keys from extract_lab_values()
            Expected keys (any subset — missing → population median):
              patient_id, source, bert_entities,
              glucose, hemoglobin, cholesterol,
              bp_systolic, bp_diastolic, tsh, vitamin_d,
              creatinine, ldl, hdl, crp,
              age, gender (optional)

    OUTPUT: {
        "patient_id": str,
        "features": {<feature>: float, ...},  # 14 features, ordered
        "imputed_fields": [str, ...],          # fields filled by median
        "invalid_fields": [str, ...],          # fields clamped due to impossible value
        "conditions": [str, ...],              # from bert_entities
        "medications": [str, ...],             # from bert_entities
        "disclaimer": str
    }

    PASS to: predict_safe(result["features"])
    """
    if isinstance(ner_row, pd.Series):
        ner_row = ner_row.to_dict()

    # Normalize column names via alias map
    normalized: dict = {}
    for raw_key, val in ner_row.items():
        canonical = _ALIASES.get(raw_key.lower().replace(" ", "_"), None)
        if canonical and val is not None:
            try:
                if canonical == "gender":
                    normalized["gender"] = val  # handle below
                else:
                    normalized[canonical] = float(val)
            except (TypeError, ValueError):
                pass  # non-numeric → skip

    imputed_fields: list[str] = []
    invalid_fields: list[str] = []
    features: dict = {}

    # ── gender encoding ───────────────────────────────────────
    features["gender_encoded"] = _encode_gender(normalized.get("gender"))
    if "gender" not in normalized:
        imputed_fields.append("gender_encoded")

    # ── age ───────────────────────────────────────────────────
    age_raw = normalized.get("age")
    if age_raw is not None:
        age_val, warn = _validate_value("age", age_raw)
        features["age"] = age_val
        if warn:
            invalid_fields.append("age")
    else:
        features["age"] = POPULATION_MEDIANS["age"]
        imputed_fields.append("age")

    # ── numeric lab features ──────────────────────────────────
    LAB_FEATURES = [
        "glucose", "hemoglobin", "cholesterol",
        "bp_systolic", "bp_diastolic",
        "tsh", "vitamin_d", "creatinine", "ldl", "hdl", "crp",
    ]

    for feat in LAB_FEATURES:
        raw_val = normalized.get(feat)
        if raw_val is not None:
            validated, warn = _validate_value(feat, raw_val)
            features[feat] = validated
            if warn:
                invalid_fields.append(feat)
        else:
            features[feat] = POPULATION_MEDIANS.get(feat, 0.0)
            imputed_fields.append(feat)

    # ── derived feature: pulse_pressure ──────────────────────
    features["pulse_pressure"] = round(
        features["bp_systolic"] - features["bp_diastolic"], 2
    )
    # If both BP were imputed, pulse_pressure is also imputed
    if "bp_systolic" in imputed_fields and "bp_diastolic" in imputed_fields:
        imputed_fields.append("pulse_pressure")

    # ── enforce feature order (must match scaler.pkl fit order) ──
    ordered_features = {f: features[f] for f in XGBOOST_FEATURES}

    # ── bert_entities → conditions + medications ──────────────
    parsed_entities = _parse_bert_entities(ner_row.get("bert_entities"))

    # ── confidence of this extraction ────────────────────────
    # Ratio of non-imputed lab features → extraction quality signal
    non_imputed = len(LAB_FEATURES) - sum(1 for f in LAB_FEATURES if f in imputed_fields)
    extraction_confidence = round(non_imputed / len(LAB_FEATURES), 2)

    result = {
        "patient_id":            str(ner_row.get("patient_id", "UNKNOWN")),
        "features":              ordered_features,
        "imputed_fields":        imputed_fields,
        "invalid_fields":        invalid_fields,
        "extraction_confidence": extraction_confidence,
        "conditions":            parsed_entities["conditions"],
        "medications":           parsed_entities["medications"],
        "disclaimer":            DISCLAIMER,
    }

    log.info(
        f"[tharigha_to_predict] patient={result['patient_id']} "
        f"confidence={extraction_confidence} "
        f"imputed={imputed_fields} invalid={invalid_fields}"
    )
    return result


def batch_convert(ner_df: pd.DataFrame) -> list[dict]:
    """
    Convert entire ner_df → list of feature dicts.
    Use for bulk /analyze/report jobs.
    """
    return [tharigha_to_predict(row) for _, row in ner_df.iterrows()]


# ── QUICK TEST ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import pprint

    # Simulate ner_df row (what Thaariha's extract_lab_values produces)
    sample_ner_row = {
        "patient_id":  "VDY-2026-0042",
        "source":      "Bajaj",
        "glucose":     186.0,
        "hemoglobin":  13.5,
        "cholesterol": 224.0,
        "bp_systolic": 142.0,
        "bp_diastolic": 88.0,
        "tsh":         None,       # missing → median
        "vitamin_d":   None,       # missing → median
        "creatinine":  1.1,
        "ldl":         142.0,
        "hdl":         38.0,
        "crp":         None,       # missing → median
        "age":         52,
        "gender":      "M",
        "bert_entities": json.dumps([
            {"text": "Type 2 Diabetes Mellitus", "label": "DiseaseClass"},
            {"text": "Metformin",                "label": "MedicalProcedure"},
            {"text": "Hypertension",             "label": "Condition"},
            {"text": "Atorvastatin",             "label": "MedicalProcedure"},
        ]),
    }

    result = tharigha_to_predict(sample_ner_row)

    print("=" * 60)
    print("tharigha_to_predict() OUTPUT")
    print("=" * 60)
    pprint.pprint(result)

    print("\n--- Features (pass directly to predict_safe()) ---")
    pprint.pprint(result["features"])

    print(f"\n--- Extraction confidence: {result['extraction_confidence']} ---")
    print(f"--- Imputed fields ({len(result['imputed_fields'])}): {result['imputed_fields']} ---")
    print(f"--- Conditions: {result['conditions']} ---")
    print(f"--- Medications: {result['medications']} ---")

    # Edge case: impossible hemoglobin
    bad_row = {**sample_ner_row, "hemoglobin": 500.0, "patient_id": "BAD-001"}
    bad_result = tharigha_to_predict(bad_row)
    print(f"\n--- Bad Hb=500 clamped. invalid_fields: {bad_result['invalid_fields']} ---")
    print(f"--- Hb after clamp: {bad_result['features']['hemoglobin']} ---")
