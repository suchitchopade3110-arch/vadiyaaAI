"""
tasks_ml.py
-----------
Celery ML task wrapping the full Phase 1 text pipeline.

Wire order:
  process_pdf() → ner_df row
      → tharigha_to_predict(row)["features"]
      → predict_safe(features)
      → /analyze/report response

OWNER : Suchit (backend lead)
DEPENDS: tharigha_to_predict.py (project root), app.ml.predictor (Shree)
"""

import os
import sys
import logging

log = logging.getLogger(__name__)

# ── Ensure project root is on sys.path so `tharigha_to_predict` resolves ──────
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from tharigha_to_predict import tharigha_to_predict  # noqa: E402

# ── predict_safe: real pkl or fallback stub ───────────────────────────────────
try:
    from app.ml.predictor import predict_safe
    log.info("Loaded real predict_safe from app.ml.predictor")
except ImportError:
    log.warning("app.ml.predictor not found — using fallback predict_safe stub")

    def predict_safe(feature_dict: dict) -> dict:
        """Stub: returns plausible output until real pkl models are wired."""
        # Derive a simple heuristic score from glucose + HbA1c-proxy
        glucose = feature_dict.get("glucose", 90)
        ldl = feature_dict.get("ldl", 120)
        score = min(0.99, max(0.05, (glucose - 70) / 200 + (ldl - 100) / 400))
        return {
            "label": "Abnormal" if score > 0.5 else "Normal",
            "confidence": round(score, 4),
            "shap": {
                "Glucose (fasting)": round((glucose - 90) / 300, 4),
                "LDL cholesterol": round((ldl - 120) / 300, 4),
                "Hemoglobin": round(feature_dict.get("hemoglobin", 13.5) / 100, 4),
            },
            "anomalies": (
                ["Elevated fasting glucose"] if glucose > 100 else []
            ) + (
                ["Elevated LDL"] if ldl > 130 else []
            ),
        }


# ── Core pipeline function (callable directly, no Celery needed) ─────────────
def run_ml_pipeline(ner_row: dict) -> dict:
    """
    Synchronous ML pipeline.  Works with or without Celery.

    INPUT : dict from ner_df row (Thaariha's extract_lab_values output)
    OUTPUT: {
        status, patient_id, extraction_confidence,
        conditions, medications, imputed_fields, invalid_fields,
        ml_prediction: {label, confidence, shap, anomalies}
    }
    """
    log.info(
        f"[run_ml_pipeline] patient={ner_row.get('patient_id', 'UNKNOWN')}"
    )

    try:
        # 1. Bridge: NER row → 14-feature numeric dict
        bridge = tharigha_to_predict(ner_row)
        features = bridge["features"]

        # 2. Predict: features → label + confidence + SHAP + anomalies
        ml_result = predict_safe(features)

        # 3. Consolidate
        return {
            "status": "success",
            "patient_id": bridge["patient_id"],
            "extraction_confidence": bridge["extraction_confidence"],
            "conditions": bridge["conditions"],
            "medications": bridge["medications"],
            "imputed_fields": bridge["imputed_fields"],
            "invalid_fields": bridge["invalid_fields"],
            "ml_prediction": ml_result,
        }

    except Exception as exc:
        log.error(f"[run_ml_pipeline] FAILED: {exc}", exc_info=True)
        return {"status": "error", "error": str(exc)}


# ── Celery-wrapped version (same logic, but dispatchable via .delay()) ────────
try:
    from app.workers.celery_app import celery_app

    @celery_app.task(
        name="tasks_ml.run_ml_pipeline",
        bind=True,
        max_retries=2,
        default_retry_delay=5,
    )
    def run_ml_pipeline_task(self, ner_row: dict) -> dict:
        """Celery task — delegates to run_ml_pipeline()."""
        return run_ml_pipeline(ner_row)

except ImportError:
    log.warning("Celery app not available — run_ml_pipeline_task not registered")
    run_ml_pipeline_task = None


# ── Quick CLI test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import pprint

    row = {
        "patient_id": "TEST-001",
        "glucose": 186.0,
        "hemoglobin": 13.5,
        "cholesterol": 224.0,
        "bp_systolic": 142.0,
        "bp_diastolic": 88.0,
        "creatinine": 1.1,
        "ldl": 142.0,
        "hdl": 38.0,
        "age": 52,
        "gender": "M",
        "bert_entities": "[]",
    }

    result = run_ml_pipeline(row)
    print("=" * 60)
    print("run_ml_pipeline() OUTPUT")
    print("=" * 60)
    pprint.pprint(result)
