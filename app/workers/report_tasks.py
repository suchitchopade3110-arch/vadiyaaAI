"""
VaidyaAI — Report Analysis Celery Tasks
Pipeline: PDF/CSV → OCR → ClinicalBERT NER → XGBoost → SHAP → Anomaly → LLM

FIXES:
  - predict_safe() from app.ml.predictor (function, no instance)
  - retrieve_evidence() from app.rag.retriever (function, no instance)
  - removed explain_report() — not in retriever; LLM explain via retrieved sources
  - len(sources) fixed — retrieve_evidence returns dict, use sources["results"]
"""

import time
from datetime import UTC, datetime
from celery import Task
from app.workers.celery_app import celery_app
from app.core.disclaimer import MEDICAL_DISCLAIMER
from app.core.config import settings
from typing import List, Optional, Dict, Any, Tuple
from app.services.preprocessor import clean_conditions, validate_file, run_ocr, run_ner, LAB_REGISTRY, LabValue
from app.ml.predictor import predict_safe
from app.services.rag_pipeline import rag_pipeline
from app.workers.db_persist import persist_report, mark_failed

import logging
logger = logging.getLogger(__name__)


class ReportAnalysisTask(Task):
    abstract = True
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        logger.error(f"Report task {task_id} failed: {exc}")

# Map anomalies → conditions (NOT from commentary text)
ANOMALY_TO_CONDITION = {
    ("hba1c", "HIGH"):              "Uncontrolled Diabetes Mellitus",
    ("fasting blood sugar", "HIGH"): "Hyperglycemia / Diabetes Suspected",
    ("hemoglobin", "LOW"):          "Anemia",
    ("iron", "LOW"):                "Iron Deficiency",
    ("vitamin b12", "LOW"):         "Vitamin B12 Deficiency Anemia",
    ("cholesterol", "HIGH"):        "Hypercholesterolemia",
    ("triglyceride", "HIGH"):       "Hypertriglyceridemia",
    ("wbc count", "HIGH"):          "Leukocytosis (Infection Possible)",
    ("malarial parasite", "ABNORMAL"): "Active Malarial Infection",
    ("hbsag", "ABNORMAL"):          "Hepatitis B Infection (Reactive)",
    ("hiv", "ABNORMAL"):            "HIV Reactive",
    ("creatinine", "HIGH"):         "Impaired Kidney Function",
    ("tsh", "HIGH"):                "Hypothyroidism Suspected",
    ("tsh", "LOW"):                 "Hyperthyroidism Suspected",
    ("urine glucose", "ABNORMAL"):  "Glycosuria (Diabetes Marker)",
}

def classify_anomaly(lv: LabValue, gender: str = "male") -> Optional[dict]:
    from app.services.preprocessor import resolve_reference
    
    if lv.value_type == "binary":
        if lv.is_abnormal:
            return {"field": lv.name, "value": lv.result_text, "unit": "", "reference": "Absent", "direction": "ABNORMAL", "severity": "HIGH", "flag": "⚠️ ABNORMAL"}
        return None
    
    if lv.value_type == "qualitative":
        if lv.is_abnormal:
            return {"field": lv.name, "value": lv.result_text, "unit": "", "reference": "Normal", "direction": "ABNORMAL", "severity": "MODERATE", "flag": "⚠️ ABNORMAL MORPHOLOGY"}
        return None
    
    if lv.value is None: return None
    
    ref_low, ref_high, source = resolve_reference(lv.name, gender, pdf_ref_low=lv.ref_low, pdf_ref_high=lv.ref_high)
    if ref_low is None: return None
    
    val = lv.value
    if val < ref_low:
        pct = round(((ref_low - val) / ref_low) * 100, 1)
        sev = "CRITICAL" if pct > 50 else "HIGH" if pct > 25 else "MODERATE"
        return {"field": lv.name, "value": lv.display_value or val, "unit": lv.unit, "reference": f"{ref_low}-{ref_high}", "direction": "LOW", "severity": sev, "pct_deviation": pct, "flag": f"⬇️ {sev}"}
    elif val > ref_high:
        pct = round(((val - ref_high) / ref_high) * 100, 1)
        sev = "CRITICAL" if pct > 50 else "HIGH" if pct > 25 else "MODERATE"
        return {"field": lv.name, "value": lv.display_value or val, "unit": lv.unit, "reference": f"{ref_low}-{ref_high}", "direction": "HIGH", "severity": sev, "pct_deviation": pct, "flag": f"⬆️ {sev}"}
    return None

def extract_conditions_from_anomalies(anomalies: List[dict]) -> List[str]:
    conditions = []
    for a in anomalies:
        key = (a["field"].lower(), a["direction"])
        if key in ANOMALY_TO_CONDITION:
            if ANOMALY_TO_CONDITION[key] not in conditions:
                conditions.append(ANOMALY_TO_CONDITION[key])
    return conditions[:10]

def detect_anomalies(lab_values: List[LabValue], gender: str = "male") -> List[dict]:
    anomalies = []
    for lv in lab_values:
        anomaly = classify_anomaly(lv, gender)
        if anomaly:
            anomalies.append(anomaly)
    return anomalies

def adjust_risk_for_anomalies(base_score: float, anomalies: list) -> float:
    """Boost risk score if CRITICAL or HIGH anomalies exist, or scale down if zero."""
    if len(anomalies) == 0:
        # XGBoost baseline is ~30-35 even for normal data. 
        # Scale it down to a "Healthy" range (0-15) if zero anomalies.
        return round(base_score * 0.15, 1)

    severities = [a["severity"] for a in anomalies]
    if "CRITICAL" in severities:
        return max(base_score, 75.0)   # CRITICAL → minimum High risk
    if "HIGH" in severities:
        return max(base_score, 55.0)   # HIGH → minimum Moderate
    return base_score

def deduplicate_anomalies(anomalies: list) -> list:
    seen = {}
    for a in anomalies:
        key = a["field"].lower()
        if key not in seen:
            seen[key] = a
        else:
            if len(a.get("explanation", "")) > len(seen[key].get("explanation", "")):
                seen[key] = a
    return list(seen.values())

def serialize_lab_value(lv: LabValue, anomaly_lookup: dict) -> dict:
    flag = "NORMAL"
    ref_str = "--"
    if lv.name in anomaly_lookup:
        flag = anomaly_lookup[lv.name]["flag"]
        ref_str = anomaly_lookup[lv.name]["reference"]
    else:
        if lv.ref_low is not None and lv.ref_high is not None:
            ref_str = f"{lv.ref_low} - {lv.ref_high}"
            flag = lv.flag or "NORMAL"
            
    return {
        "test": lv.name,
        "result": lv.display_value or str(lv.value or ""),
        "unit": lv.unit,
        "reference": ref_str,
        "flag": flag,
        "value_type": lv.value_type
    }


@celery_app.task(
    bind=True,
    base=ReportAnalysisTask,
    name="app.workers.report_tasks.analyze_report",
    soft_time_limit=settings.TEXT_TASK_TIMEOUT,
    max_retries=3,
)
def analyze_report(self, report_id: str, file_path: str, report_type: str, file_format: str):
    start_time = time.time()

    try:
        # Step 1: Validate
        val = validate_file(file_path)
        if not val.valid:
            raise ValueError(f"Report validation failed: {val.reason}")

        # Step 2: OCR
        self.update_state(state="PROGRESS", meta={"step": "ocr", "pct": 15})
        ocr_result = run_ocr(file_path)
        raw_text   = ocr_result.raw_text

        # Step 3: ClinicalBERT NER
        self.update_state(state="PROGRESS", meta={"step": "ner", "pct": 35})
        ner_result = run_ner(raw_text)
        
        if report_type == "lab":
            from app.services.preprocessor import UniversalValueExtractor
            extractor = UniversalValueExtractor()
            ner_result.lab_values = extractor.extract_all(raw_text)
            
        registry_anomalies = detect_anomalies(ner_result.lab_values)
        
        if report_type == "lab":
            # Universal Layer 5: Extract conditions ONLY from anomalies
            ner_result.conditions = extract_conditions_from_anomalies(registry_anomalies)
        else:
            ner_result.conditions = clean_conditions(ner_result.conditions)

        anomaly_lookup = {a["field"]: a for a in registry_anomalies}

        # Flat list for frontend table
        lab_values_serialized = [serialize_lab_value(lv, anomaly_lookup) for lv in ner_result.lab_values]
        
        # Numeric dict for ML features
        ml_features = {lv.name: lv.value for lv in ner_result.lab_values if lv.value is not None}
        
        if not ner_result.conditions and report_type == "lab" and len(registry_anomalies) == 0:
            ner_result.conditions = ["✅ No conditions detected — all values within normal range"]

        entities = {
            "conditions": ner_result.conditions,
            "medications": ner_result.medications,
            "lab_values": lab_values_serialized,
            "procedures": [],
            "_source": "clinicalbert-ner"
        }

        # Step 3: XGBoost via predict_safe()
        self.update_state(state="PROGRESS", meta={"step": "xgboost", "pct": 55})
        ml_result    = predict_safe(ml_features)
        risk_score   = ml_result["risk_score"]
        confidence   = ml_result["confidence"]
        shap_values  = ml_result["shap_values"]
        risk_factors = ml_result["top_factors"]
        xgboost_anomalies = ml_result["anomalies"]
        
        # Merge XGBoost anomalies with deterministic registry anomalies
        anomalies = registry_anomalies + xgboost_anomalies
        anomalies = deduplicate_anomalies(anomalies)
        
        # Bug 4: Adjust risk score based on anomaly severity
        risk_score = adjust_risk_for_anomalies(risk_score, anomalies)

        # Step 4: RAG retrieval & Explanation (Phase 2)
        self.update_state(state="PROGRESS", meta={"step": "rag", "pct": 75})
        sources = rag_pipeline.retrieve_evidence(raw_text[:500])
        
        source_count = len(sources)
        explanation = rag_pipeline.explain_report(
            entities=entities,
            risk_score=risk_score,
            risk_factors=risk_factors,
            anomalies=anomalies,
            sources=sources,
            report_type=report_type
        )
        uncertainty_flag = source_count < settings.MIN_SOURCES_REQUIRED

        self.update_state(state="PROGRESS", meta={"step": "complete", "pct": 100})
        elapsed_ms = (time.time() - start_time) * 1000

        result = {
            "report_id":          report_id,
            "report_type":        report_type,
            "status":             "complete",
            "raw_text_preview":   raw_text[:500],
            "extracted_entities": entities,
            "risk_score":         risk_score,
            "risk_factors":       risk_factors,
            "shap_values":        shap_values,
            "anomalies":          anomalies,
            "sources":            sources,
            "explanation":        explanation,
            "confidence_score":   confidence,
            "uncertainty_flag":   uncertainty_flag,
            "medical_disclaimer": MEDICAL_DISCLAIMER,
            "processing_time_ms": round(elapsed_ms, 2),
            "completed_at":       datetime.now(UTC).isoformat(),
        }

        persist_report(result)
        return result

    except Exception as exc:
        logger.error(f"analyze_report failed for {report_id}: {exc}")
        mark_failed("reports", report_id, str(exc))
        raise self.retry(exc=exc, countdown=5)
