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
from typing import List
from app.services.preprocessor import validate_file, run_ocr, run_ner, LAB_REGISTRY, LabValue
from app.ml.predictor import predict_safe
from app.services.rag_pipeline import rag_pipeline
from app.workers.db_persist import persist_report, mark_failed

import logging
logger = logging.getLogger(__name__)


class ReportAnalysisTask(Task):
    abstract = True
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        logger.error(f"Report task {task_id} failed: {exc}")

def detect_anomalies(lab_values: List[LabValue], gender: str = "male") -> List[dict]:
    """
    Compare extracted lab values against LAB_REGISTRY reference ranges,
    or use the pre-computed flags from the PDF table parser (Layer 1).
    Returns list of anomalies with severity.
    """
    anomalies = []

    for lv in lab_values:
        # If Layer 1 already flagged this as anomalous
        if lv.flag and lv.status != "NORMAL":
            anomalies.append({
                "field": lv.name,
                "value": lv.value,
                "unit": lv.unit,
                "reference": f"{lv.ref_low} - {lv.ref_high}",
                "direction": "LOW" if lv.value < (lv.ref_low or 0) else "HIGH",
                "severity": lv.status,
                "pct_deviation": lv.pct_deviation or 0.0,
                "flag": lv.flag
            })
            continue
            
        # Layer 2 Fallback: Check registry if no flag is set (or if normal, no need to flag)
        if lv.flag: 
            continue # It was processed by Layer 1 and was NORMAL
            
        config = LAB_REGISTRY.get(lv.name.lower())
        if not config:
            continue

        # Get gender-specific range if available
        range_key = f"normal_range_{gender}" 
        ref = config.get(range_key) or config.get("normal_range")
        if not ref:
            continue

        low, high = ref
        value = lv.value

        if value < low:
            pct_below = ((low - value) / low) * 100
            severity = "CRITICAL" if pct_below > 50 else \
                       "HIGH" if pct_below > 25 else "MODERATE"
            anomalies.append({
                "field": lv.name,
                "value": value,
                "unit": lv.unit,
                "reference": f"{low} - {high}",
                "direction": "LOW",
                "severity": severity,
                "pct_deviation": round(pct_below, 1),
                "flag": f"⬇️ {severity}"
            })
        elif value > high:
            pct_above = ((value - high) / high) * 100
            severity = "CRITICAL" if pct_above > 50 else \
                       "HIGH" if pct_above > 25 else "MODERATE"
            anomalies.append({
                "field": lv.name,
                "value": value,
                "unit": lv.unit,
                "reference": f"{low} - {high}",
                "direction": "HIGH",
                "severity": severity,
                "pct_deviation": round(pct_above, 1),
                "flag": f"⬆️ {severity}"
            })

    return anomalies

def adjust_risk_for_anomalies(base_score: float, anomalies: list) -> float:
    """Boost risk score if CRITICAL or HIGH anomalies exist."""
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
            # Keep the more detailed one
            if len(a.get("explanation", "")) > len(seen[key].get("explanation", "")):
                seen[key] = a
    return list(seen.values())


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
        
        if report_type == "discharge":
            ner_result.lab_values = []

        registry_anomalies = detect_anomalies(ner_result.lab_values)
        anomaly_lookup = {a["field"]: a for a in registry_anomalies}

        feature_dict = {}
        for lv in ner_result.lab_values:
            flag = "NORMAL"
            ref_str = "--"
            if lv.name in anomaly_lookup:
                flag = anomaly_lookup[lv.name]["flag"]
                ref_str = anomaly_lookup[lv.name]["reference"]
            else:
                if lv.ref_low is not None and lv.ref_high is not None:
                    ref_str = f"{lv.ref_low} - {lv.ref_high}"
                    flag = lv.flag or "NORMAL"
                else:
                    config = LAB_REGISTRY.get(lv.name.lower())
                    if config:
                        ref_range = config.get("normal_range_male") or config.get("normal_range")
                        if ref_range:
                            ref_str = f"{ref_range[0]} - {ref_range[1]}"
            feature_dict[lv.name] = {"value": lv.value, "unit": lv.unit, "ref": ref_str, "flag": flag}
        
        entities = {
            "conditions": ner_result.conditions,
            "medications": ner_result.medications,
            "lab_values": feature_dict,
            "procedures": [],
            "_source": "clinicalbert-ner"
        }

        # Step 3: XGBoost via predict_safe()
        self.update_state(state="PROGRESS", meta={"step": "xgboost", "pct": 55})
        feature_dict = entities.get("lab_values", {})
        ml_result    = predict_safe(feature_dict)
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
