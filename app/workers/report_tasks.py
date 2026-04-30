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
from app.services.ocr import ocr_service
from app.services.clinicalbert import clinicalbert_service
from app.ml.predictor import predict_safe
from app.rag.retriever import retrieve_evidence
from app.workers.db_persist import persist_report, mark_failed

import logging
logger = logging.getLogger(__name__)


class ReportAnalysisTask(Task):
    abstract = True
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        logger.error(f"Report task {task_id} failed: {exc}")


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
        # Step 1: OCR
        self.update_state(state="PROGRESS", meta={"step": "ocr", "pct": 15})
        ocr_result = ocr_service.extract_text(file_path, file_format)
        raw_text   = ocr_result["raw_text"]
        if ocr_result.get("error"):
            logger.warning(f"OCR partial failure: {ocr_result['error']}")

        # Step 2: ClinicalBERT NER
        self.update_state(state="PROGRESS", meta={"step": "ner", "pct": 35})
        entities = clinicalbert_service.extract_entities(raw_text)

        # Step 3: XGBoost via predict_safe()
        self.update_state(state="PROGRESS", meta={"step": "xgboost", "pct": 55})
        feature_dict = entities.get("lab_values", {})
        ml_result    = predict_safe(feature_dict)
        risk_score   = ml_result["risk_score"]
        confidence   = ml_result["confidence"]
        shap_values  = ml_result["shap_values"]
        risk_factors = ml_result["top_factors"]
        anomalies    = ml_result["anomalies"]

        # Step 4: RAG retrieval
        self.update_state(state="PROGRESS", meta={"step": "rag", "pct": 75})
        rag_result   = retrieve_evidence(raw_text[:500])
        sources      = rag_result.get("results", [])
        
        source_count     = len(sources)
        explanation  = f"Risk score: {risk_score}. Top factors: {risk_factors}. Sources retrieved: {source_count}."
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
