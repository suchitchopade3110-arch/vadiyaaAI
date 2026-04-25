"""
VaidyaAI — Report Analysis Celery Tasks
Pipeline: PDF/CSV → OCR → ClinicalBERT NER → XGBoost → SHAP → Anomaly → LLM
"""

import time
from datetime import datetime
from celery import Task
from app.workers.celery_app import celery_app
from app.core.disclaimer import MEDICAL_DISCLAIMER
from app.core.config import settings
from app.services.ocr import ocr_service
from app.services.clinicalbert import clinicalbert_service
from app.services.ml_predictor import ml_predictor
from app.services.rag_pipeline import rag_pipeline
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
    """
    Full report analysis pipeline.
    Steps:
    1. OCR (Tesseract + PyMuPDF / pandas)
    2. ClinicalBERT NER
    3. XGBoost risk prediction + Platt scaling
    4. SHAP feature importance
    5. Anomaly detection
    6. RAG retrieval + LLM explanation
    """
    start_time = time.time()

    try:
        # ── Step 1: OCR ───────────────────────────────────────────────────
        self.update_state(state="PROGRESS", meta={"step": "ocr", "pct": 15})
        ocr_result = ocr_service.extract_text(file_path, file_format)
        raw_text   = ocr_result["raw_text"]

        if ocr_result.get("error"):
            logger.warning(f"OCR partial failure: {ocr_result['error']}")

        # ── Step 2: ClinicalBERT NER ──────────────────────────────────────
        self.update_state(state="PROGRESS", meta={"step": "ner", "pct": 35})
        entities = clinicalbert_service.extract_entities(raw_text)

        # ── Step 3: XGBoost + Platt ───────────────────────────────────────
        self.update_state(state="PROGRESS", meta={"step": "xgboost", "pct": 55})
        ml_result    = ml_predictor.predict_risk(entities)
        risk_score   = ml_result["risk_score"]
        risk_factors = ml_result["risk_factors"]
        shap_values  = ml_result["shap_values"]

        # ── Step 4: Anomaly Detection ─────────────────────────────────────
        self.update_state(state="PROGRESS", meta={"step": "anomaly", "pct": 68})
        anomalies = ml_predictor.detect_anomalies(entities.get("lab_values", {}))

        # ── Step 5: RAG + LLM ─────────────────────────────────────────────
        self.update_state(state="PROGRESS", meta={"step": "llm", "pct": 85})
        sources     = rag_pipeline.retrieve_evidence(raw_text[:500])
        explanation = rag_pipeline.explain_report(
            entities, risk_score, risk_factors, anomalies, sources
        )

        # Confidence
        source_count     = len(sources)
        confidence_score = ml_predictor.platt_scale(risk_score)
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
            "confidence_score":   confidence_score,
            "uncertainty_flag":   uncertainty_flag,
            "medical_disclaimer": MEDICAL_DISCLAIMER,
            "processing_time_ms": round(elapsed_ms, 2),
            "completed_at":       datetime.utcnow().isoformat(),
        }

        # ── Persist to PostgreSQL (non-fatal if DB unavailable) ───────────
        persist_report(result)

        return result

    except Exception as exc:
        logger.error(f"analyze_report failed for {report_id}: {exc}")
        mark_failed("reports", report_id, str(exc))
        raise self.retry(exc=exc, countdown=5)
