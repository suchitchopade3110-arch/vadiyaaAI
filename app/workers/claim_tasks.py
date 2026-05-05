"""
VaidyaAI — Claim Verification Celery Tasks
Pipeline: ClinicalBERT NER → BioGPT → ChromaDB → GPT-4/Llama → Hallucination Check
"""

import time
from datetime import UTC, datetime
from celery import Task
from app.workers.celery_app import celery_app
from app.core.disclaimer import MEDICAL_DISCLAIMER
from app.core.config import settings
from app.services.preprocessor import run_ner
from app.services.rag_pipeline import rag_pipeline
from app.workers.db_persist import persist_claim, mark_failed

import logging
logger = logging.getLogger(__name__)


class ClaimVerificationTask(Task):
    abstract = True
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        logger.error(f"Claim task {task_id} failed: {exc}")
    def on_retry(self, exc, task_id, args, kwargs, einfo):
        logger.warning(f"Claim task {task_id} retrying: {exc}")


@celery_app.task(
    bind=True,
    base=ClaimVerificationTask,
    name="app.workers.claim_tasks.verify_claim",
    soft_time_limit=settings.TEXT_TASK_TIMEOUT,
    max_retries=3,
)
def verify_claim(self, claim_id: str, claim_text: str, patient_id: str = None):
    """
    Full claim verification pipeline.
    Steps:
    1. ClinicalBERT NER
    2. BioGPT encode → ChromaDB search
    3. GPT-4/Llama reasoning
    4. Hallucination check (Phase 2: 3-layer)
    5. Platt-scaled confidence
    """
    start_time = time.time()

    try:
        # ── Step 1: ClinicalBERT NER ──────────────────────────────────────
        self.update_state(state="PROGRESS", meta={"step": "ner", "pct": 10})
        ner_result = run_ner(claim_text)
        feature_dict = {lv.name: {"value": lv.value, "unit": lv.unit} for lv in ner_result.lab_values}
        entities = {
            "conditions": ner_result.conditions,
            "medications": ner_result.medications,
            "lab_values": feature_dict,
            "procedures": [],
            "_source": "clinicalbert-ner"
        }

        # ── Step 2: BioGPT → ChromaDB ─────────────────────────────────────
        self.update_state(state="PROGRESS", meta={"step": "rag", "pct": 30})
        from app.services.intelligence.rag_retrieval import retrieve_evidence
        evidence = retrieve_evidence(claim_text)
        sources = evidence["results"]

        if not evidence["sufficient"]:
            result = {
                "claim_id":              claim_id,
                "status":                "complete",
                "extracted_entities":    entities,
                "verdict":               "Uncertain",
                "explanation":           "Insufficient evidence. Fewer than 2 sources retrieved.",
                "sources":               sources,
                "source_count":          evidence["count"],
                "confidence_score":      0.0,
                "uncertainty_flag":      True,
                "hallucination_detected": False,
                "hallucination_details": {},
                "shap_values":           {},
                "medical_disclaimer":    MEDICAL_DISCLAIMER,
                "processing_time_ms":    round((time.time() - start_time) * 1000, 2),
                "completed_at":          datetime.now(UTC).isoformat(),
            }
            persist_claim(result)
            return result

        # ── Step 3: LLM Reasoning ─────────────────────────────────────────
        self.update_state(state="PROGRESS", meta={"step": "llm", "pct": 60})
        llm_result  = rag_pipeline.verify_claim(claim_text, entities, sources)
        verdict     = llm_result["verdict"]
        explanation = llm_result["explanation"]

        # ── Step 4: Hallucination Check (Phase 2: real 3-layer) ───────────
        self.update_state(state="PROGRESS", meta={"step": "hallucination_check", "pct": 80})
        hallucination_detected = llm_result.get("hallucination_detected", False)
        hallucination_details  = llm_result.get("hallucination_details", {})

        # ── Step 5: Confidence ────────────────────────────────────────────
        source_count     = len(sources)
        raw_confidence   = _score(verdict, source_count)
        uncertainty_flag = (
            source_count < settings.MIN_SOURCES_REQUIRED
            or raw_confidence < settings.MIN_CONFIDENCE_THRESHOLD * 100
        )

        self.update_state(state="PROGRESS", meta={"step": "complete", "pct": 100})
        elapsed_ms = (time.time() - start_time) * 1000

        result = {
            "claim_id":              claim_id,
            "status":                "complete",
            "extracted_entities":    entities,
            "verdict":               verdict,
            "explanation":           explanation,
            "sources":               sources,
            "source_count":          source_count,
            "confidence_score":      raw_confidence,
            "uncertainty_flag":      uncertainty_flag,
            "hallucination_detected": hallucination_detected,
            "hallucination_details": hallucination_details,
            "shap_values":           {},
            "medical_disclaimer":    MEDICAL_DISCLAIMER,
            "processing_time_ms":    round(elapsed_ms, 2),
            "completed_at":          datetime.now(UTC).isoformat(),
        }

        # ── Persist to PostgreSQL (non-fatal if DB unavailable) ───────────
        persist_claim(result)

        return result

    except Exception as exc:
        logger.error(f"verify_claim failed for {claim_id}: {exc}")
        mark_failed("claims", claim_id, str(exc))
        raise self.retry(exc=exc, countdown=5)


def _score(verdict: str, source_count: int) -> float:
    """Stub confidence — Platt scaling in Phase 2."""
    from app.utils.confidence import platt_scale
    if source_count < 2:    return platt_scale(0.15)
    if verdict == "verified": return platt_scale(0.75)
    if verdict == "refuted":  return platt_scale(0.70)
    return platt_scale(0.20)
