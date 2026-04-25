"""
VaidyaAI Celery Tasks
Each task owns its DB session (sync via psycopg2 — Celery is sync).
Pattern: PENDING → PROCESSING → result state → write to DB.
"""

import logging
from uuid import UUID
from celery import Task
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker, Session

from app.db.base import Base  # noqa: F401 - registers ALL models for SQLAlchemy mapper
from app.workers.celery_app import celery_app
from app.core.config import settings

log = logging.getLogger(__name__)

# Sync engine for Celery (workers are sync; asyncpg won't work here)
_SYNC_DB_URL = settings.DATABASE_URL.replace("+asyncpg", "+psycopg2")
_engine = create_engine(_SYNC_DB_URL, pool_pre_ping=True, pool_size=5)
SyncSession = sessionmaker(bind=_engine, expire_on_commit=False)


def get_sync_db() -> Session:
    return SyncSession()


# ─────────────────────────────────────────────
# Base task with error hook
# ─────────────────────────────────────────────

class VaidyaTask(Task):
    abstract = True

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        log.error(f"Task {self.name}[{task_id}] failed: {exc}")


# ─────────────────────────────────────────────
# CLAIM PIPELINE
# Text → ClinicalBERT → BioGPT → ChromaDB → XGBoost → SHAP → LLM → hallucination check
# ─────────────────────────────────────────────

@celery_app.task(
    bind=True,
    base=VaidyaTask,
    name="tasks.run_claim_pipeline",
    max_retries=3,
    default_retry_delay=10,
)
def run_claim_pipeline(self, claim_id: str):
    from app.models.claim import Claim, ClaimStatus
    from app.services.rag_pipeline import verify_claim

    db = get_sync_db()
    try:
        claim = db.execute(
            select(Claim).where(Claim.id == UUID(claim_id))
        ).scalar_one_or_none()

        if not claim:
            log.warning(f"Claim {claim_id} not found")
            return {"error": "not_found"}

        # ── Mark processing ───────────────────────────────────────────────────
        claim.status = ClaimStatus.PROCESSING
        db.commit()
        log.info(f"[claim:{claim_id}] PROCESSING")

        # ── Run full RAG pipeline (Weeks 1-4) ────────────────────────────────
        # Replaces all 5 stubs:
        #   STEP 1: ClinicalBERT NER        → run_rag_pipeline extracts entities
        #   STEP 2: BioGPT → ChromaDB RAG   → retrieve_evidence() inside pipeline
        #   STEP 3: XGBoost predict          → confidence from reasoning output
        #   STEP 4: Llama reasoning + verify → run_rag_pipeline reasoning step
        #   STEP 5: Hallucination check      → uncertainty signal inside pipeline
        result = verify_claim(claim=claim.claim_text, top_k=5, prompt_version="v2")
        rag_result = result.get("raw", {})
        final = rag_result.get("final_output", {})
        retrieval = rag_result.get("retrieval", {})
        draft = rag_result.get("draft_reasoning") or {}
        uncert = rag_result.get("uncertainty", {})

        # ── Map verdict → ClaimStatus ─────────────────────────────────────────
        verdict_map = {
            "supported":             ClaimStatus.VERIFIED,
            "partially_supported":   ClaimStatus.VERIFIED,
            "contradicted":          ClaimStatus.REFUTED,
            "insufficient_evidence": ClaimStatus.UNCERTAIN,
        }
        verdict_str = final.get("verdict", "insufficient_evidence")
        verdict     = verdict_map.get(verdict_str, ClaimStatus.UNCERTAIN)

        # ── Confidence 0–100 (Platt-scaled from RAG confidence float) ─────────
        raw_conf = float(result.get("confidence", 0.0))
        confidence = round(raw_conf * 100, 1)

        # ── Uncertainty flag ──────────────────────────────────────────────────
        uncertainty_flag = (
            uncert.get("uncertain", True) or
            confidence < (settings.CONFIDENCE_THRESHOLD * 100)
        )

        # ── Persist all RAG artifacts ─────────────────────────────────────────
        claim.extracted_entities = {
            "conditions":  draft.get("conditions", []),
            "medications": draft.get("medications", []),
            "lab_values":  [],
            "rag_status":  rag_result.get("status", "unknown"),
        }

        claim.source_citations = result.get("citations", [])

        claim.shap_values = {
            "confidence_note":     final.get("confidence_note", ""),
            "verification_summary": (
                rag_result.get("verification") or {}
            ).get("verification_summary", ""),
            "uncertainty_reasons": uncert.get("reasons", []),
        }

        claim.status           = verdict
        claim.confidence       = confidence
        claim.uncertainty_flag = uncertainty_flag

        # Attach mandatory disclaimer (PRD §9)
        claim.disclaimer = result.get("disclaimer")

        db.commit()
        log.info(
            f"[claim:{claim_id}] DONE "
            f"verdict={verdict_str} confidence={confidence} "
            f"uncertain={uncertainty_flag} "
            f"citations={len(result.get('citations', []))}"
        )

        return {
            "claim_id":        claim_id,
            "verdict":         verdict_str,
            "confidence":      confidence,
            "uncertainty_flag": uncertainty_flag,
            "citation_count":  len(result.get("citations", [])),
            "disclaimer":      claim.disclaimer,
        }

    except Exception as exc:
        db.rollback()
        log.error(f"[claim:{claim_id}] FAILED: {exc}", exc_info=True)
        # mark claim failed if possible
        try:
            claim = db.execute(
                select(Claim).where(Claim.id == UUID(claim_id))
            ).scalar_one_or_none()
            if claim:
                claim.status = ClaimStatus.UNCERTAIN
                claim.uncertainty_flag = True
                db.commit()
        except Exception:
            pass
        raise self.retry(exc=exc)
    finally:
        db.close()


# ─────────────────────────────────────────────
# IMAGE PIPELINE
# Normalize+CLAHE → LiteMedSAM → ROI → CheXNet → GradCAM → RAG → LLM
# GPU-bound: 10–60s per image
# ─────────────────────────────────────────────

@celery_app.task(
    bind=True,
    base=VaidyaTask,
    name="tasks.run_image_pipeline",
    max_retries=2,
    default_retry_delay=30,
    soft_time_limit=120,
    time_limit=180,
)
def run_image_pipeline(self, analysis_id: str):
    from app.models.image_analysis import ImageAnalysis

    db = get_sync_db()
    try:
        obj = db.execute(
            select(ImageAnalysis).where(ImageAnalysis.id == UUID(analysis_id))
        ).scalar_one_or_none()
        if not obj:
            return {"error": "not_found"}

        log.info(f"[image:{analysis_id}] PROCESSING type={obj.image_type}")

        # ── STEP 1: Normalize + CLAHE ─────────────────
        # TODO: Data/Preprocessing engineer
        # normalized_path = preprocessing_client.normalize(obj.file_path, obj.image_type)

        # ── STEP 2: LiteMedSAM segmentation ──────────
        # TODO: Data/Preprocessing engineer
        obj.segmentation_mask_path = None
        obj.roi_metadata = {
            "_stub": "LiteMedSAM ROI — wire when Data/Preprocessing PoC done",
            "bounding_boxes": [],
            "confidence": None,
        }

        # ── STEP 3: CheXNet classify ──────────────────
        # TODO: ML/Prediction engineer
        obj.classification = {
            "_stub": "CheXNet classification — wire when ML PoC done",
            "findings": [],
            "probabilities": {},
        }

        # ── STEP 4: GradCAM ───────────────────────────
        # TODO: ML/Prediction engineer
        obj.gradcam_path = None

        # ── STEP 5: RAG citations + LLM explain ───────
        # TODO: LLM/RAG engineer
        obj.source_citations = [
            {"_stub": "ChromaDB RAG — wire when LLM/RAG PoC done", "sources": []}
        ]

        obj.confidence = 0.0
        obj.uncertainty_flag = True

        db.commit()
        log.info(f"[image:{analysis_id}] DONE")
        return {"analysis_id": analysis_id, "status": "complete"}

    except Exception as exc:
        db.rollback()
        log.error(f"[image:{analysis_id}] ERROR: {exc}")
        raise self.retry(exc=exc)
    finally:
        db.close()


# ─────────────────────────────────────────────
# REPORT PIPELINE
# OCR → ClinicalBERT NER → XGBoost → SHAP → RAG → LLM
# ─────────────────────────────────────────────

@celery_app.task(
    bind=True,
    base=VaidyaTask,
    name="tasks.run_report_pipeline",
    max_retries=3,
    default_retry_delay=10,
)
def run_report_pipeline(self, report_id: str):
    from app.models.report import Report

    db = get_sync_db()
    try:
        report = db.execute(
            select(Report).where(Report.id == UUID(report_id))
        ).scalar_one_or_none()
        if not report:
            return {"error": "not_found"}

        log.info(f"[report:{report_id}] PROCESSING type={report.report_type}")

        # ── STEP 1: OCR (Tesseract + PyMuPDF) ────────
        # TODO: Data/Preprocessing engineer
        report.raw_text = None

        # ── STEP 2: ClinicalBERT NER ─────────────────
        # TODO: Data/Preprocessing engineer
        report.parsed_data = {
            "_stub": "ClinicalBERT NER — wire in Phase 1 wiring sprint",
            "conditions": [],
            "medications": [],
            "lab_values": {},
        }

        # ── STEP 3: XGBoost + SHAP ───────────────────
        from app.workers.tasks_ml import run_ml_pipeline
        ner_row = report.parsed_data or {}
        ner_row["patient_id"] = str(report.patient_id or report.id)
        ml_output = run_ml_pipeline(ner_row)
        report.analysis_result = {
            "risk_score": ml_output.get("ml_prediction", {}).get("confidence"),
            "label": ml_output.get("ml_prediction", {}).get("label"),
            "shap_values": ml_output.get("ml_prediction", {}).get("shap", {}),
            "anomalies": ml_output.get("ml_prediction", {}).get("anomalies", []),
            "imputed_fields": ml_output.get("imputed_fields", []),
            "extraction_confidence": ml_output.get("extraction_confidence", 0),
            "conditions": ml_output.get("conditions", []),
            "medications": ml_output.get("medications", []),
        }
        report.confidence = ml_output.get("ml_prediction", {}).get("confidence", 0.0) * 100

        # ── STEP 4: RAG + LLM ────────────────────────
        # TODO: LLM/RAG engineer
        report.source_citations = [
            {"_stub": "ChromaDB RAG — wire when LLM/RAG PoC done", "sources": []}
        ]

        report.confidence = 0.0
        report.uncertainty_flag = True

        db.commit()
        log.info(f"[report:{report_id}] DONE")
        return {"report_id": report_id, "status": "complete"}

    except Exception as exc:
        db.rollback()
        log.error(f"[report:{report_id}] ERROR: {exc}")
        raise self.retry(exc=exc)
    finally:
        db.close()
