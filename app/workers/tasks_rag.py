"""Celery task wrapper for the Week 4 RAG claim verifier."""

from __future__ import annotations

import logging
from datetime import timezone, datetime
UTC = timezone.utc

from app.services.rag_pipeline import MEDICAL_DISCLAIMER
from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(
    name="rag.verify_claim",
    bind=True,
    max_retries=2,
    default_retry_delay=3,
    soft_time_limit=60,
    time_limit=90,
)
def verify_claim_task(self, claim_text: str, claim_id: str) -> dict:
    """Run RAG verification for a claim and return a Celery result payload."""
    logger.info("RAG task start | claim_id=%s", claim_id)

    try:
        from app.services.rag_pipeline import verify_claim

        result = verify_claim(claim_text, top_k=5, prompt_version="v2")
    except Exception as exc:
        logger.warning("RAG task failed (attempt %d): %s", self.request.retries, exc)
        try:
            raise self.retry(exc=exc)
        except self.MaxRetriesExceededError:
            logger.error("RAG task max retries exceeded for claim_id=%s", claim_id)
            return {
                "claim_id": claim_id,
                "claim": claim_text,
                "verdict": "insufficient_evidence",
                "summary": f"Pipeline failed after retries: {exc}",
                "confidence": 0.0,
                "citations": [],
                "uncertain": True,
                "disclaimer": MEDICAL_DISCLAIMER,
                "completed_at": datetime.now(UTC).isoformat(),
                "error": str(exc),
            }

    result["claim_id"] = claim_id
    result["completed_at"] = datetime.now(UTC).isoformat()

    logger.info(
        "RAG task done | claim_id=%s verdict=%s confidence=%.2f",
        claim_id,
        result.get("verdict"),
        result.get("confidence", 0.0),
    )

    return result
