import logging
from typing import Any
from app.core.config import settings
from app.core.disclaimer import MEDICAL_DISCLAIMER
from app.utils.retry import with_retry

logger = logging.getLogger(__name__)


def _empty_evidence_fallback(self, query: str, top_k: int = 5) -> list[dict]:
    return []


def _claim_fallback(self, claim_text: str, entities: dict, sources: list[dict]) -> dict:
    return {
        "verdict": "uncertain",
        "explanation": "Insufficient evidence. Reasoning service unavailable after retries.",
        "hallucination_detected": False,
        "hallucination_details": {"retry_exhausted": True},
    }


def _report_explanation_fallback(
    self,
    entities: dict,
    risk_score: float,
    risk_factors: list,
    anomalies: list,
    sources: list[dict],
) -> str:
    return "Insufficient evidence. Explanation service unavailable after retries."


def _image_explanation_fallback(
    self,
    image_type: str,
    classification: dict,
    segmentation: dict,
    sources: list[dict],
) -> str:
    return "Insufficient evidence. Image explanation service unavailable after retries."

class RAGPipeline:
    @with_retry(max_retries=2, backoff_seconds=5.0, fallback=_empty_evidence_fallback)
    def retrieve_evidence(self, query: str, top_k: int = 5) -> list[dict]:
        logger.info(f"[Phase 1] RAG stub — ChromaDB not connected. Query: {query[:60]}")
        return []
    @with_retry(max_retries=2, backoff_seconds=5.0, fallback=_claim_fallback)
    def verify_claim(self, claim_text: str, entities: dict, sources: list[dict]) -> dict:
        min_req = getattr(settings, "MIN_SOURCES_REQUIRED", 2)
        source_count = len(sources)
        if source_count < min_req:
            return {
                "verdict": "uncertain",
                "explanation": "Insufficient sources for verification.",
                "hallucination_detected": False,
                "hallucination_details": {},
            }
        return {
            "verdict": "uncertain",
            "explanation": "[Phase 1] LLM reasoning not yet connected.",
            "hallucination_detected": False,
            "hallucination_details": {},
        }
    @with_retry(max_retries=2, backoff_seconds=5.0, fallback=_report_explanation_fallback)
    def explain_report(self, entities: dict, risk_score: float, risk_factors: list, anomalies: list, sources: list[dict]) -> str:
        if not sources: return "Insufficient evidence."
        return "[Phase 1] LLM explanation not yet connected."
    @with_retry(max_retries=2, backoff_seconds=5.0, fallback=_image_explanation_fallback)
    def explain_image(self, image_type: str, classification: dict, segmentation: dict, sources: list[dict]) -> str:
        if not sources: return "Insufficient evidence."
        return "[Phase 1] Image LLM explanation not yet connected."

rag_pipeline = RAGPipeline()
