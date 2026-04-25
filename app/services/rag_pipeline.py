import logging
from typing import Any
from app.core.config import settings
from app.core.disclaimer import MEDICAL_DISCLAIMER

logger = logging.getLogger(__name__)

class RAGPipeline:
    def retrieve_evidence(self, query: str, top_k: int = 5) -> list[dict]:
        logger.info(f"[Phase 1] RAG stub — ChromaDB not connected. Query: {query[:60]}")
        return []
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
    def explain_report(self, entities: dict, risk_score: float, risk_factors: list, anomalies: list, sources: list[dict]) -> str:
        if not sources: return "Insufficient evidence."
        return "[Phase 1] LLM explanation not yet connected."
    def explain_image(self, image_type: str, classification: dict, segmentation: dict, sources: list[dict]) -> str:
        if not sources: return "Insufficient evidence."
        return "[Phase 1] Image LLM explanation not yet connected."

rag_pipeline = RAGPipeline()
