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

DISCHARGE_SYSTEM_PROMPT = """
You are a medical AI assistant explaining a hospital discharge summary 
to a patient in simple, plain English.

Rules:
- NO medical jargon. Explain like talking to a non-doctor.
- Structure: What happened → What was found → What to do next
- Keep it under 200 words
- Always end with the mandatory disclaimer
"""

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
    def explain_report(self, entities: dict, risk_score: float, risk_factors: list, anomalies: list, sources: list[dict], report_type: str = "lab") -> str:
        # Mocking LLM response for Phase 1
        if report_type == "discharge":
            # In a real scenario, this would call an LLM with DISCHARGE_SYSTEM_PROMPT
            # For now, we simulate the structure requested by the user
            return (
                "WHAT HAPPENED:\nYou were admitted because you had difficulty swallowing food "
                "for several months...\n\n"
                "WHAT WAS FOUND:\nDoctors found a tumor in the middle part of your food pipe "
                "(esophagus). This is called Squamous Cell Carcinoma...\n\n"
                "WHAT TO DO NEXT:\n• Follow up with Dr. Tapan Kumar Dass\n"
                "• Attend scheduled CT scan and endoscopy appointments\n"
                "• Continue prescribed medications\n\n"
                f"{MEDICAL_DISCLAIMER}"
            )
        
        if report_type == "lab":
            if not anomalies:
                return f"No significant anomalies detected in your lab results. Your values appear to be within normal ranges.\n\n{MEDICAL_DISCLAIMER}"
                
            anomaly_summary = "\n".join([
                f"• {a.get('field', 'Test')}: {a.get('value')} {a.get('unit')} (Normal: {a.get('reference')}) — {a.get('severity')}"
                for a in anomalies
            ])
            
            return (
                "WHAT WAS FOUND:\n"
                f"The analysis detected the following key findings in your report:\n{anomaly_summary}\n\n"
                "WHAT THIS MEANS:\n"
                "These results indicate specific areas that require your attention. Your doctor "
                "will evaluate these findings in the context of your overall health and medical history.\n\n"
                "WHAT TO DO NEXT:\n"
                "• Schedule a follow-up appointment with your primary care physician.\n"
                "• Do not make any changes to your medications or diet based on these results alone.\n"
                "• Bring a copy of this report to your next consultation.\n\n"
                f"{MEDICAL_DISCLAIMER}"
            )
            
        if not sources: return "Insufficient evidence."
        return "[Phase 1] LLM explanation not yet connected."
    @with_retry(max_retries=2, backoff_seconds=5.0, fallback=_image_explanation_fallback)
    def explain_image(self, image_type: str, classification: dict, segmentation: dict, sources: list[dict]) -> str:
        if not sources: return "Insufficient evidence."
        return "[Phase 1] Image LLM explanation not yet connected."

rag_pipeline = RAGPipeline()
