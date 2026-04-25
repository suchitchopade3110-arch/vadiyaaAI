import logging
from typing import Any
from app.core.config import settings
from app.core.disclaimer import MEDICAL_DISCLAIMER, UNCERTAINTY_MESSAGE

logger = logging.getLogger(__name__)

# ── Phase 2: Uncomment ────────────────────────────────────────────────────────
# import chromadb
# from transformers import BioGptTokenizer, BioGptModel
# import torch
# import openai
#
# _chroma_client = chromadb.HttpClient(host=settings.CHROMADB_HOST, port=settings.CHROMADB_PORT)
# _collection    = _chroma_client.get_collection(settings.CHROMADB_COLLECTION)
# _bio_tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
# _bio_model     = BioGptModel.from_pretrained("microsoft/biogpt")


class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline.
    Used by: claim_tasks, report_tasks, image_tasks (explanation step).
    """

    # ── Public API ────────────────────────────────────────────────────────────

    def retrieve_evidence(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Encode query with BioGPT → search ChromaDB → return top_k sources.
        
        Returns list of SourceCitation dicts matching API contract schema.
        Phase 1: Returns empty list (ChromaDB not connected).
        Phase 2: Replace with real retrieval below.
        """
        # ── Phase 2: Real retrieval ───────────────────────────────────────
        # embedding = self._biogpt_encode(query)
        # results   = _collection.query(query_embeddings=[embedding], n_results=top_k)
        # return self._format_sources(results)

        logger.info(f"[Phase 1] RAG stub — ChromaDB not connected. Query: {query[:60]}")
        return []   # Triggers uncertainty flag in workers

    def verify_claim(
        self,
        claim_text: str,
        entities: dict,
        sources: list[dict],
    ) -> dict:
        """
        GPT-4/Llama reasoning: claim + entities + evidence → verdict + explanation.
        
        Returns: {verdict, explanation, hallucination_detected, hallucination_details}
        Phase 1: Returns uncertain (no LLM connected).
        Phase 2: Replace with real LLM call below.
        """
        # We handle MIN_SOURCES_REQUIRED if present in settings. 
        # But default to 2 if not in settings.
        min_req = getattr(settings, "MIN_SOURCES_REQUIRED", 2)
        source_count = len(sources)

        if source_count < min_req:
            return {
                "verdict": "uncertain",
                "explanation": getattr(settings, "UNCERTAINTY_MESSAGE", "Insufficient sources for verification."),
                "hallucination_detected": False,
                "hallucination_details": {},
            }

        # ── Phase 2: Real LLM reasoning ───────────────────────────────────
        # prompt   = self._build_claim_prompt(claim_text, entities, sources)
        # response = self._call_llm(prompt)
        # verified = self._hallucination_check(response, sources)
        # return {
        #     "verdict":                verified["verdict"],
        #     "explanation":            verified["explanation"],
        #     "hallucination_detected": verified["hallucination_detected"],
        #     "hallucination_details":  verified["details"],
        # }

        return {
            "verdict": "uncertain",
            "explanation": "[Phase 1] LLM reasoning not yet connected.",
            "hallucination_detected": False,
            "hallucination_details": {},
        }

    def explain_report(
        self,
        entities: dict,
        risk_score: float,
        risk_factors: list,
        anomalies: list,
        sources: list[dict],
    ) -> str:
        """
        GPT-4/Llama: generate cited explanation for report analysis.
        Phase 1: Stub.
        """
        if not sources:
            msg = getattr(settings, "UNCERTAINTY_MESSAGE", "Insufficient evidence.")
            return f"[Phase 1] LLM explanation pending. {msg}"

        # ── Phase 2 ───────────────────────────────────────────────────────
        # prompt = self._build_report_prompt(entities, risk_score, risk_factors, anomalies, sources)
        # return self._call_llm(prompt)

        return "[Phase 1] LLM explanation not yet connected."

    def explain_image(
        self,
        image_type: str,
        classification: dict,
        segmentation: dict,
        sources: list[dict],
    ) -> str:
        """
        GPT-4/Llama: generate cited explanation for image analysis.
        Phase 1: Stub.
        """
        if not sources:
            msg = getattr(settings, "UNCERTAINTY_MESSAGE", "Insufficient evidence.")
            return f"[Phase 1] Image LLM explanation pending. {msg}"

        # ── Phase 2 ───────────────────────────────────────────────────────
        # prompt = self._build_image_prompt(image_type, classification, segmentation, sources)
        # return self._call_llm(prompt)

        return "[Phase 1] Image LLM explanation not yet connected."

    # ── Internal helpers (Phase 2) ────────────────────────────────────────────

    def _biogpt_encode(self, text: str) -> list[float]:
        """
        BioGPT encoder — produces embedding for ChromaDB search.
        BioGPT NEVER generates text here. Encoder role ONLY.
        """
        # Phase 2:
        # inputs  = _bio_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        # with torch.no_grad():
        #     outputs = _bio_model(**inputs)
        # embedding = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
        # return embedding
        raise NotImplementedError("Phase 2")

    def _call_llm(self, prompt: str) -> str:
        """
        Call GPT-4 (or Ollama/Llama fallback).
        GPT-4/Llama handles reasoning + generation ONLY.
        """
        # Phase 2:
        # if settings.OPENAI_API_KEY:
        #     client   = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
        #     response = client.chat.completions.create(
        #         model=settings.OPENAI_MODEL,
        #         messages=[
        #             {"role": "system", "content": SYSTEM_PROMPT},
        #             {"role": "user",   "content": prompt},
        #         ],
        #         temperature=0.1,   # Low temp for medical accuracy
        #         max_tokens=800,
        #     )
        #     return response.choices[0].message.content
        # else:
        #     # Ollama/Llama fallback
        #     import httpx
        #     r = httpx.post(f"{settings.OLLAMA_BASE_URL}/api/generate",
        #                    json={"model": settings.OLLAMA_MODEL, "prompt": prompt, "stream": False})
        #     return r.json()["response"]
        raise NotImplementedError("Phase 2")

    def _hallucination_check(self, llm_output: dict, sources: list[dict]) -> dict:
        """
        3-layer hallucination mitigation:
        Layer 1: RAG grounding — every claim must map to a source
        Layer 2: Self-verify loop — LLM re-checks its own output vs sources
        Layer 3: Confidence threshold — flag if score < MIN_CONFIDENCE_THRESHOLD
        Phase 2 implementation.
        """
        # Phase 2:
        # layer1 = self._check_rag_grounding(llm_output["explanation"], sources)
        # layer2 = self._self_verify(llm_output["explanation"], sources)
        # layer3 = llm_output["confidence"] > settings.MIN_CONFIDENCE_THRESHOLD
        # hallucination_detected = not (layer1 and layer2 and layer3)
        # return {...}
        raise NotImplementedError("Phase 2")

    def _format_sources(self, chroma_results: dict) -> list[dict]:
        """Convert ChromaDB results → SourceCitation schema."""
        # Phase 2:
        # sources = []
        # for i, doc in enumerate(chroma_results["documents"][0]):
        #     sources.append({
        #         "source_id":       chroma_results["ids"][0][i],
        #         "title":           chroma_results["metadatas"][0][i].get("title", "Unknown"),
        #         "excerpt":         doc[:300],
        #         "relevance_score": 1 - chroma_results["distances"][0][i],
        #         "url":             chroma_results["metadatas"][0][i].get("url"),
        #         "publication":     chroma_results["metadatas"][0][i].get("publication"),
        #     })
        # return sources
        raise NotImplementedError("Phase 2")


# ── Singleton ─────────────────────────────────────────────────────────────────
rag_pipeline = RAGPipeline()
