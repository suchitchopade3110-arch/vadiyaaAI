"""RAG pipeline backed by ChromaDB retrieval and optional Groq reasoning."""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.core.disclaimer import MEDICAL_DISCLAIMER
from app.utils.retry import with_retry

logger = logging.getLogger(__name__)

MODEL = os.getenv("GROQ_RAG_MODEL", os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"))

_groq = None
_chroma_client = None
_collections: dict[str, Any] = {}


def _get_groq():
    global _groq
    if _groq is None:
        api_key = os.getenv("GROQ_API_KEY") or getattr(settings, "GROQ_API_KEY", "")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set")
        from groq import Groq

        _groq = Groq(api_key=api_key)
    return _groq


def _get_chroma():
    """Connect to Chroma HTTP when configured, otherwise use local persistent data."""
    global _chroma_client
    if _chroma_client is not None:
        return _chroma_client

    import chromadb
    from chromadb.config import Settings

    use_http = os.getenv("CHROMADB_USE_HTTP", "").lower() in {"1", "true", "yes"}
    host = os.getenv("CHROMADB_HOST") or getattr(settings, "CHROMADB_HOST", "")
    port = os.getenv("CHROMADB_PORT") or getattr(settings, "CHROMADB_PORT", "")
    if use_http and host and port:
        try:
            _chroma_client = chromadb.HttpClient(host=host, port=int(port))
            _chroma_client.heartbeat()
            logger.info("ChromaDB connected at %s:%s", host, port)
            return _chroma_client
        except Exception as exc:
            logger.warning("ChromaDB HttpClient failed: %s; using persistent client", exc)

    chroma_path = os.getenv("CHROMA_PATH") or getattr(settings, "CHROMA_PATH", "") or "data/chromadb"
    path = Path(os.path.expanduser(chroma_path))
    if not path.exists() and Path("data/chromadb").exists():
        path = Path("data/chromadb")
    path.mkdir(parents=True, exist_ok=True)
    _chroma_client = chromadb.PersistentClient(
        path=str(path),
        settings=Settings(anonymized_telemetry=False),
    )
    logger.info("ChromaDB persistent client loaded at %s", path)
    return _chroma_client


def _hash_embedding_function():
    """Deterministic lightweight embedding function matching local KB ingesters."""

    class HashEmbeddingFunction:
        def __init__(self, dimensions: int = 96):
            self.dimensions = dimensions

        def __call__(self, input):
            texts = input if isinstance(input, list) else [input]
            return [self._embed(str(text)) for text in texts]

        def _embed(self, text: str) -> list[float]:
            import hashlib

            vector = [0.0] * self.dimensions
            for token in re.split(r"[^a-z0-9]+", text.lower()):
                if len(token) <= 2:
                    continue
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                index = int.from_bytes(digest[:4], "big") % self.dimensions
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                vector[index] += sign
            norm = sum(value * value for value in vector) ** 0.5 or 1.0
            return [value / norm for value in vector]

    return HashEmbeddingFunction()


def _get_collection(name: str):
    if name in _collections:
        return _collections[name]
    try:
        _collections[name] = _get_chroma().get_collection(
            name=name,
            embedding_function=_hash_embedding_function(),
        )
        return _collections[name]
    except Exception as exc:
        logger.debug("Chroma collection %s unavailable: %s", name, exc)
        return None


def _empty_evidence_fallback(self, query: str, top_k: int = 5) -> list[dict]:
    return []


def _claim_fallback(self, claim_text: str, entities: dict, sources: list[dict]) -> dict:
    return {
        "verdict": "uncertain",
        "explanation": "Insufficient evidence. Reasoning service unavailable after retries.",
        "hallucination_detected": False,
        "hallucination_details": {"retry_exhausted": True},
        "confidence_score": 0.2,
        "disclaimer": MEDICAL_DISCLAIMER,
    }


def _report_explanation_fallback(
    self,
    entities: dict,
    risk_score: float,
    risk_factors: list,
    anomalies: list,
    sources: list[dict],
    report_type: str = "lab",
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


def _image_explanation_from_classification(
    image_type: str,
    classification: dict,
    segmentation: dict,
    sources: list[dict],
) -> str:
    label = str(classification.get("label") or classification.get("top_class") or "unknown").replace("_", " ")
    confidence = classification.get("confidence")
    if confidence is None:
        confidence = classification.get("top_confidence", 0.0)
    try:
        confidence_pct = float(confidence) * 100 if float(confidence) <= 1 else float(confidence)
    except Exception:
        confidence_pct = 0.0

    support_text = "retrieved evidence was available for context" if sources else "no additional evidence sources were available"
    if segmentation and segmentation.get("bbox"):
        support_text = f"segmentation localized a focused region of interest; {support_text}"

    return (
        f"WHAT WAS FOUND:\n"
        f"The image was classified as {label} with about {confidence_pct:.1f}% confidence.\n\n"
        f"WHAT THIS MEANS:\n"
        f"This suggests {label.lower()} is the leading model finding on this {image_type} study. "
        f"The result should be interpreted with the full clinical picture because this is an AI-assisted output, not a diagnosis.\n\n"
        f"WHAT TO DO NEXT:\n"
        f"- Review the image with a radiologist or treating clinician.\n"
        f"- Correlate with symptoms, exam findings, and prior imaging.\n"
        f"- Consider follow-up if symptoms persist or the finding is clinically concerning.\n\n"
        f"ADDITIONAL CONTEXT:\n"
        f"{support_text}.\n\n"
        f"{MEDICAL_DISCLAIMER}"
    )


def _parse_json(raw: str) -> dict[str, Any]:
    clean = raw.strip().replace("```json", "").replace("```", "").strip()
    if not clean.startswith("{"):
        match = re.search(r"\{.*\}", clean, flags=re.DOTALL)
        if match:
            clean = match.group(0)
    return json.loads(clean)


def _normalize_source(row: dict[str, Any], default_source: str = "chroma") -> dict[str, Any]:
    text = str(row.get("text") or row.get("document") or row.get("snippet") or "")
    title = row.get("title") or row.get("pattern") or row.get("label") or row.get("id") or "Evidence"
    return {
        "id": str(row.get("id") or f"{default_source}:{hash(text)}"),
        "text": text,
        "score": row.get("score"),
        "source": row.get("source") or row.get("source_file") or default_source,
        "title": title,
        "category": row.get("category", ""),
        "snippet": row.get("snippet") or text[:240],
        **{key: value for key, value in row.items() if key not in {"document"}},
    }


class RAGPipeline:
    @with_retry(max_retries=2, backoff_seconds=5.0, fallback=_empty_evidence_fallback)
    def retrieve_evidence(self, query: str, top_k: int = 5) -> list[dict]:
        """Retrieve evidence from the main Chroma KB and domain-specific KBs."""
        results: list[dict] = []

        # Main medical evidence KB from app/rag/retriever.py, when present.
        try:
            from app.rag.retriever import retrieve_evidence as retrieve_main_evidence

            payload = retrieve_main_evidence(query, top_k=top_k)
            for row in payload.get("results", []) if isinstance(payload, dict) else []:
                results.append(_normalize_source(row, "medical_evidence"))
        except Exception as exc:
            logger.warning("Main RAG retrieval unavailable: %s", exc)

        # Local domain KB helpers include deterministic keyword fallbacks.
        try:
            from app.services.ingest_radiology_kb import query_radiology_kb

            for row in query_radiology_kb(query, n_results=max(2, top_k // 2)):
                results.append(_normalize_source(row, "radiology_patterns"))
        except Exception as exc:
            logger.debug("Radiology KB retrieval unavailable: %s", exc)

        try:
            from app.services.ingest_urinalysis_kb import query_urinalysis_kb

            for row in query_urinalysis_kb(query, n_results=max(2, top_k // 2)):
                results.append(_normalize_source(row, "urinalysis_patterns"))
        except Exception as exc:
            logger.debug("Urinalysis KB retrieval unavailable: %s", exc)

        # Optional generic collections, useful when ChromaDB runs as a service.
        for collection_name in ("vaidyaai_rag", "radiology_patterns", "urinalysis_patterns", "chexnet_labels"):
            collection = _get_collection(collection_name)
            if collection is None:
                continue
            try:
                count = collection.count()
                if count <= 0:
                    continue
                raw = collection.query(
                    query_texts=[query],
                    n_results=min(top_k, count),
                    include=["documents", "metadatas", "distances"],
                )
                ids = raw.get("ids", [[]])[0]
                docs = raw.get("documents", [[]])[0]
                metas = raw.get("metadatas", [[]])[0]
                distances = (raw.get("distances") or [[]])[0]
                for index, item_id in enumerate(ids):
                    metadata = metas[index] if index < len(metas) and metas[index] else {}
                    text = docs[index] if index < len(docs) else ""
                    distance = distances[index] if index < len(distances) else None
                    results.append(
                        _normalize_source(
                            {
                                "id": item_id,
                                "text": text,
                                "score": round(1 - float(distance), 3) if distance is not None else None,
                                "source": metadata.get("source", collection_name),
                                "title": metadata.get("pattern") or metadata.get("label") or metadata.get("title") or item_id,
                                "category": metadata.get("category", ""),
                                "snippet": text[:240],
                            },
                            collection_name,
                        )
                    )
            except Exception as exc:
                logger.warning("ChromaDB query failed for %s: %s", collection_name, exc)

        seen: set[str] = set()
        unique: list[dict] = []
        for row in sorted(results, key=lambda item: item.get("score") or 0, reverse=True):
            dedupe_key = row.get("id") or row.get("text", "")[:120]
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            unique.append(row)
            if len(unique) >= top_k:
                break

        logger.info("RAG retrieved %s sources for: %s", len(unique), query[:60])
        return unique

    @with_retry(max_retries=2, backoff_seconds=5.0, fallback=_claim_fallback)
    def verify_claim(self, claim_text: str, entities: dict, sources: list[dict]) -> dict:
        """Verify a medical claim against retrieved evidence using Groq."""
        if not sources:
            return {
                "verdict": "uncertain",
                "explanation": "Insufficient evidence retrieved from the knowledge base. Unable to verify claim.",
                "hallucination_detected": False,
                "hallucination_details": {},
                "confidence_score": 0.15,
                "disclaimer": MEDICAL_DISCLAIMER,
            }

        evidence_text = "\n".join(
            f"[{index + 1}] {source.get('title', 'Evidence')}\n{source.get('text', '')[:700]}"
            for index, source in enumerate(sources[:5])
        )
        prompt = f"""You are a medical fact-checker. Verify the claim against the evidence.

CLAIM:
{claim_text}

EVIDENCE:
{evidence_text}

Return ONLY valid JSON:
{{
  "verdict": "verified" | "refuted" | "uncertain",
  "explanation": "2-3 sentence explanation citing the evidence",
  "confidence_score": 0.0,
  "hallucination_detected": false,
  "hallucination_details": {{}}
}}"""

        try:
            response = _get_groq().chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500,
            )
            result = _parse_json(response.choices[0].message.content)
            result.setdefault("hallucination_detected", False)
            result.setdefault("hallucination_details", {})
            result["disclaimer"] = MEDICAL_DISCLAIMER
            return result
        except Exception as exc:
            logger.warning("Groq claim verification failed: %s", exc)
            return {
                "verdict": "uncertain",
                "explanation": "Evidence was retrieved, but reasoning could not be completed.",
                "hallucination_detected": False,
                "hallucination_details": {"error": str(exc)},
                "confidence_score": 0.2,
                "disclaimer": MEDICAL_DISCLAIMER,
            }

    @with_retry(max_retries=2, backoff_seconds=5.0, fallback=_report_explanation_fallback)
    def explain_report(
        self,
        entities: dict,
        risk_score: float,
        risk_factors: list,
        anomalies: list,
        sources: list[dict],
        report_type: str = "lab",
    ) -> str:
        """Generate a plain-language report explanation using Groq when available."""
        conditions = [str(item) for item in entities.get("conditions", [])[:5]]
        top_anomalies = [
            f"{item.get('field', 'Test')}: {item.get('value', '')} ({item.get('severity', '')})"
            for item in (anomalies or [])[:5]
        ]
        evidence = "\n".join(source.get("text", "")[:220] for source in (sources or [])[:3])

        prompt = f"""You are a clinical assistant. Write a clear, patient-friendly explanation.

Report type: {report_type}
Risk score: {round(float(risk_score or 0), 1)}/100
Conditions: {', '.join(conditions) or 'None identified'}
Key anomalies: {'; '.join(top_anomalies) or 'None'}
Evidence: {evidence or 'No retrieved evidence available'}

Write 3-4 sentences in plain English.
End with: "Please consult your healthcare provider for clinical guidance."
Do not mention AI or this system."""

        try:
            response = _get_groq().chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=300,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            logger.warning("Groq report explanation failed: %s", exc)
            if report_type == "lab" and anomalies:
                anomaly_lines = []
                for item in anomalies[:5]:
                    anomaly_lines.append(
                        f"- {item.get('field', 'Test')}: {item.get('value')} {item.get('unit', '')} "
                        f"(Normal: {item.get('reference', '--')}) - {item.get('severity', 'abnormal')}"
                    )
                return (
                    "WHAT WAS FOUND:\n"
                    f"The analysis detected these key findings:\n{chr(10).join(anomaly_lines)}\n\n"
                    "WHAT THIS MEANS:\n"
                    "These results may need clinical review in the context of symptoms and medical history.\n\n"
                    "WHAT TO DO NEXT:\n"
                    f"Please consult your healthcare provider for clinical guidance.\n\n{MEDICAL_DISCLAIMER}"
                )
            return f"Please consult your healthcare provider for clinical guidance.\n\n{MEDICAL_DISCLAIMER}"

    @with_retry(max_retries=2, backoff_seconds=5.0, fallback=_image_explanation_fallback)
    def explain_image(
        self,
        image_type: str,
        classification: dict,
        segmentation: dict,
        sources: list[dict],
    ) -> str:
        """Generate an image explanation using Groq plus retrieved evidence."""
        label = str(classification.get("label") or classification.get("top_class") or "unknown").replace("_", " ")
        confidence = classification.get("confidence")
        if confidence is None:
            confidence = classification.get("top_confidence", 0)
        try:
            confidence_pct = float(confidence) * 100 if float(confidence) <= 1 else float(confidence)
        except Exception:
            confidence_pct = 0.0
        evidence = "\n".join(source.get("text", "")[:220] for source in (sources or [])[:3])

        prompt = f"""You are a radiology assistant. Explain this medical imaging finding.

Image type: {image_type}
Primary finding: {label} ({confidence_pct:.1f}% confidence)
Evidence: {evidence or 'No retrieved evidence available'}

Write 2-3 sentences in plain English.
End with: "Please consult a qualified radiologist for formal interpretation."
Do not mention AI or this system."""

        try:
            response = _get_groq().chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=220,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            logger.warning("Groq image explanation failed: %s", exc)
            return _image_explanation_from_classification(image_type, classification, segmentation, sources)


rag_pipeline = RAGPipeline()
