from app.services.rag_pipeline import rag_pipeline


def reason(claim_text, entities, prediction, evidence) -> dict:
    sources = evidence.get("results", evidence) if isinstance(evidence, dict) else evidence
    entity_dict = entities.as_dict() if hasattr(entities, "as_dict") else entities
    return rag_pipeline.verify_claim(claim_text, entity_dict, sources)


def reason_image(classification, mask, evidence) -> dict:
    sources = evidence.get("results", evidence) if isinstance(evidence, dict) else evidence
    explanation = rag_pipeline.explain_image("image", classification, mask, sources)
    return {
        "verdict": "uncertain",
        "confidence": 0.0,
        "explanation": explanation,
        "sources_used": sources,
    }

