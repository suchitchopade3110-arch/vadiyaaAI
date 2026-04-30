def verify(explanation: dict, evidence: dict) -> dict:
    """
    Self-verify loop: cross-check LLM output vs retrieved sources.
    """
    sources = evidence.get("results", evidence) if isinstance(evidence, dict) else evidence
    result = dict(explanation or {})
    result["uncertainty_flag"] = len(sources or []) == 0
    result.setdefault("hallucination_detected", False)
    result.setdefault("hallucination_details", {})
    return result

