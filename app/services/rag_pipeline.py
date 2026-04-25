"""
Backend adapter for Subhiksha's Week 4 RAG pipeline.

This module is the stable boundary the backend should call. Subhiksha can paste
the notebook implementation into app/services/rag_notebook.py later without
requiring worker or route code to know about notebook internals.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable

logger = logging.getLogger(__name__)

_pipeline_fn: Callable[..., dict[str, Any]] | None = None
_initialized = False

MEDICAL_DISCLAIMER = (
    "AI-assisted analysis only. NOT a medical diagnosis. "
    "Consult a qualified healthcare professional before making any clinical decision."
)


def _bootstrap() -> None:
    """Import the notebook pipeline once, falling back to a safe stub."""
    global _pipeline_fn, _initialized

    if _initialized:
        return

    try:
        # Integration point:
        from app.services.rag_notebook import run_week4_pipeline_llama_groq

        _pipeline_fn = run_week4_pipeline_llama_groq
    except ImportError as exc:
        raise ImportError(
            f"Could not import RAG pipeline: {exc}. "
            "Install the notebook dependencies and verify app/services/rag_notebook.py."
        ) from exc

    _initialized = True


def _stub_pipeline(claim: str, top_k: int = 5, prompt_version: str = "v2") -> dict[str, Any]:
    """Safe fallback until the real notebook pipeline is committed."""
    return {
        "claim": claim,
        "retrieval_query": claim,
        "timestamp": "",
        "status": "stub",
        "retrieval": {"results": [], "count": 0},
        "draft_reasoning": None,
        "verification": None,
        "uncertainty": {
            "uncertain": True,
            "reasons": ["rag_not_initialized"],
            "source_count": 0,
            "final_confidence": 0.0,
        },
        "final_output": {
            "verdict": "insufficient_evidence",
            "summary": "RAG pipeline not yet initialized. Stub response.",
            "confidence_note": "stub",
            "citations": [],
        },
    }


def verify_claim(
    claim: str,
    top_k: int = 5,
    prompt_version: str = "v2",
) -> dict[str, Any]:
    """
    Run the Week 4 RAG pipeline on a single medical claim.

    Returns a compact backend schema plus the raw notebook output for debugging.
    """
    _bootstrap()

    try:
        assert _pipeline_fn is not None
        raw = _pipeline_fn(claim=claim, top_k=top_k, prompt_version=prompt_version)
    except Exception as exc:
        logger.exception("RAG pipeline error for claim: %s", claim)
        return {
            "claim": claim,
            "verdict": "insufficient_evidence",
            "summary": f"Pipeline error: {exc}",
            "confidence": 0.0,
            "citations": [],
            "uncertain": True,
            "disclaimer": MEDICAL_DISCLAIMER,
            "raw": {},
        }

    final = raw.get("final_output", {})
    uncertainty = raw.get("uncertainty", {})

    return {
        "claim": claim,
        "verdict": final.get("verdict", "insufficient_evidence"),
        "summary": final.get("summary", ""),
        "confidence": _parse_confidence(final.get("confidence_note", "")),
        "citations": final.get("citations", []),
        "uncertain": bool(uncertainty.get("uncertain", True)),
        "disclaimer": MEDICAL_DISCLAIMER,
        "raw": raw,
    }


def _parse_confidence(note: str) -> float:
    """Extract confidence from notebook confidence_note text."""
    for key in ("verification_confidence", "draft_confidence"):
        match = re.search(rf"{key}=([\d.]+)", note)
        if not match:
            continue
        try:
            return round(min(1.0, max(0.0, float(match.group(1)))), 4)
        except ValueError:
            continue
    return 0.0
