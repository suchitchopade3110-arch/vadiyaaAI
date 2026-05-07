"""
app/rag/reasoner.py
-------------------
LLM reasoning layer — extracted from Subhiksha's Week 2-4 Colab notebook.
Owner: LLM/RAG Engineer (Subhiksha)
Wired by: Backend Lead (Suchit) via tasks.py → /verify/claim route

Responsibilities:
- format evidence context for LLM prompt
- run Groq Llama-3 reasoning (Week 2)
- run self-verification loop (Week 3)
- compute uncertainty signal
- link citations back to evidence
- expose run_rag_pipeline(claim, top_k) → final structured output

Env vars required:
  GROQ_API_KEY   — from Colab Secrets / .env file
  GROQ_MODEL     — default: llama-3.1-8b-instant
"""

import json
import logging
import os
import re
import time
from datetime import timezone, datetime
UTC = timezone.utc
from functools import lru_cache

from groq import Groq

from .retriever import retrieve_evidence

log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
DISCLAIMER  = "AI-assisted analysis. NOT a medical diagnosis."

DEFAULT_THRESHOLDS = {
    "min_results":       1,
    "min_score":         0.30,
    "min_avg_score":     0.35,
    "min_unique_sources": 1,
}

# ── Groq client (lazy) ────────────────────────────────────────────────────────
_groq_client = None

def _get_groq() -> Groq:
    global _groq_client
    if _groq_client is None:
        key = os.getenv("GROQ_API_KEY")
        if not key:
            raise EnvironmentError("GROQ_API_KEY not set in environment / .env")
        _groq_client = Groq(api_key=key)
        log.info("Groq client initialized")
    return _groq_client


# ── Prompts ───────────────────────────────────────────────────────────────────
_REASON_SYSTEM = """
You are a medical evidence reasoning assistant for a prototype RAG system.

Rules:
- Use only the retrieved evidence provided.
- Do not use outside knowledge.
- If the evidence is weak, insufficient, or unrelated, return insufficient_evidence.
- Every factual statement must be grounded in the retrieved evidence.
- Every citation must use a real retrieved evidence id.
- Return valid JSON only.
"""

_VERIFY_SYSTEM = """
You are a medical evidence verification assistant.

Rules:
- Use only the retrieved evidence.
- Do not use outside knowledge.
- Check whether each claim in the reasoning is grounded in the evidence.
- If support is weak or missing, mark as insufficient_evidence.
- Return valid JSON only.
"""

def _reason_prompt(claim: str, evidence_context: str) -> str:
    return f"""
Evaluate the following medical claim using only the retrieved evidence.

Return ONLY valid JSON:
{{
  "verdict": "supported | contradicted | insufficient_evidence | partially_supported",
  "reasoning_summary": "short explanation",
  "confidence": 0.0,
  "status": "ok | insufficient_evidence | error",
  "citations": [
    {{
      "id": "retrieved_evidence_id",
      "source": "source_name",
      "title": "source title",
      "url": "source url",
      "text": "short supporting snippet"
    }}
  ]
}}

Claim:
{claim}

Retrieved Evidence:
{evidence_context}
"""

def _verify_prompt(claim: str, evidence_context: str, draft: dict) -> str:
    return f"""
Verify whether the following reasoning output is fully supported by the retrieved evidence.

Return ONLY valid JSON:
{{
  "verified_verdict": "supported | contradicted | insufficient_evidence",
  "verification_summary": "short explanation",
  "supported_claims": [{{"claim": "...", "source_ids": ["id1"]}}],
  "unsupported_claims": ["claim text"],
  "citation_coverage_ok": true,
  "final_confidence": 0.0
}}

Original Claim:
{claim}

Draft Reasoning:
{json.dumps(draft, indent=2)}

Retrieved Evidence:
{evidence_context}
"""


# ── Helpers ───────────────────────────────────────────────────────────────────
def _format_evidence(retrieval: dict) -> str:
    chunks = []
    for i, item in enumerate(retrieval.get("results", []), start=1):
        chunks.append(
            f"[{i}]\nid: {item.get('id','')}\n"
            f"source: {item.get('source','')}\n"
            f"title: {item.get('title','')}\n"
            f"url: {item.get('url','')}\n"
            f"score: {item.get('score',0)}\n"
            f"text: {item.get('text','')}\n"
        )
    return "\n".join(chunks) if chunks else "No evidence retrieved."


def _extract_json(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return {"verdict": "insufficient_evidence", "citations": [],
            "reasoning_summary": "JSON parse failed."}


def _llm_call(system: str, user: str, retries: int = 2) -> str:
    client = _get_groq()
    for attempt in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model=GROQ_MODEL,
                temperature=0,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ]
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if attempt < retries:
                time.sleep(1.5 ** attempt)
            else:
                log.error(f"Groq call failed: {e}")
                raise


def _evidence_is_weak(retrieval: dict, thresholds: dict = DEFAULT_THRESHOLDS) -> bool:
    results = retrieval.get("results", [])
    if len(results) < thresholds["min_results"]:
        return True
    strong = [r for r in results if float(r.get("score", 0)) >= thresholds["min_score"]]
    if len(strong) < thresholds["min_results"]:
        return True
    avg = sum(float(r["score"]) for r in strong) / max(1, len(strong))
    if avg < thresholds["min_avg_score"]:
        return True
    if len(set(r.get("source","") for r in strong)) < thresholds["min_unique_sources"]:
        return True
    return False


def _compute_uncertainty(retrieval: dict, verification: dict) -> dict:
    results  = retrieval.get("results", [])
    sources  = len(set((r.get("source",""), r.get("url","")) for r in results))
    conf     = float(verification.get("final_confidence", 0.0))
    unsup    = verification.get("unsupported_claims", [])
    cov_ok   = bool(verification.get("citation_coverage_ok", False))

    reasons = []
    if sources < 2:       reasons.append("fewer_than_2_sources")
    if conf < 0.30:       reasons.append("low_verification_confidence")
    if unsup:             reasons.append("unsupported_claims_present")
    if not cov_ok:        reasons.append("citation_coverage_incomplete")

    return {"uncertain": len(reasons) > 0, "reasons": reasons,
            "source_count": sources, "final_confidence": conf}


def _link_citations(citations: list, retrieval: dict) -> list:
    lookup = {r["id"]: r for r in retrieval.get("results", [])}
    linked = []
    for c in citations:
        if not isinstance(c, dict):
            continue
        ev = lookup.get(c.get("id",""), {})
        linked.append({
            "id":     ev.get("id",    c.get("id","")),
            "source": ev.get("source", c.get("source","")),
            "title":  ev.get("title",  c.get("title","")),
            "url":    ev.get("url",    c.get("url","")),
            "text":   c.get("text",   ev.get("text",""))[:300],
            "score":  ev.get("score", 0.0),
        })
    return linked


# ── In-memory cache (per process) ─────────────────────────────────────────────
_retrieval_cache: dict = {}
_reasoning_cache: dict = {}


def _cached_retrieve(claim: str, top_k: int) -> dict:
    key = (claim.strip().lower(), top_k)
    if key not in _retrieval_cache:
        _retrieval_cache[key] = retrieve_evidence(claim, top_k=top_k)
    return _retrieval_cache[key]


# ── PUBLIC API ────────────────────────────────────────────────────────────────
def run_rag_pipeline(claim: str, top_k: int = 5) -> dict:
    """
    Full Week 4 RAG pipeline: retrieve → reason → verify → uncertainty → output.

    INPUT:  claim (str) — medical claim text
    OUTPUT: {
        "claim": str,
        "timestamp": str,
        "status": "ok | error | insufficient_evidence",
        "retrieval": dict,
        "draft_reasoning": dict,
        "verification": dict,
        "uncertainty": dict,
        "final_output": {
            "verdict": "supported | contradicted | insufficient_evidence | partially_supported",
            "summary": str,
            "confidence_note": str,
            "citations": list,
        },
        "disclaimer": str
    }

    WIRED TO: app/workers/tasks.py → run_claim_pipeline Celery task
              app/api/v1/routes/claims.py → POST /verify/claim
    """
    ts = datetime.now(UTC).isoformat()

    # ── 1. Retrieval ──────────────────────────────────────────────────────────
    retrieval = _cached_retrieve(claim, top_k)

    if "error" in retrieval:
        return _error_response(claim, ts, retrieval["error"])

    # ── 2. Weak evidence gate ─────────────────────────────────────────────────
    if _evidence_is_weak(retrieval):
        return {
            "claim": claim, "timestamp": ts, "status": "ok",
            "retrieval": retrieval,
            "draft_reasoning": None, "verification": None,
            "uncertainty": {"uncertain": True, "reasons": ["weak_retrieval"],
                            "source_count": len(retrieval.get("results",[])),
                            "final_confidence": 0.0},
            "final_output": {
                "verdict": "insufficient_evidence",
                "summary": "Retrieved evidence is too weak or too limited.",
                "confidence_note": "Weak retrieval triggered uncertainty gate.",
                "citations": retrieval.get("results", [])[:2],
            },
            "disclaimer": DISCLAIMER,
        }

    evidence_ctx = _format_evidence(retrieval)

    # ── 3. Reasoning (Week 2) ─────────────────────────────────────────────────
    try:
        raw_reason  = _llm_call(_REASON_SYSTEM, _reason_prompt(claim, evidence_ctx))
        draft       = _extract_json(raw_reason)
    except Exception as e:
        return _error_response(claim, ts, f"Reasoning LLM failed: {e}")

    # clean draft
    draft.setdefault("verdict", "insufficient_evidence")
    draft.setdefault("reasoning_summary", "No summary.")
    draft.setdefault("confidence", 0.0)
    draft.setdefault("status", "ok")
    draft.setdefault("citations", [])

    # ── 4. Self-verification (Week 3) ─────────────────────────────────────────
    try:
        raw_verify   = _llm_call(_VERIFY_SYSTEM, _verify_prompt(claim, evidence_ctx, draft))
        verification = _extract_json(raw_verify)
    except Exception as e:
        log.warning(f"Verification LLM failed: {e} — skipping verification")
        verification = {
            "verified_verdict": draft["verdict"],
            "verification_summary": "Verification skipped.",
            "supported_claims": [], "unsupported_claims": [],
            "citation_coverage_ok": False, "final_confidence": draft["confidence"],
        }

    # ── 5. Uncertainty signal ─────────────────────────────────────────────────
    uncertainty = _compute_uncertainty(retrieval, verification)

    # ── 6. Link citations ─────────────────────────────────────────────────────
    linked = _link_citations(draft.get("citations", []), retrieval)

    # ── 7. Final verdict ──────────────────────────────────────────────────────
    final_verdict  = draft["verdict"]
    final_summary  = draft["reasoning_summary"]
    draft_conf     = float(draft.get("confidence", 0.0))
    verify_conf    = float(verification.get("final_confidence", 0.0))

    verified_verdict = verification.get("verified_verdict", "insufficient_evidence")
    if verified_verdict == "contradicted":
        final_verdict = verified_verdict
        final_summary = verification.get("verification_summary", final_summary)

    # uncertainty gate — only block on pipeline errors, not weak verification
    if uncertainty["uncertain"] and "pipeline_error" in uncertainty.get("reasons", []):
        final_verdict = "insufficient_evidence"

    if not linked:
        verify_conf = min(verify_conf, 0.30)

    conf_note = (
        f"draft_confidence={draft_conf:.2f}, "
        f"verification_confidence={verify_conf:.2f}"
    )
    if not linked:             conf_note += "; no_grounded_citations"
    if uncertainty["uncertain"]:
        conf_note += f"; uncertainty={','.join(uncertainty['reasons'])}"

    return {
        "claim":            claim,
        "timestamp":        ts,
        "status":           "ok",
        "retrieval":        retrieval,
        "draft_reasoning":  draft,
        "verification":     verification,
        "uncertainty":      uncertainty,
        "final_output": {
            "verdict":         final_verdict,
            "summary":         final_summary,
            "confidence_note": conf_note,
            "citations":       linked,
        },
        "disclaimer": DISCLAIMER,
    }


def _error_response(claim: str, ts: str, detail: str) -> dict:
    return {
        "claim": claim, "timestamp": ts, "status": "error",
        "retrieval": {}, "draft_reasoning": None,
        "verification": None,
        "uncertainty": {"uncertain": True, "reasons": ["pipeline_error"]},
        "final_output": {
            "verdict": "insufficient_evidence",
            "summary": "Pipeline error occurred.",
            "confidence_note": detail,
            "citations": [],
        },
        "disclaimer": DISCLAIMER,
    }
