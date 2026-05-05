"""
Backend-importable version of Subhiksha's Week 1-4 RAG notebook.

Colab-only lines are intentionally absent:
- no google.colab drive/userdata imports
- no drive.mount(...)
- no !pip installs

Runtime configuration comes from .env / environment variables:
    GROQ_API_KEY
    CHROMA_PATH
    RAG_MODELS_PATH
    RAG_EXPORTS_PATH
    RAG_TOP_K
    RAG_EMBED_BATCH_SIZE
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import UTC, datetime
from typing import Any, Literal

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    def _load_dotenv_fallback(path: str = ".env") -> None:
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as env_file:
            for line in env_file:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

    _load_dotenv_fallback()

try:
    from pydantic import BaseModel, Field
except ImportError:
    BaseModel = object  # type: ignore[assignment]

    def Field(default: Any = None, **_: Any) -> Any:  # type: ignore[no-redef]
        return default

logger = logging.getLogger(__name__)

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "medical_evidence_week1_clean")
BIOGPT_MODEL_ID = os.getenv("BIOGPT_MODEL_ID", "microsoft/biogpt")
MODELS_PATH = os.getenv("RAG_MODELS_PATH", "/tmp/rag_models")
EXPORTS_PATH = os.getenv("RAG_EXPORTS_PATH", "/tmp/rag_exports")
TOP_K_DEFAULT = int(os.getenv("RAG_TOP_K", "5"))
EMBED_BATCH_SIZE = int(os.getenv("RAG_EMBED_BATCH_SIZE", "8"))
MAX_TOKEN_LENGTH = int(os.getenv("RAG_MAX_TOKEN_LENGTH", "512"))

DISCLAIMER = (
    "AI-assisted analysis only. NOT a medical diagnosis. "
    "Consult a qualified healthcare professional before making any clinical decision."
)

DEFAULT_THRESHOLDS = {
    "min_results": 1,
    "min_score": 0.30,
    "min_avg_score": 0.35,
    "min_unique_sources": 1,
}

SYSTEM_PROMPT = """
You are a medical evidence reasoning assistant for a prototype RAG system.

Rules:
- Use only the retrieved evidence provided.
- Do not use outside knowledge.
- If the evidence is weak, insufficient, or unrelated, return insufficient_evidence.
- Every factual statement must be grounded in the retrieved evidence.
- Every citation must use a real retrieved evidence id.
- Return valid JSON only.
"""

VERIFY_SYSTEM_PROMPT = """
You are a medical evidence verification assistant.

Rules:
- Use only the retrieved evidence.
- Do not use outside knowledge.
- Check whether each claim in the reasoning is grounded in the evidence.
- If support is weak or missing, mark as insufficient_evidence.
- Return valid JSON only.
"""


class CitationOutput(BaseModel):
    id: str = ""
    source: str = ""
    title: str = ""
    url: str = ""
    text: str = ""


class ReasoningOutput(BaseModel):
    verdict: Literal[
        "supported",
        "contradicted",
        "insufficient_evidence",
        "partially_supported",
    ] = "insufficient_evidence"
    reasoning_summary: str = "No summary."
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    status: str = "ok"
    citations: list[CitationOutput] = []


class VerificationOutput(BaseModel):
    verified_verdict: Literal[
        "supported",
        "contradicted",
        "insufficient_evidence",
    ] = "insufficient_evidence"
    verification_summary: str = ""
    supported_claims: list[dict[str, Any]] = []
    unsupported_claims: list[str] = []
    citation_coverage_ok: bool = False
    final_confidence: float = Field(default=0.0, ge=0.0, le=1.0)


_initialized = False
_tokenizer = None
_model = None
_collection = None
_device = None
_groq_client = None
_retrieval_cache: dict[tuple[str, int], dict[str, Any]] = {}
_reasoning_cache: dict[tuple[str, str], dict[str, Any]] = {}


def _env_required(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise EnvironmentError(f"{name} is not set. Add it to .env and restart.")
    return value


def _get_device() -> str:
    global _device
    if _device is None:
        import torch

        _device = "cuda" if torch.cuda.is_available() else "cpu"
    return _device


def _init() -> None:
    """Load BioGPT and ChromaDB once per process."""
    global _initialized, _tokenizer, _model, _collection

    if _initialized:
        return

    chroma_path = _env_required("CHROMA_PATH")
    if not os.path.exists(chroma_path):
        raise FileNotFoundError(
            f"ChromaDB not found at {chroma_path}. Set CHROMA_PATH to the local folder."
        )

    import chromadb
    from transformers import BioGptModel, BioGptTokenizer

    device = _get_device()
    logger.info("Loading BioGPT from %s on %s", BIOGPT_MODEL_ID, device)
    _tokenizer = BioGptTokenizer.from_pretrained(
        BIOGPT_MODEL_ID,
        cache_dir=MODELS_PATH,
    )
    _model = BioGptModel.from_pretrained(
        BIOGPT_MODEL_ID,
        cache_dir=MODELS_PATH,
    ).to(device)
    _model.eval()

    client = chromadb.PersistentClient(path=chroma_path)
    _collection = client.get_collection(name=CHROMA_COLLECTION)
    logger.info("ChromaDB loaded: %s chunks", _collection.count())

    _initialized = True


def _get_groq():
    global _groq_client
    if _groq_client is None:
        from groq import Groq

        os.environ["GROQ_API_KEY"] = os.environ["GROQ_API_KEY"]
        _groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
        logger.info("Groq client initialized")
    return _groq_client


def embed_texts(
    texts: list[str],
    batch_size: int = EMBED_BATCH_SIZE,
) -> list[list[float]]:
    """BioGPT mean-pool embeddings. Encoder only; never generates text."""
    _init()
    import torch

    assert _tokenizer is not None
    assert _model is not None

    device = _get_device()
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = _tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=MAX_TOKEN_LENGTH,
        ).to(device)

        with torch.no_grad():
            outputs = _model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).cpu().tolist()

        all_embeddings.extend(embeddings)

    return all_embeddings


def _safe_cosine(a: list[float], b: list[float]) -> float:
    import numpy as np

    arr_a = np.array(a, dtype=np.float32)
    arr_b = np.array(b, dtype=np.float32)
    denom = np.linalg.norm(arr_a) * np.linalg.norm(arr_b)
    return float(np.dot(arr_a, arr_b) / denom) if denom != 0 else 0.0


def _keyword_overlap(query: str, text: str) -> float:
    q = set(re.findall(r"[a-z0-9]+", query.lower()))
    t = set(re.findall(r"[a-z0-9]+", text.lower()))
    return len(q & t) / len(q) if q else 0.0


def _source_diversity(
    scored: list[dict[str, Any]],
    top_k: int,
    max_per_source: int = 2,
) -> list[dict[str, Any]]:
    kept = []
    counts: dict[str, int] = {}
    for row in scored:
        src = row.get("source", "unknown")
        counts[src] = counts.get(src, 0)
        if counts[src] >= max_per_source:
            continue
        kept.append(row)
        counts[src] += 1
        if len(kept) >= top_k:
            break
    return kept


def rewrite_claim_for_retrieval(claim: str) -> str:
    """Normalize the claim into a retrieval-friendly query."""
    normalized = re.sub(r"\s+", " ", claim).strip()
    normalized = re.sub(r"\bmay indicate\b", "is associated with", normalized, flags=re.I)
    normalized = re.sub(r"\bmay suggest\b", "is associated with", normalized, flags=re.I)
    return normalized


def retrieve_evidence(query: str, top_k: int = TOP_K_DEFAULT) -> dict[str, Any]:
    """Retrieve top-k medical evidence chunks for a claim/query."""
    _init()
    assert _collection is not None

    total_docs = _collection.count()
    if total_docs == 0:
        return {"error": "ChromaDB collection is empty", "query": query, "results": [], "count": 0}

    retrieval_query = rewrite_claim_for_retrieval(query)
    query_embedding = embed_texts([retrieval_query], batch_size=1)[0]
    candidate_k = min(max(top_k * 4, 12), total_docs)

    raw = _collection.query(
        query_embeddings=[query_embedding],
        n_results=candidate_k,
        include=["documents", "metadatas", "distances"],
    )

    ids = raw["ids"][0]
    docs = raw["documents"][0]
    metas = raw["metadatas"][0]
    distances = raw["distances"][0]

    if not docs:
        return {
            "query": retrieval_query,
            "results": [],
            "count": 0,
            "timestamp": datetime.now(UTC).isoformat(),
        }

    doc_embeddings = embed_texts(docs, batch_size=min(8, len(docs)))
    scored = []

    for doc_id, text, meta, dist, doc_emb in zip(ids, docs, metas, distances, doc_embeddings):
        retrieval_score = round(1 - float(dist), 4)
        rerank_score = round(_safe_cosine(query_embedding, doc_emb), 4)
        keyword_score = round(_keyword_overlap(retrieval_query, text), 4)
        combined_score = round(
            0.60 * rerank_score + 0.25 * retrieval_score + 0.15 * keyword_score,
            4,
        )

        scored.append(
            {
                "id": doc_id,
                "text": text,
                "source": meta.get("source_name", ""),
                "source_file": meta.get("source_file", ""),
                "title": meta.get("title", ""),
                "url": meta.get("url", ""),
                "section_heading": meta.get("section_heading", ""),
                "page_number": meta.get("page_number", ""),
                "chunk_index": meta.get("chunk_index", ""),
                "score": combined_score,
                "retrieval_score": retrieval_score,
                "rerank_score": rerank_score,
                "keyword_score": keyword_score,
            }
        )

    scored = [
        row
        for row in scored
        if row["rerank_score"] >= 0.45 or row["keyword_score"] >= 0.20
    ]
    scored = sorted(scored, key=lambda row: row["score"], reverse=True)
    scored = _source_diversity(scored, top_k=top_k)

    return {
        "query": retrieval_query,
        "results": scored,
        "count": len(scored),
        "timestamp": datetime.now(UTC).isoformat(),
    }


def format_evidence_context(retrieval: dict[str, Any]) -> str:
    chunks = []
    for i, item in enumerate(retrieval.get("results", []), start=1):
        chunks.append(
            f"[{i}]\n"
            f"id: {item.get('id', '')}\n"
            f"source: {item.get('source', '')}\n"
            f"title: {item.get('title', '')}\n"
            f"url: {item.get('url', '')}\n"
            f"score: {item.get('score', 0)}\n"
            f"text: {item.get('text', '')}\n"
        )
    return "\n".join(chunks) if chunks else "No evidence retrieved."


def evidence_is_weak(
    retrieval: dict[str, Any],
    thresholds: dict[str, Any] = DEFAULT_THRESHOLDS,
) -> bool:
    results = retrieval.get("results", [])
    if len(results) < thresholds["min_results"]:
        return True
    strong = [r for r in results if float(r.get("score", 0)) >= thresholds["min_score"]]
    if len(strong) < thresholds["min_results"]:
        return True
    avg_score = sum(float(r["score"]) for r in strong) / max(1, len(strong))
    if avg_score < thresholds["min_avg_score"]:
        return True
    unique_sources = {r.get("source", "") for r in strong}
    return len(unique_sources) < thresholds["min_unique_sources"]


def _reason_prompt(claim: str, evidence_context: str, prompt_version: str) -> str:
    return f"""
Evaluate the following medical claim using only the retrieved evidence.

Prompt version: {prompt_version}

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


def _verify_prompt(claim: str, evidence_context: str, draft: dict[str, Any]) -> str:
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


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass

    return {
        "verdict": "insufficient_evidence",
        "citations": [],
        "reasoning_summary": "JSON parse failed.",
    }


def _llm_call(system: str, user: str, retries: int = 2) -> str:
    client = _get_groq()
    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                temperature=0,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            if attempt < retries:
                time.sleep(1.5**attempt)
                continue
            logger.error("Groq call failed: %s", exc)
            raise


def run_reasoning_step_llama_groq(
    claim: str,
    evidence_context: str,
    prompt_version: str = "v2",
) -> dict[str, Any]:
    cache_key = (claim.strip().lower(), evidence_context)
    if cache_key in _reasoning_cache:
        return _reasoning_cache[cache_key]

    raw = _llm_call(SYSTEM_PROMPT, _reason_prompt(claim, evidence_context, prompt_version))
    draft = _extract_json(raw)
    draft.setdefault("verdict", "insufficient_evidence")
    draft.setdefault("reasoning_summary", "No summary.")
    draft.setdefault("confidence", 0.0)
    draft.setdefault("status", "ok")
    draft.setdefault("citations", [])
    _reasoning_cache[cache_key] = draft
    return draft


def run_verification_step_llama_groq(
    claim: str,
    evidence_context: str,
    draft: dict[str, Any],
) -> dict[str, Any]:
    raw = _llm_call(VERIFY_SYSTEM_PROMPT, _verify_prompt(claim, evidence_context, draft))
    verification = _extract_json(raw)
    verification.setdefault("verified_verdict", draft.get("verdict", "insufficient_evidence"))
    verification.setdefault("verification_summary", "")
    verification.setdefault("supported_claims", [])
    verification.setdefault("unsupported_claims", [])
    verification.setdefault("citation_coverage_ok", False)
    verification.setdefault("final_confidence", draft.get("confidence", 0.0))
    return verification


def compute_uncertainty_signal(
    retrieval: dict[str, Any],
    verification: dict[str, Any] | None,
) -> dict[str, Any]:
    verification = verification or {}
    results = retrieval.get("results", [])
    source_count = len({(r.get("source", ""), r.get("url", "")) for r in results})
    final_confidence = float(verification.get("final_confidence", 0.0))
    unsupported = verification.get("unsupported_claims", [])
    citation_coverage_ok = bool(verification.get("citation_coverage_ok", False))

    reasons = []
    if source_count < 2:
        reasons.append("fewer_than_2_sources")
    if final_confidence < 0.30:
        reasons.append("low_verification_confidence")
    if unsupported:
        reasons.append("unsupported_claims_present")
    if not citation_coverage_ok:
        reasons.append("citation_coverage_incomplete")

    return {
        "uncertain": bool(reasons),
        "reasons": reasons,
        "source_count": source_count,
        "final_confidence": final_confidence,
    }


def build_evidence_lookup(retrieval: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["id"]: item for item in retrieval.get("results", []) if "id" in item}


def link_citations_to_evidence(
    citations: list[dict[str, Any]],
    retrieval: dict[str, Any],
) -> list[dict[str, Any]]:
    lookup = build_evidence_lookup(retrieval)
    linked = []

    for citation in citations:
        if not isinstance(citation, dict):
            continue
        evidence = lookup.get(citation.get("id", ""), {})
        linked.append(
            {
                "id": evidence.get("id", citation.get("id", "")),
                "source": evidence.get("source", citation.get("source", "")),
                "title": evidence.get("title", citation.get("title", "")),
                "url": evidence.get("url", citation.get("url", "")),
                "text": citation.get("text", evidence.get("text", ""))[:300],
                "score": evidence.get("score", 0.0),
            }
        )

    return linked


def _error_response(claim: str, retrieval_query: str, ts: str, detail: str) -> dict[str, Any]:
    return {
        "claim": claim,
        "retrieval_query": retrieval_query,
        "timestamp": ts,
        "status": "error",
        "retrieval": {},
        "draft_reasoning": None,
        "verification": None,
        "uncertainty": {
            "uncertain": True,
            "reasons": ["pipeline_error"],
            "source_count": 0,
            "final_confidence": 0.0,
        },
        "final_output": {
            "verdict": "insufficient_evidence",
            "summary": "Pipeline error occurred.",
            "confidence_note": detail,
            "citations": [],
        },
        "disclaimer": DISCLAIMER,
    }


def run_week4_pipeline_llama_groq(
    claim: str,
    top_k: int = 5,
    prompt_version: str = "v2",
) -> dict[str, Any]:
    """Full Week 4 pipeline: retrieve, reason, verify, uncertainty, citations."""
    ts = datetime.now(UTC).isoformat()
    retrieval_query = rewrite_claim_for_retrieval(claim)
    cache_key = (retrieval_query.strip().lower(), top_k)

    try:
        if cache_key not in _retrieval_cache:
            _retrieval_cache[cache_key] = retrieve_evidence(retrieval_query, top_k=top_k)
        retrieval = _retrieval_cache[cache_key]
    except Exception as exc:
        return _error_response(claim, retrieval_query, ts, f"Retrieval failed: {exc}")

    if retrieval.get("error"):
        return _error_response(claim, retrieval_query, ts, retrieval["error"])

    if evidence_is_weak(retrieval):
        return {
            "claim": claim,
            "retrieval_query": retrieval_query,
            "timestamp": ts,
            "status": "ok",
            "retrieval": retrieval,
            "draft_reasoning": None,
            "verification": None,
            "uncertainty": {
                "uncertain": True,
                "reasons": ["weak_retrieval"],
                "source_count": len(retrieval.get("results", [])),
                "final_confidence": 0.0,
            },
            "final_output": {
                "verdict": "insufficient_evidence",
                "summary": "Retrieved evidence is too weak or too limited.",
                "confidence_note": "Weak retrieval triggered uncertainty gate.",
                "citations": retrieval.get("results", [])[:2],
            },
            "disclaimer": DISCLAIMER,
        }

    evidence_context = format_evidence_context(retrieval)

    try:
        draft = run_reasoning_step_llama_groq(
            claim=claim,
            evidence_context=evidence_context,
            prompt_version=prompt_version,
        )
    except Exception as exc:
        return _error_response(claim, retrieval_query, ts, f"Reasoning LLM failed: {exc}")

    try:
        verification = run_verification_step_llama_groq(
            claim=claim,
            evidence_context=evidence_context,
            draft=draft,
        )
    except Exception as exc:
        logger.warning("Verification LLM failed: %s; using draft confidence", exc)
        verification = {
            "verified_verdict": draft.get("verdict", "insufficient_evidence"),
            "verification_summary": "Verification skipped.",
            "supported_claims": [],
            "unsupported_claims": [],
            "citation_coverage_ok": False,
            "final_confidence": draft.get("confidence", 0.0),
        }

    uncertainty = compute_uncertainty_signal(retrieval, verification)
    linked = link_citations_to_evidence(draft.get("citations", []), retrieval)

    final_verdict = draft.get("verdict", "insufficient_evidence")
    final_summary = draft.get("reasoning_summary", "")
    draft_confidence = float(draft.get("confidence", 0.0))
    verification_confidence = float(verification.get("final_confidence", 0.0))

    verified_verdict = verification.get("verified_verdict", "insufficient_evidence")
    if verified_verdict == "contradicted":
        final_verdict = verified_verdict
        final_summary = verification.get("verification_summary", final_summary)

    if not linked:
        verification_confidence = min(verification_confidence, 0.30)

    confidence_note = (
        f"draft_confidence={draft_confidence:.2f}, "
        f"verification_confidence={verification_confidence:.2f}"
    )
    if not linked:
        confidence_note += "; no_grounded_citations"
    if uncertainty["uncertain"]:
        confidence_note += f"; uncertainty={','.join(uncertainty['reasons'])}"

    return {
        "claim": claim,
        "retrieval_query": retrieval_query,
        "timestamp": ts,
        "status": "ok",
        "retrieval": retrieval,
        "draft_reasoning": draft,
        "verification": verification,
        "uncertainty": uncertainty,
        "final_output": {
            "verdict": final_verdict,
            "summary": final_summary,
            "confidence_note": confidence_note,
            "citations": linked,
        },
        "disclaimer": DISCLAIMER,
    }
