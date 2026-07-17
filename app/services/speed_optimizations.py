"""Latency helpers for the report analysis pipeline.

The functions in this module are intentionally thin wrappers around the
existing NER, RAG, and prediction services. They add Redis-backed caching,
parallel NER/RAG execution, and a startup warmup for the Groq client without
changing downstream result contracts.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from typing import Any

try:
    import redis
except ModuleNotFoundError:  # Keep the app importable in minimal/dev envs.
    redis = None

from app.core.config import settings

log = logging.getLogger(__name__)

NER_CACHE_TTL = 3600
RAG_CACHE_TTL = 7200
XGBOOST_CACHE_TTL = 1800
CACHE_SOCKET_TIMEOUT = 0.25

REDIS_MAX_CONNECTIONS = 20

# Explicit, bounded connection pool shared by every _redis call in this module.
# `redis.Redis.from_url()` would build its own pool implicitly (default
# max_connections is unbounded), so we construct it ourselves to cap it.
_redis_pool = (
    redis.ConnectionPool.from_url(
        settings.REDIS_URL,
        decode_responses=True,
        socket_connect_timeout=CACHE_SOCKET_TIMEOUT,
        socket_timeout=CACHE_SOCKET_TIMEOUT,
        max_connections=REDIS_MAX_CONNECTIONS,
    )
    if redis is not None
    else None
)
_redis = redis.Redis(connection_pool=_redis_pool) if _redis_pool is not None else None
_cache_available: bool | None = False if redis is None else None


def _text_hash(text: str) -> str:
    """Stable short hash for report/query cache keys."""
    return hashlib.sha256((text or "")[:1000].encode("utf-8")).hexdigest()[:16]


def _json_default(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _cache_get(key: str) -> Any | None:
    global _cache_available
    if _redis is None or _cache_available is False:
        return None
    try:
        cached = _redis.get(key)
        _cache_available = True
        if cached:
            return json.loads(cached)
    except Exception as exc:
        _cache_available = False
        log.warning("Redis cache read failed for %s: %s", key, exc)
    return None


def _cache_set(key: str, ttl: int, value: Any) -> None:
    global _cache_available
    if _redis is None or _cache_available is False:
        return
    try:
        _redis.setex(key, ttl, json.dumps(value, default=_json_default))
        _cache_available = True
    except Exception as exc:
        _cache_available = False
        log.warning("Redis cache write failed for %s: %s", key, exc)


def run_ner_cached(text: str):
    """Run the existing NER pipeline with a Redis cache keyed by report text."""
    from app.services.preprocessor import NerOutput, run_ner

    cache_key = f"ner:{_text_hash(text)}"
    cached = _cache_get(cache_key)
    if cached is not None:
        log.info("NER cache HIT (%s)", cache_key)
        return NerOutput(**cached)

    t0 = time.time()
    result = run_ner(text)
    log.info("NER cache MISS (%s), pipeline took %.2fs", cache_key, time.time() - t0)
    _cache_set(cache_key, NER_CACHE_TTL, result)
    return result


def retrieve_evidence_cached(query: str, top_k: int = 5) -> list[dict]:
    """Run existing RAG retrieval with Redis caching."""
    from app.services.rag_pipeline import rag_pipeline

    cache_key = f"rag:{_text_hash(query)}:{top_k}"
    cached = _cache_get(cache_key)
    if cached is not None:
        log.info("RAG cache HIT (%s)", cache_key)
        return cached

    t0 = time.time()
    result = rag_pipeline.retrieve_evidence(query, top_k=top_k)
    log.info("RAG cache MISS (%s), retrieval took %.2fs", cache_key, time.time() - t0)
    _cache_set(cache_key, RAG_CACHE_TTL, result)
    return result


def predict_safe_cached(features: dict) -> dict:
    """Run XGBoost prediction with a cache keyed by the normalized feature dict."""
    from app.ml.predictor import predict_safe

    feature_hash = hashlib.sha256(
        json.dumps(features or {}, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]
    cache_key = f"xgb:{feature_hash}"
    cached = _cache_get(cache_key)
    if cached is not None:
        log.info("XGBoost cache HIT (%s)", cache_key)
        return cached

    result = predict_safe(features)
    _cache_set(cache_key, XGBOOST_CACHE_TTL, result)
    return result


async def run_ner_and_rag_parallel(text: str, top_k: int = 5) -> tuple[Any, list[dict]]:
    """Run blocking NER and RAG services concurrently in worker threads."""
    ner_task = asyncio.to_thread(run_ner_cached, text)
    rag_task = asyncio.to_thread(retrieve_evidence_cached, text[:500], top_k)
    return await asyncio.gather(ner_task, rag_task)


def run_ner_and_rag_parallel_sync(text: str, top_k: int = 5) -> tuple[Any, list[dict]]:
    """Sync wrapper for Celery tasks."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(run_ner_and_rag_parallel(text, top_k=top_k))

    log.warning("Active event loop detected; falling back to sequential NER/RAG")
    return run_ner_cached(text), retrieve_evidence_cached(text[:500], top_k=top_k)


def prewarm_groq() -> None:
    """Warm the Groq client in the background during API startup."""
    if not settings.GROQ_API_KEY:
        log.info("Groq pre-warm skipped: GROQ_API_KEY is not set")
        return

    try:
        from app.services.clinical_ner_service import MODEL, _get_client

        t0 = time.time()
        _get_client().chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
            temperature=0,
        )
        log.info("Groq pre-warm completed in %.2fs", time.time() - t0)
    except Exception as exc:
        log.warning("Groq pre-warm failed (non-fatal): %s", exc)


def verify_cache_connection() -> bool:
    """Check Redis availability without making cache use mandatory."""
    global _cache_available
    if _redis is None:
        _cache_available = False
        log.warning("Redis package is not installed; speed caches disabled")
        return False
    try:
        _cache_available = bool(_redis.ping())
        return _cache_available
    except Exception as exc:
        _cache_available = False
        log.warning("Redis cache unavailable; speed caches disabled: %s", exc)
        return False


def benchmark_pipeline(sample_text: str) -> dict[str, Any]:
    """Quick local timing helper for the optimized text path."""
    timings: dict[str, Any] = {}

    t0 = time.time()
    ner = run_ner_cached(sample_text)
    timings["ner_cold_or_cached"] = round(time.time() - t0, 3)

    t0 = time.time()
    run_ner_cached(sample_text)
    timings["ner_warm_cached"] = round(time.time() - t0, 3)

    t0 = time.time()
    retrieve_evidence_cached(sample_text[:500])
    timings["rag_cached_or_cold"] = round(time.time() - t0, 3)

    features = {
        lab.name.lower(): lab.value
        for lab in getattr(ner, "lab_values", [])
        if lab.name and lab.value is not None
    }
    if features:
        t0 = time.time()
        predict_safe_cached(features)
        timings["xgboost_cached_or_cold"] = round(time.time() - t0, 3)

    t0 = time.time()
    run_ner_and_rag_parallel_sync(sample_text)
    timings["parallel_ner_rag"] = round(time.time() - t0, 3)
    return timings
