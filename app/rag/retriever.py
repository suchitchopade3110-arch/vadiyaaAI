from dotenv import load_dotenv
load_dotenv()

"""
app/rag/retriever.py
--------------------
ChromaDB retrieval layer — extracted from Subhiksha's Week 1-4 Colab notebook.
Owner: LLM/RAG Engineer (Subhiksha)
Wired by: Backend Lead (Suchit) via app/rag/reasoner.py → tasks.py

Responsibilities:
- Load BioGPT tokenizer + model ONCE at module startup
- Load ChromaDB collection ONCE at module startup
- expose retrieve_evidence(query, top_k) → dict

ChromaDB path: set via env var CHROMA_PATH
  default: data/chromadb
  override in .env: CHROMA_PATH=/absolute/path/to/chromadb
"""

import logging
import os
import re
import sqlite3
from datetime import timezone, datetime
UTC = timezone.utc
from pathlib import Path

import chromadb
from chromadb.config import Settings
import numpy as np
import torch
from transformers import BioGptModel, BioGptTokenizer

try:
    import chromadb.segment.impl.metadata.sqlite as chroma_sqlite
    import chromadb.segment.impl.vector.local_persistent_hnsw as chroma_hnsw

    _original_decode_seq_id = chroma_sqlite._decode_seq_id
    _original_load_hnsw_metadata = chroma_hnsw.PersistentData.load_from_file

    def _decode_seq_id_compat(seq_id):
        if isinstance(seq_id, int):
            return seq_id
        return _original_decode_seq_id(seq_id)

    def _load_hnsw_metadata_compat(filename):
        data = _original_load_hnsw_metadata(filename)
        if isinstance(data, dict):
            dimensionality = data.get("dimensionality")
            if dimensionality is None:
                sqlite_path = Path(filename).resolve().parent.parent / "chroma.sqlite3"
                with sqlite3.connect(sqlite_path) as conn:
                    row = conn.execute(
                        "SELECT vector FROM embeddings_queue WHERE vector IS NOT NULL LIMIT 1"
                    ).fetchone()
                if row and row[0]:
                    dimensionality = len(row[0]) // 4

            return chroma_hnsw.PersistentData(
                dimensionality=dimensionality,
                total_elements_added=data.get("total_elements_added", 0),
                max_seq_id=data.get("max_seq_id"),
                id_to_label=data.get("id_to_label", {}),
                label_to_id=data.get("label_to_id", {}),
                id_to_seq_id=data.get("id_to_seq_id", {}),
            )
        return data

    chroma_sqlite._decode_seq_id = _decode_seq_id_compat
    chroma_hnsw.PersistentData.load_from_file = staticmethod(_load_hnsw_metadata_compat)
except Exception:
    pass

log = logging.getLogger(__name__)

# ── Config from env ───────────────────────────────────────────────────────────
CHROMA_PATH = os.path.abspath(os.path.expanduser(os.getenv(
    "CHROMA_PATH",
    "data/chromadb"
)))
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "medical_evidence_week1_clean")
BIOGPT_MODEL_ID   = os.getenv("BIOGPT_MODEL_ID", "microsoft/biogpt")

TOP_K_DEFAULT     = int(os.getenv("RAG_TOP_K", 5))
EMBED_BATCH_SIZE  = int(os.getenv("RAG_EMBED_BATCH_SIZE", 8))
MAX_TOKEN_LENGTH  = int(os.getenv("RAG_MAX_TOKEN_LENGTH", 512))

# ── Lazy singletons ───────────────────────────────────────────────────────────
_tokenizer  = None
_model      = None
_collection = None
_device     = None


def _get_device() -> str:
    global _device
    if _device is None:
        _device = "cuda" if torch.cuda.is_available() else "cpu"
    return _device


def _load_biogpt():
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        log.info(f"Loading BioGPT from {BIOGPT_MODEL_ID} on {_get_device()}")
        _tokenizer = BioGptTokenizer.from_pretrained(BIOGPT_MODEL_ID)
        _model     = BioGptModel.from_pretrained(BIOGPT_MODEL_ID).to(_get_device())
        _model.eval()
        log.info("BioGPT loaded OK")
    return _tokenizer, _model


def _load_collection():
    global _collection
    if _collection is None:
        if not os.path.exists(CHROMA_PATH):
            raise FileNotFoundError(
                f"ChromaDB not found at {CHROMA_PATH}. "
                f"Set CHROMA_PATH env var to correct path."
            )
        client      = chromadb.PersistentClient(
            path=CHROMA_PATH,
            settings=Settings(anonymized_telemetry=False),
        )
        _collection = client.get_collection(name=CHROMA_COLLECTION)
        log.info(f"ChromaDB loaded: {_collection.count()} chunks")
    return _collection


# ── Embedding ─────────────────────────────────────────────────────────────────
def embed_texts(texts: list[str], batch_size: int = EMBED_BATCH_SIZE) -> list[list[float]]:
    """BioGPT mean-pool embeddings. ENCODER ONLY — never generates text."""
    tokenizer, model = _load_biogpt()
    device = _get_device()
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=MAX_TOKEN_LENGTH
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).cpu().tolist()

        all_embeddings.extend(embeddings)

    return all_embeddings


# ── Scoring helpers ───────────────────────────────────────────────────────────
def _safe_cosine(a, b) -> float:
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom != 0 else 0.0


def _keyword_overlap(query: str, text: str) -> float:
    q = set(re.findall(r"[a-z0-9]+", query.lower()))
    t = set(re.findall(r"[a-z0-9]+", text.lower()))
    return len(q & t) / len(q) if q else 0.0


def _source_diversity(scored: list[dict], top_k: int, max_per_source: int = 2) -> list[dict]:
    kept, counts = [], {}
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


# ── PUBLIC API ────────────────────────────────────────────────────────────────
def retrieve_evidence(query: str, top_k: int = TOP_K_DEFAULT) -> dict:
    """
    Retrieve top-k medical evidence chunks for a query.

    Returns:
    {
        "query": str,
        "results": [
            {
                "id", "text", "source", "source_file", "title",
                "url", "section_heading", "page_number",
                "chunk_index", "score", "retrieval_score",
                "rerank_score", "keyword_score"
            }
        ],
        "count": int,
        "timestamp": str
    }
    """
    try:
        collection = _load_collection()
    except (FileNotFoundError, Exception) as e:
        log.error(f"ChromaDB load failed: {e}")
        return {
            "results": [],
            "sources": [],
            "source_count": 0,
            "retrieved_context": "",
            "uncertainty_flag": True
        }

    try:
        total_docs = collection.count()
    except Exception:
        total_docs = 0

    if total_docs == 0:
        return {
            "results": [],
            "sources": [],
            "source_count": 0,
            "retrieved_context": "",
            "uncertainty_flag": True
        }

    query_embedding = embed_texts([query], batch_size=1)[0]
    candidate_k     = min(max(top_k * 4, 12), total_docs)

    raw = collection.query(
        query_embeddings=[query_embedding],
        n_results=candidate_k,
        include=["documents", "metadatas", "distances"]
    )

    ids, docs, metas, distances = (
        raw["ids"][0], raw["documents"][0],
        raw["metadatas"][0], raw["distances"][0]
    )

    if not docs:
        return {"query": query, "results": [], "count": 0,
                "timestamp": datetime.now(UTC).isoformat()}

    doc_embeddings = embed_texts(docs, batch_size=min(8, len(docs)))
    scored = []

    for doc_id, text, meta, dist, doc_emb in zip(ids, docs, metas, distances, doc_embeddings):
        r_score   = round(1 - float(dist), 4)
        rr_score  = round(_safe_cosine(query_embedding, doc_emb), 4)
        kw_score  = round(_keyword_overlap(query, text), 4)
        combined  = round(0.60 * rr_score + 0.25 * r_score + 0.15 * kw_score, 4)

        scored.append({
            "id":              doc_id,
            "text":            text,
            "source":          meta.get("source_name", ""),
            "source_file":     meta.get("source_file", ""),
            "title":           meta.get("title", ""),
            "url":             meta.get("url", ""),
            "section_heading": meta.get("section_heading", ""),
            "page_number":     meta.get("page_number", ""),
            "chunk_index":     meta.get("chunk_index", ""),
            "score":           combined,
            "retrieval_score": r_score,
            "rerank_score":    rr_score,
            "keyword_score":   kw_score,
        })

    # filter weak results
    scored = [r for r in scored if r["rerank_score"] >= 0.45 or r["keyword_score"] >= 0.20]
    scored = sorted(scored, key=lambda x: x["score"], reverse=True)
    scored = _source_diversity(scored, top_k=top_k)

    return {
        "query":     query,
        "results":   scored,
        "count":     len(scored),
        "timestamp": datetime.now(UTC).isoformat(),
    }


def health_check() -> dict:
    """Called at FastAPI startup to verify ChromaDB + BioGPT load."""
    try:
        col   = _load_collection()
        _load_biogpt()
        return {"status": "ok", "chroma_count": col.count(), "device": _get_device()}
    except Exception as e:
        return {"status": "error", "detail": str(e)}
