"""ChromaDB ingestion and query helpers for the radiology pattern KB."""

from __future__ import annotations

import json
import hashlib
import os
import re
from pathlib import Path
from typing import Any

from app.services.radiology_knowledge_base import (
    CHEXNET_LABELS_EXTENDED,
    RADIOLOGY_KNOWLEDGE_BASE,
    REGION_TEMPLATE_MAP,
    REPORT_TEMPLATES,
)

CHROMA_PATH = Path(os.getenv("RADIOLOGY_CHROMA_PATH", "data/chromadb/radiology"))
PATTERN_COLLECTION = "radiology_patterns"
LABEL_COLLECTION = "chexnet_labels"


def _doc_text(entry: dict[str, Any]) -> str:
    return (
        f"Pattern: {entry['pattern']}. "
        f"Description: {entry['description']} "
        f"Associated conditions: {', '.join(entry['associated_conditions'])}. "
        f"Body region: {entry['body_region']}. "
        f"Modality: {', '.join(entry['modality'])}. "
        f"Keywords: {', '.join(entry['keywords'])}."
    )


def _embedding_function():
    class HashEmbeddingFunction:
        """Small deterministic embedding function that avoids model downloads."""

        def __init__(self, dimensions: int = 96):
            self.dimensions = dimensions

        def __call__(self, input):
            texts = input if isinstance(input, list) else [input]
            return [self._embed(text) for text in texts]

        def _embed(self, text: str) -> list[float]:
            vector = [0.0] * self.dimensions
            for token in _tokens(str(text)):
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                index = int.from_bytes(digest[:4], "big") % self.dimensions
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                vector[index] += sign
            norm = sum(value * value for value in vector) ** 0.5 or 1.0
            return [value / norm for value in vector]

    return HashEmbeddingFunction()


def _client_and_collections():
    import chromadb
    from chromadb.config import Settings

    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(CHROMA_PATH),
        settings=Settings(anonymized_telemetry=False),
    )
    embedding_fn = _embedding_function()
    kwargs = {"embedding_function": embedding_fn} if embedding_fn is not None else {}
    patterns = client.get_or_create_collection(
        name=PATTERN_COLLECTION,
        metadata={"hnsw:space": "cosine"},
        **kwargs,
    )
    labels = client.get_or_create_collection(name=LABEL_COLLECTION, **kwargs)
    return client, patterns, labels


def ingest_knowledge_base() -> int:
    """Ingest radiology pattern documents into ChromaDB."""
    _client, collection, _labels = _client_and_collections()
    collection.upsert(
        ids=[entry["id"] for entry in RADIOLOGY_KNOWLEDGE_BASE],
        documents=[_doc_text(entry) for entry in RADIOLOGY_KNOWLEDGE_BASE],
        metadatas=[
            {
                "category": entry["category"],
                "pattern": entry["pattern"],
                "body_region": entry["body_region"],
                "urgency": entry["urgency"],
                "modality": json.dumps(entry["modality"]),
                "associated_conditions": json.dumps(entry["associated_conditions"]),
                "source": "WHO_Diagnostic_Imaging_Protocols",
            }
            for entry in RADIOLOGY_KNOWLEDGE_BASE
        ],
    )
    return len(RADIOLOGY_KNOWLEDGE_BASE)


def ingest_label_metadata() -> int:
    """Ingest CheXNet label ICD-10 and urgency metadata into ChromaDB."""
    _client, _patterns, collection = _client_and_collections()
    labels = list(CHEXNET_LABELS_EXTENDED.items())
    collection.upsert(
        ids=[f"label_{label}" for label, _meta in labels],
        documents=[
            (
                f"CheXNet classification label: {label}. "
                f"ICD-10 code: {meta['icd10']}. "
                f"Urgency: {meta['urgency']}. "
                f"Body region: {meta['body_region']}."
            )
            for label, meta in labels
        ],
        metadatas=[{**meta, "label": label, "source": "WHO_CheXNet_Extended"} for label, meta in labels],
    )
    return len(labels)


def ingest_all() -> dict[str, int]:
    return {
        "radiology_patterns": ingest_knowledge_base(),
        "chexnet_labels": ingest_label_metadata(),
    }


def _tokens(text: str) -> set[str]:
    return {token for token in re.split(r"[^a-z0-9]+", text.lower()) if len(token) > 2}


def _keyword_query(query: str, n_results: int = 3, category: str | None = None) -> list[dict[str, Any]]:
    query_tokens = _tokens(query)
    scored = []
    for entry in RADIOLOGY_KNOWLEDGE_BASE:
        if category and entry["category"] != category:
            continue
        haystack = " ".join([
            entry["pattern"],
            entry["description"],
            " ".join(entry["associated_conditions"]),
            " ".join(entry["keywords"]),
            entry["body_region"],
            entry["category"],
        ])
        score = len(query_tokens & _tokens(haystack))
        if score:
            scored.append((score, entry))

    scored.sort(key=lambda item: (-item[0], item[1]["id"]))
    return [
        {
            "id": entry["id"],
            "pattern": entry["pattern"],
            "category": entry["category"],
            "urgency": entry["urgency"],
            "associated_conditions": entry["associated_conditions"],
            "text": _doc_text(entry),
            "score": round(score / max(len(query_tokens), 1), 3),
            "source": "keyword_fallback",
        }
        for score, entry in scored[:n_results]
    ]


def query_radiology_kb(query: str, n_results: int = 3, category: str | None = None) -> list[dict[str, Any]]:
    """Query ChromaDB when available, falling back to deterministic keyword matching."""
    keyword_results = _keyword_query(query, n_results=n_results, category=category)
    if keyword_results and (keyword_results[0].get("score") or 0) >= 0.2:
        return keyword_results

    try:
        _client, collection, _labels = _client_and_collections()
        where = {"category": category} if category else None
        results = collection.query(query_texts=[query], n_results=n_results, where=where)
        output = []
        for index, item_id in enumerate(results.get("ids", [[]])[0]):
            metadata = results["metadatas"][0][index]
            distance = (results.get("distances") or [[None]])[0][index]
            output.append(
                {
                    "id": item_id,
                    "pattern": metadata["pattern"],
                    "category": metadata["category"],
                    "urgency": metadata["urgency"],
                    "associated_conditions": json.loads(metadata["associated_conditions"]),
                    "text": results["documents"][0][index],
                    "score": round(1 - distance, 3) if distance is not None else None,
                    "source": metadata.get("source", "chroma"),
                }
            )
        if output:
            return output
    except Exception:
        pass
    return keyword_results


def get_label_metadata(label: str) -> dict[str, str]:
    normalized = str(label or "").strip()
    if normalized in CHEXNET_LABELS_EXTENDED:
        return CHEXNET_LABELS_EXTENDED[normalized]

    compact = normalized.replace(" ", "_")
    if compact in CHEXNET_LABELS_EXTENDED:
        return CHEXNET_LABELS_EXTENDED[compact]

    lowered = normalized.lower().replace("_", " ")
    for known_label, meta in CHEXNET_LABELS_EXTENDED.items():
        if known_label.lower().replace("_", " ") == lowered:
            return meta
    return {"icd10": "R91.8", "urgency": "unknown", "body_region": "unknown"}


def get_report_template(body_region: str) -> dict[str, Any]:
    template_key = REGION_TEMPLATE_MAP.get(str(body_region or "").lower(), "thoracic")
    return REPORT_TEMPLATES.get(template_key, REPORT_TEMPLATES["thoracic"])


if __name__ == "__main__":
    counts = ingest_all()
    print(f"Ingested {counts['radiology_patterns']} radiology patterns")
    print(f"Ingested {counts['chexnet_labels']} CheXNet labels")
    for result in query_radiology_kb("pneumonia consolidation right lower lobe", n_results=3):
        print(f"[{result['score']}] {result['pattern']} ({result['urgency']})")
