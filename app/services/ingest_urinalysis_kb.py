"""ChromaDB ingestion/query helpers for urinalysis patterns."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from app.services.urinalysis_knowledge_base import URINALYSIS_KNOWLEDGE_BASE


CHROMA_PATH = Path(os.getenv("URINALYSIS_CHROMA_PATH", "data/chromadb/urinalysis"))
COLLECTION_NAME = "urinalysis_patterns"


def _tokens(text: str) -> set[str]:
    return {token for token in re.split(r"[^a-z0-9]+", str(text).lower()) if len(token) > 2}


def _doc_text(entry: dict[str, Any]) -> str:
    return (
        f"Pattern: {entry['pattern']}. "
        f"Description: {entry['description']} "
        f"Associated conditions: {', '.join(entry['associated_conditions'])}. "
        f"Diagnostic criteria: {entry['diagnostic_criteria']}. "
        f"Threshold: {entry['threshold']}. "
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
            for token in _tokens(text):
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                index = int.from_bytes(digest[:4], "big") % self.dimensions
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                vector[index] += sign
            norm = sum(value * value for value in vector) ** 0.5 or 1.0
            return [value / norm for value in vector]

    return HashEmbeddingFunction()


def _collection():
    import chromadb
    from chromadb.config import Settings

    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(CHROMA_PATH),
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=_embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )


def ingest_urinalysis_kb() -> int:
    """Ingest urinalysis patterns into ChromaDB."""
    collection = _collection()
    collection.upsert(
        documents=[_doc_text(entry) for entry in URINALYSIS_KNOWLEDGE_BASE],
        ids=[entry["id"] for entry in URINALYSIS_KNOWLEDGE_BASE],
        metadatas=[
            {
                "category": entry["category"],
                "pattern": entry["pattern"],
                "urgency": entry["urgency"],
                "icd10": entry.get("icd10", ""),
                "threshold": entry["threshold"],
                "source": "WHO_StatPearls_Urinalysis",
                "associated_conditions": json.dumps(entry["associated_conditions"]),
            }
            for entry in URINALYSIS_KNOWLEDGE_BASE
        ],
    )
    return len(URINALYSIS_KNOWLEDGE_BASE)


def _keyword_query(query: str, n_results: int = 3) -> list[dict[str, Any]]:
    query_tokens = _tokens(query)
    scored = []
    for entry in URINALYSIS_KNOWLEDGE_BASE:
        haystack = " ".join(
            [
                entry["pattern"],
                entry["description"],
                " ".join(entry["associated_conditions"]),
                entry["diagnostic_criteria"],
                entry["threshold"],
                " ".join(entry["keywords"]),
            ]
        )
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
            "icd10": entry.get("icd10", ""),
            "text": _doc_text(entry),
            "score": round(score / max(len(query_tokens), 1), 3),
            "source": "keyword_fallback",
        }
        for score, entry in scored[:n_results]
    ]


def query_urinalysis_kb(query: str, n_results: int = 3) -> list[dict[str, Any]]:
    """Query urinalysis patterns; falls back to keyword matching."""
    keyword_results = _keyword_query(query, n_results=n_results)
    if keyword_results and (keyword_results[0].get("score") or 0) >= 0.2:
        return keyword_results
    try:
        collection = _collection()
        results = collection.query(query_texts=[query], n_results=n_results)
        output = []
        for index, item_id in enumerate(results.get("ids", [[]])[0]):
            metadata = results["metadatas"][0][index]
            distance = (results.get("distances") or [[None]])[0][index]
            output.append(
                {
                    "id": item_id,
                    "pattern": metadata["pattern"],
                    "category": metadata.get("category", ""),
                    "urgency": metadata["urgency"],
                    "icd10": metadata.get("icd10", ""),
                    "text": results["documents"][0][index],
                    "score": round(1 - distance, 3) if distance is not None else None,
                    "source": metadata.get("source", "chroma"),
                }
            )
        return output or keyword_results
    except Exception:
        return keyword_results


if __name__ == "__main__":
    count = ingest_urinalysis_kb()
    print(f"Ingested {count} urinalysis patterns")
    for result in query_urinalysis_kb("protein blood rbc casts hematuria", n_results=3):
        print(f"[{result['score']}] {result['pattern']} ({result['urgency']})")
