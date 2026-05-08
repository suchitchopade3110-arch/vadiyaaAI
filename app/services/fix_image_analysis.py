"""Image-analysis fixes for severity bands and WHO imaging RAG retrieval."""

from __future__ import annotations

import json
import re
from typing import Any, Optional


IMAGING_TOPICS = [
    "imaging_standards",
    "radiology",
    "chest_xray",
    "ct",
    "CT",
    "WHO_Diagnostic_Imaging_Protocols",
    "imaging",
]

CONDITION_QUERIES = {
    "nodule": "pulmonary nodule chest xray detection follow-up CT recommendation",
    "emphysema": "emphysema COPD air trapping lung hyperinflation radiograph findings",
    "cardiomegaly": "cardiomegaly enlarged heart chest xray cardiothoracic ratio",
    "pneumonia": "pneumonia consolidation opacity chest xray clinical findings",
    "pneumothorax": "pneumothorax pleural air chest xray emergency findings",
    "fibrosis": "pulmonary fibrosis lung scarring chronic disease radiograph",
    "infiltration": "lung infiltration opacity infection inflammation chest xray",
    "pleural": "pleural effusion blunting costophrenic angle chest xray",
    "atelectasis": "atelectasis lung collapse volume loss chest xray",
    "tuberculosis": "tuberculosis TB chest xray upper lobe cavitation WHO screening",
}


def _to_percent(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return round(number * 100, 1) if 0 <= number <= 1 else round(number, 1)


def get_severity(confidence_pct: float) -> str:
    """Map calibrated confidence to severity: >=80 HIGH, 50-79 MODERATE, <50 LOW."""
    confidence_pct = _to_percent(confidence_pct)
    if confidence_pct >= 80:
        return "HIGH"
    if confidence_pct >= 50:
        return "MODERATE"
    return "LOW"


def get_severity_color(severity: str) -> str:
    return {
        "HIGH": "#b2182b",
        "MODERATE": "#EF9F27",
        "LOW": "#1D9E75",
    }.get(str(severity or "").upper(), "#888")


def build_classification_findings(raw_predictions: list[dict]) -> list[dict]:
    """Build frontend-ready findings with separate class and detection labels."""
    findings = []
    for pred in raw_predictions or []:
        label = pred.get("label", "Unknown")
        class_prob = _to_percent(pred.get("classification_prob", pred.get("probability", pred.get("confidence", 0))))
        detect_conf = _to_percent(pred.get("detection_confidence", pred.get("confidence", class_prob)))
        severity = get_severity(detect_conf)
        findings.append(
            {
                "label": label,
                "classification_prob": round(class_prob, 1),
                "detection_confidence": round(detect_conf, 1),
                "confidence": round(detect_conf, 1),
                "severity": severity,
                "severity_color": get_severity_color(severity),
                "description": pred.get("description") or pred.get("clinical_meaning", ""),
                "label_classification": "Class probability",
                "label_detection": "Detection confidence",
            }
        )
    order = {"HIGH": 0, "MODERATE": 1, "LOW": 2}
    findings.sort(key=lambda item: (order.get(item["severity"], 3), -item["detection_confidence"]))
    return findings


def build_image_rag_query(label: str, gradcam_regions: list[str] | None = None) -> str:
    key = str(label or "").lower().replace("_", " ")
    base = next((query for token, query in CONDITION_QUERIES.items() if token in key), None)
    if base is None:
        base = f"{label} chest xray clinical findings imaging WHO diagnostic imaging"
    regions = " ".join((gradcam_regions or [])[:2])
    return f"{base} {regions}".strip()


def _parse_chroma_results(raw: dict, top_k: int, min_score: float = 0.35) -> list[dict]:
    docs = (raw.get("documents") or [[]])[0]
    metas = (raw.get("metadatas") or [[]])[0]
    distances = (raw.get("distances") or [[]])[0]
    ids = (raw.get("ids") or [[]])[0]

    results = []
    for index, (doc, meta, distance) in enumerate(zip(docs, metas, distances)):
        meta = meta or {}
        score = round(1.0 - float(distance), 4) if distance is not None else 0.0
        if score < min_score:
            continue
        source = meta.get("source_name") or meta.get("source") or meta.get("source_file") or ""
        results.append(
            {
                "id": meta.get("chunk_id") or meta.get("id") or (ids[index] if index < len(ids) else ""),
                "text": str(doc)[:300],
                "source": source,
                "source_file": meta.get("source_file", ""),
                "title": meta.get("title") or meta.get("pattern") or source or "WHO Imaging Standards",
                "url": meta.get("url", ""),
                "score": score,
                "topic": meta.get("topic") or meta.get("category") or "",
                "pattern": meta.get("pattern") or "",
                "urgency": meta.get("urgency") or "",
                "associated_conditions": _json_or_empty(meta.get("associated_conditions")),
            }
        )
        if len(results) >= top_k:
            break
    return results


def _keyword_overlap(query: str, text: str) -> float:
    query_tokens = {token for token in re.split(r"[^a-z0-9]+", query.lower()) if len(token) > 2}
    text_tokens = {token for token in re.split(r"[^a-z0-9]+", text.lower()) if len(token) > 2}
    return len(query_tokens & text_tokens) / max(len(query_tokens), 1)


def _metadata_keyword_results(collection, query: str, top_k: int) -> list[dict]:
    """Fallback for small curated WHO imaging sets when vector query misses metadata rows."""
    rows = []
    filters = [
        {"topic": "imaging_standards"},
        {"source_name": "WHO Diagnostic Imaging Standards"},
        {"source_name": "WHO_Diagnostic_Imaging_Protocols"},
    ]
    seen = set()
    for where in filters:
        try:
            raw = collection.get(where=where, include=["documents", "metadatas"], limit=50)
        except Exception:
            continue
        for item_id, doc, meta in zip(raw.get("ids", []), raw.get("documents", []), raw.get("metadatas", [])):
            if item_id in seen:
                continue
            seen.add(item_id)
            meta = meta or {}
            haystack = " ".join(
                str(value)
                for value in (
                    doc,
                    meta.get("title"),
                    meta.get("section_heading"),
                    meta.get("condition"),
                    meta.get("topic"),
                )
            )
            score = _keyword_overlap(query, haystack)
            rows.append(
                {
                    "id": meta.get("chunk_id") or item_id,
                    "text": str(doc)[:300],
                    "source": meta.get("source_name") or meta.get("source_file") or "WHO Diagnostic Imaging Standards",
                    "source_file": meta.get("source_file", ""),
                    "title": meta.get("title") or meta.get("section_heading") or "WHO Imaging Standards",
                    "url": meta.get("url", ""),
                    "score": round(score, 4),
                    "topic": meta.get("topic", ""),
                    "pattern": meta.get("section_heading", ""),
                    "urgency": "",
                    "associated_conditions": [],
                }
            )
    rows.sort(key=lambda item: item["score"], reverse=True)
    return [row for row in rows if row["score"] > 0][:top_k] or rows[:top_k]


def _json_or_empty(value: Any) -> list:
    if isinstance(value, list):
        return value
    if not value:
        return []
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


def retrieve_image_evidence(
    collection,
    embed_texts_fn,
    classification_label: str,
    gradcam_regions: list[str] | None = None,
    top_k: int = 3,
) -> list[dict]:
    """Retrieve WHO imaging evidence, then retry broadly before returning empty."""
    query = build_image_rag_query(classification_label, gradcam_regions)
    query_embedding = embed_texts_fn([query], batch_size=1)[0]
    total_docs = int(collection.count())
    if total_docs <= 0:
        return []
    candidate_k = min(top_k * 5, total_docs)

    try:
        raw = collection.query(
            query_embeddings=[query_embedding],
            n_results=candidate_k,
            where={"topic": {"$in": IMAGING_TOPICS}},
            include=["documents", "metadatas", "distances"],
        )
        results = _parse_chroma_results(raw, top_k, min_score=0.35)
        if len(results) >= 2:
            return results
        metadata_results = _metadata_keyword_results(collection, query, top_k)
        if len(metadata_results) >= 2:
            return metadata_results
        print(f"[RAG] Imaging filter returned {len(results)} results for '{classification_label}' - retrying broad")
    except Exception as exc:
        print(f"[RAG] Imaging filter failed for '{classification_label}': {exc} - retrying broad")
        metadata_results = _metadata_keyword_results(collection, query, top_k)
        if len(metadata_results) >= 2:
            return metadata_results

    try:
        raw = collection.query(
            query_embeddings=[query_embedding],
            n_results=candidate_k,
            include=["documents", "metadatas", "distances"],
        )
        results = _parse_chroma_results(raw, top_k, min_score=0.30)
        return results or _metadata_keyword_results(collection, query, top_k)
    except Exception as exc:
        print(f"[RAG] Broad imaging retrieval failed for '{classification_label}': {exc}")
        return []


def retrieve_image_evidence_from_main_kb(
    classification_label: str,
    gradcam_regions: list[str] | None = None,
    top_k: int = 3,
) -> list[dict]:
    """Convenience wrapper around the app's main Chroma collection."""
    from app.rag.retriever import _load_collection, embed_texts

    return retrieve_image_evidence(
        collection=_load_collection(),
        embed_texts_fn=embed_texts,
        classification_label=classification_label,
        gradcam_regions=gradcam_regions,
        top_k=top_k,
    )


def filter_keyword_fallback_sources(sources: list[dict]) -> list[dict]:
    """Keep keyword_fallback only when needed and collapse duplicate citations."""
    rows = list(sources or [])
    non_fallback = [
        row for row in rows
        if str(row.get("source") or row.get("source_name") or "").lower() != "keyword_fallback"
    ]
    rows = non_fallback or rows

    unique = []
    seen_ids = set()
    seen_titles = set()
    for row in rows:
        row_id = str(row.get("id") or row.get("chunk_id") or "").strip()
        title_key = str(row.get("title") or row.get("source") or row.get("source_name") or "").strip().lower()[:80]
        if row_id and row_id in seen_ids:
            continue
        if title_key and title_key in seen_titles:
            continue
        if row_id:
            seen_ids.add(row_id)
        if title_key:
            seen_titles.add(title_key)
        unique.append(row)
    return unique


def run_image_explanation_chain_fixed(
    collection,
    embed_texts_fn,
    llm_chain,
    classification_label: str,
    chexnet_confidence: float,
    gradcam_regions: list[str] | None = None,
    top_k: int = 3,
) -> dict:
    """Colab-compatible fixed image explanation chain."""
    confidence_pct = _to_percent(chexnet_confidence)
    severity = get_severity(confidence_pct)
    evidence = retrieve_image_evidence(collection, embed_texts_fn, classification_label, gradcam_regions, top_k)
    evidence_text = "\n".join(
        f"[{index + 1}] {row.get('source') or row.get('title')}: {row.get('text', '')}"
        for index, row in enumerate(evidence)
    ) or "No specific evidence retrieved - use conservative clinical language."
    regions = ", ".join(gradcam_regions or []) or "not specified"
    prompt = f"""You are a medical AI assistant interpreting chest X-ray analysis results.

Classification: {classification_label}
Detection confidence: {confidence_pct}% ({severity})
GradCAM activated regions: {regions}

Retrieved medical evidence:
{evidence_text}

Return JSON with clinical_interpretation, patient_friendly_summary, recommendations, uncertainty_flag, verdict.
"""
    try:
        raw = llm_chain.invoke({"input": prompt})
        if not isinstance(raw, str):
            raw = getattr(raw, "content", str(raw))
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        parsed = json.loads(match.group()) if match else {"clinical_interpretation": raw}
    except Exception as exc:
        print(f"[LLM] Image explanation failed: {exc}")
        parsed = {
            "clinical_interpretation": f"{classification_label} detected with {confidence_pct}% confidence.",
            "patient_friendly_summary": f"The AI detected possible {classification_label}.",
            "recommendations": ["Consult a radiologist for clinical interpretation."],
            "uncertainty_flag": confidence_pct < 60,
            "verdict": "Uncertain",
        }

    return {
        "classification_label": classification_label,
        "classification_prob": confidence_pct,
        "detection_confidence": confidence_pct,
        "label_classification": "Class probability",
        "label_detection": "Detection confidence",
        "severity": severity,
        "severity_color": get_severity_color(severity),
        "clinical_interpretation": parsed.get("clinical_interpretation", ""),
        "patient_friendly_summary": parsed.get("patient_friendly_summary", ""),
        "recommendations": parsed.get("recommendations", []),
        "verdict": parsed.get("verdict", "Uncertain"),
        "uncertainty_flag": bool(parsed.get("uncertainty_flag", confidence_pct < 60)),
        "sources": evidence,
        "source_count": len(evidence),
    }


def fix_severity_in_response(api_response: dict) -> dict:
    """Post-process an image response dict with corrected confidence severity bands."""
    result = api_response.get("result", api_response)
    classification = result.get("classification", {})
    conf = _to_percent(classification.get("confidence", classification.get("top_confidence", result.get("confidence_score", 0))))
    classification["severity"] = get_severity(conf)
    for finding in result.get("findings", []) or []:
        raw_conf = finding.get("detection_confidence", finding.get("confidence", 0))
        finding["severity"] = get_severity(raw_conf)
        finding["severity_color"] = get_severity_color(finding["severity"])
    return api_response
