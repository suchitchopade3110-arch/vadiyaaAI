"""Panel-aware RAG filters for report evidence retrieval."""

from __future__ import annotations

from typing import Optional


PANEL_TOPIC_MAP = {
    "urinalysis": {"urinalysis", "urine_analysis", "renal", "nephrology", "kidney"},
    "haematology": {"haematology", "hematology", "blood_count", "anaemia", "anemia", "cbc", "dengue", "esr"},
    "biochemistry": {"biochemistry", "electrolytes", "metabolic", "liver_function", "renal_function", "diabetes"},
    "serology": {"serology", "immunology", "dengue", "crp", "inflammation", "infection"},
}

IMAGING_SOURCE_TOKENS = {
    "who diagnostic imaging",
    "who imaging",
    "diagnostic imaging",
    "radiology",
    "imaging",
    "chexnet",
    "gradcam",
    "xray",
    "x-ray",
    "ct interpretation",
    "dicom",
}

PATHOLOGY_SOURCE_TOKENS = {
    "who pathology",
    "digital pathology",
    "blue books",
    "cytopathology",
    "wsi",
}


def build_chroma_where_filter(panel_type: str, exclude_imaging: bool = True) -> Optional[dict]:
    """Build a conservative Chroma metadata filter when topic metadata is available."""
    allowed_topics = PANEL_TOPIC_MAP.get(str(panel_type or "").lower(), set())
    if not allowed_topics:
        return None
    return {"topic": {"$in": sorted(allowed_topics)}} if not exclude_imaging else {"topic": {"$in": sorted(allowed_topics)}}


def is_report_source_allowed(source: dict, panel_type: str = "general", report_type: str = "lab") -> bool:
    """Return False for imaging/pathology evidence in lab report RAG results."""
    if str(report_type or "").lower() not in {"lab", "urinalysis", "report"}:
        return True
    haystack = " ".join(
        str(source.get(key, ""))
        for key in ("source", "source_name", "source_file", "title", "category", "topic", "id")
    ).lower()
    if any(token in haystack for token in IMAGING_SOURCE_TOKENS):
        return False
    if any(token in haystack for token in PATHOLOGY_SOURCE_TOKENS):
        return False

    allowed_topics = PANEL_TOPIC_MAP.get(str(panel_type or "").lower(), set())
    topic = str(source.get("topic") or source.get("panel_topic") or "").lower()
    if allowed_topics and topic and topic not in allowed_topics:
        return False
    return True


def filter_report_sources(sources: list[dict], panel_type: str = "general", report_type: str = "lab") -> list[dict]:
    return [source for source in sources or [] if is_report_source_allowed(source, panel_type=panel_type, report_type=report_type)]


def retrieve_evidence_for_findings_filtered(
    findings: list[dict],
    collection,
    embed_texts_fn,
    panel_type: str,
    top_k: int = 3,
) -> dict:
    """Retrieve Chroma evidence per lab finding and drop imaging/pathology citations."""
    where_filter = build_chroma_where_filter(panel_type, exclude_imaging=True)
    evidence_map = {}

    for finding in findings or []:
        param = finding.get("name") or finding.get("param") or finding.get("test") or ""
        value = finding.get("value") or finding.get("result") or ""
        flag = finding.get("flag") or finding.get("status") or ""
        query = f"{param} {value} {flag} lab finding clinical significance".strip()

        query_embedding = embed_texts_fn([query])[0]
        query_kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where_filter:
            query_kwargs["where"] = where_filter

        try:
            results = collection.query(**query_kwargs)
        except Exception as exc:
            print(f"[RAG] Filter failed for '{param}': {exc} - retrying without metadata filter")
            query_kwargs.pop("where", None)
            results = collection.query(**query_kwargs)

        docs = (results.get("documents") or [[]])[0]
        metas = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]

        rows = []
        for doc, meta, distance in zip(docs, metas, distances):
            source = {
                "id": meta.get("chunk_id", meta.get("doc_id", "")),
                "text": str(doc)[:300],
                "source": meta.get("source_name", ""),
                "title": meta.get("title", ""),
                "url": meta.get("url", ""),
                "topic": meta.get("topic", ""),
                "score": round(1 - float(distance), 4),
            }
            if source["score"] >= 0.3 and is_report_source_allowed(source, panel_type, "lab"):
                rows.append(source)
        evidence_map[param] = rows

    return evidence_map


def patch_pipeline(pipeline_module, collection, embed_texts_fn):
    """Monkey-patch a notebook/module retrieval function with the panel-aware version."""

    def _filtered_retrieve(findings, top_k=3, panel_type="general"):
        return retrieve_evidence_for_findings_filtered(
            findings=findings,
            collection=collection,
            embed_texts_fn=embed_texts_fn,
            panel_type=panel_type,
            top_k=top_k,
        )

    pipeline_module.retrieve_evidence_for_findings = _filtered_retrieve
    print("[RAG] Patched retrieve_evidence_for_findings with panel-aware lab filter")


def run_phase2_pipeline_fixed(
    patient_id: str,
    pdf_path: str,
    collection,
    embed_texts_fn,
    generate_report_fn,
    self_verify_fn,
    extract_citations_fn,
    top_k: int = 3,
) -> dict:
    """Standalone Colab-compatible fixed Phase 2 pipeline runner."""
    from app.services.lab_report_column_parser import parse_lab_report_for_pipeline

    parsed = parse_lab_report_for_pipeline(pdf_path)
    panel_type = parsed["panel_type"]
    findings = [
        {
            "name": row["name"],
            "value": row["value"],
            "unit": row["unit"],
            "reference": row["reference"],
            "flag": row["flag"],
            "section": row["section"],
            "status": "abnormal" if row["flag"] != "NORMAL" else "normal",
        }
        for row in parsed["lab_values"]
    ]
    abnormal_findings = [row for row in findings if row["status"] == "abnormal"]
    evidence_map = retrieve_evidence_for_findings_filtered(
        abnormal_findings,
        collection=collection,
        embed_texts_fn=embed_texts_fn,
        panel_type=panel_type,
        top_k=top_k,
    )

    report = generate_report_fn(
        {
            "patient_id": patient_id,
            "findings": findings,
            "abnormal_count": parsed["abnormal_count"],
            "normal_count": parsed["normal_count"],
            "panel_type": panel_type,
        },
        evidence_map,
    )
    report = self_verify_fn(report, evidence_map)
    report = extract_citations_fn(report, evidence_map)
    report["panel_type"] = panel_type
    report["abnormal_items"] = parsed["abnormal_items"]
    report["lab_values"] = parsed["lab_values"]
    return report
