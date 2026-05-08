"""Ingest WHO pathology and digital scan standards into the main RAG collection."""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Any


CHROMA_PATH = Path(os.getenv("CHROMA_PATH", "data/chromadb"))
COLLECTION = os.getenv("CHROMA_COLLECTION", "medical_evidence_week1_clean")
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "100"))
UPLOAD_BATCH = int(os.getenv("RAG_UPLOAD_BATCH_SIZE", "32"))
EMBED_DIMENSIONS = int(os.getenv("RAG_EMBED_DIMENSIONS", "1024"))

SOURCE_META = {
    "source_name": "WHO Blue Books & Digital Pathology Standards",
    "source_type": "clinical_guideline",
    "source_file": "WHO_Pathology_Digital_Scan_Standards.txt",
    "title": "Pathological Scan Interpretation and WHO Diagnostic Criteria Framework",
    "url": "https://www.iarc.who.int/cards_page/who-classification-of-tumours/",
    "published_date": "2024-01-01",
    "condition": "pathology,cytology,oncology,WSI,digital_pathology,ICD-O,Blue_Books",
    "topic": "pathology_standards",
}

WHO_PATH_SECTIONS = [
    {
        "heading": "Pre-Analytical Phase - Tissue Preparation for Digital Scanning",
        "text": (
            "Accuracy of digital pathology diagnosis is tied to physical tissue preparation. "
            "Standard single-plane scanning is optimized for tissue sections cut at 3-5 "
            "micrometers. Thicker sections require multi-plane z-stack scanning, increasing "
            "scan time and file size. Histochemical staining must be controlled because faint "
            "or excessive background staining impedes automated tissue detection. Tissue must "
            "be mounted flat; folds, wrinkles, overlapping ribbons, air bubbles, debris, "
            "fingerprints, cracks, and coverslip artifacts can obscure lesions or mimic disease."
        ),
    },
    {
        "heading": "WSI Image Acquisition - Data Architecture of Whole Slide Images",
        "text": (
            "Whole slide image scanners capture many overlapping tiles and stitch them into a "
            "digital replica of the glass slide. Files commonly range from hundreds of "
            "megabytes to several gigabytes. WSI uses pyramidal organization: highest "
            "resolution at base, downsampled layers for lower zoom levels, and a thumbnail at "
            "the apex. This enables selective tile loading for smooth panning and zooming. "
            "Cytology smears with three-dimensional clusters may require z-stack scanning."
        ),
    },
    {
        "heading": "WSI Technical Specifications - Display and Network Standards",
        "text": (
            "Digital pathology image quality must be equivalent to conventional optical "
            "microscopy for diagnostic accuracy and patient safety. Display resolution should "
            "be at least 2560 by 1440 pixels, with 3840 by 2160 preferred. Pixel density above "
            "100 PPI supports sharp nuclear contour evaluation. Color depth should be at least "
            "10-bit per channel; 8-bit can introduce banding that obscures hematoxylin and "
            "eosin variation. Brightness should be at least 300 cd/m2, contrast ratio at least "
            "1000:1, compression visually lossless and typically no more than 20:1, and "
            "network upload at least 100 Mbps sustained. Frozen section scan, processing, "
            "transmission, and review should be available within 5 minutes when used "
            "intraoperatively."
        ),
    },
    {
        "heading": "Clinical Validation Protocol - WSI System Validation for Primary Diagnosis",
        "text": (
            "Clinical validation should assess the entire WSI system as one unit: scanner, "
            "network, viewer, display monitor, and pathologist. A globally recognized CAP "
            "framework uses at least 60 cases per specific application. Each modality is "
            "validated independently, including H&E FFPE sections, frozen sections, cytology "
            "smears, and hematology preparations. Validation uses intraobserver concordance: "
            "pathologist reviews glass slides, waits a two-week washout period, then reviews "
            "digitized slides. Acceptance threshold is at least 95 percent diagnostic "
            "concordance and major discrepancy rate below 4-7 percent. Revalidation is needed "
            "for scanner, software, monitor, or LIS changes."
        ),
    },
    {
        "heading": "Telepathology Regulatory Standards - Remote Interpretation Requirements",
        "text": (
            "Telepathology uses electronic multimedia communication for primary diagnoses, "
            "consultation, and second opinions. Static WSI store-and-forward and dynamic "
            "real-time robotic microscopy are recognized modes. WSI provides static "
            "transmission with navigational freedom. Regulatory use requires technical "
            "performance assessment, CLIA or equivalent accreditation, clinical privileges, "
            "HIPAA or GDPR compliant data protection, end-to-end encryption, and a minimal "
            "dataset including accession number, patient name, block or slide ID, and relevant "
            "clinical history. AI outputs remain AI-assisted analysis only and not diagnosis."
        ),
    },
    {
        "heading": "WHO Blue Books - Classification of Tumours Framework",
        "text": (
            "WHO Classification of Tumours Blue Books from IARC are a definitive global "
            "reference for tumor diagnosis and uniform nomenclature. Tumors are organized by "
            "anatomical site, biological category, family, type, and subtype. Modern "
            "classification integrates morphology with molecular pathology, genomic "
            "sequencing, and digital image analysis. Diagnostic reporting headings include "
            "definition, ICD-O and ICD-11 coding, related terminology, clinical features, "
            "epidemiology, pathogenesis, macroscopic appearance, histopathology, cytology, "
            "diagnostic molecular pathology, essential and desirable criteria, staging, "
            "prognosis, and prediction."
        ),
    },
    {
        "heading": "WHO Essential and Desirable Diagnostic Criteria - Core Framework",
        "text": (
            "WHO tumor classification stratifies diagnostic features into essential and "
            "desirable criteria. Essential criteria are the mandatory minimum parameters for "
            "a specific tumor diagnosis. If scan and clinical data do not fulfill all "
            "essential criteria, the diagnosis should default to a broader family-level "
            "classification or not otherwise specified category. Essential criteria include "
            "classic morphology on high-quality H&E scans, accessible immunohistochemistry, "
            "and fundamental clinical data. Desirable criteria add confirmation, prognosis, "
            "or predictive value and may include FISH, NGS, methylation profiles, and other "
            "molecular findings."
        ),
    },
    {
        "heading": "Essential and Desirable Criteria - CNS Glioma and Hematological Examples",
        "text": (
            "Diffuse glioma essential criteria include diffuse astrocytic infiltration with "
            "specific cytological atypia on digital scan. Desirable criteria include IDH1 or "
            "IDH2 mutation status and ATRX loss, enabling integrated diagnoses such as "
            "Astrocytoma, IDH-mutant, WHO grade 4. Polycythemia vera criteria combine "
            "hemoglobin or hematocrit, JAK2 mutation, and bone marrow morphology including "
            "panmyelosis and pleomorphic megakaryocytes. B-lymphoblastic leukemia/lymphoma "
            "essential criteria include more than 20 percent B-lymphoblasts or diffuse lymph "
            "node effacement; BCR-ABL1 is desirable because it guides tyrosine kinase therapy."
        ),
    },
    {
        "heading": "WHO Cytopathology Reporting Systems - Tiered Diagnostic Framework",
        "text": (
            "WHO, IAC, and IARC cytopathology reporting systems cover lung, lymph node, "
            "pancreaticobiliary, and soft tissue cytopathology. The core mechanism is a "
            "hierarchical multi-tier diagnostic categorization. Typical tiers include "
            "insufficient or non-diagnostic, benign or negative for malignancy, atypical, "
            "neoplasm low or high risk, suspicious for malignancy, and malignant. The tier "
            "forces a defined risk category and guides management and ancillary testing "
            "allocation on limited cytological material."
        ),
    },
    {
        "heading": "WHO Cytopathology Tiers - Criteria and Clinical Implications",
        "text": (
            "Tier 1 insufficient or non-diagnostic includes severe artifacts or insufficient "
            "well-preserved cells and usually requires repeat FNAB or rapid on-site evaluation. "
            "Tier 2 benign or negative includes unequivocal benign features and routine follow-up. "
            "Tier 3 atypical exceeds reactive change but is not definitive and triggers repeat "
            "sampling or molecular tests. Tier 4 neoplasm stratifies low or high risk and guides "
            "surgery or systemic therapy. Tier 5 suspicious for malignancy is strongly suggestive "
            "but incomplete and requires core biopsy or tumor board. Tier 6 malignant includes "
            "unequivocal cancer features and prompts oncologic staging."
        ),
    },
    {
        "heading": "ICD-O-4 Integration and Data Interoperability - Digital Pathology Standards",
        "text": (
            "ICD-O codes anatomical topography and histological morphology or behavior for "
            "cancer registries. ICD-O-4 adds a fifth alphanumeric digit to morphology codes "
            "to represent precise molecular subtypes. Structured diagnoses should map to "
            "ICD-O-4 and ICD-11 codes. DICOM ISO 12052 supports whole slide imaging exchange "
            "with clinical metadata. SNOMED CT with DICOM supports semantic interoperability. "
            "IHE PaLM integrates LIS, EHR, WSI scanners, and pathology viewers. ICC color "
            "management helps preserve stain consistency across scanners and displays."
        ),
    },
    {
        "heading": "Integrated Report - Synthesis of Morphology and Molecular Pathology",
        "text": (
            "Modern pathology reporting should synthesize digital morphology, IHC, and "
            "molecular studies into a single integrated diagnosis for precision oncology. "
            "AI pathology outputs should include classification label, Platt-scaled confidence, "
            "cytopathology tier, ICD-O-4 mapping, WHO Blue Books citations, uncertainty flag "
            "when confidence is low, and a mandatory medical disclaimer. Diagnostic AI outputs "
            "require validation with at least 95 percent concordance against expert review and "
            "major discrepancy rate below 4-7 percent."
        ),
    },
    {
        "heading": "AI and Computational Pathology - WHO Standards for Automated Analysis",
        "text": (
            "WHO Blue Books 5th edition standardizes mitotic count reporting as mitoses per "
            "square millimeter, replacing subjective high-power field counts. AI systems may "
            "scan gigapixel tissue, detect true mitotic figures, ignore artifacts, and compute "
            "density per mm2. AI can triage WSI scans, identify high-probability malignancy, "
            "and prioritize worklists. Cytopathology AI may screen cells and flag atypical or "
            "suspicious morphology. Clinical diagnostic algorithms require rigorous validation "
            "against diverse multi-institutional datasets."
        ),
    },
    {
        "heading": "Skin Lesion and Pathology Image Classification - WHO Framework Application",
        "text": (
            "Skin and pathology image classification should use WHO Blue Books essential "
            "criteria to drive labels and confidence thresholds. Essential morphology on "
            "digital H&E is primary. Confidence at or above 70 percent can support a specific "
            "WHO tumor type label when essential criteria are present. Confidence from 40-70 "
            "percent should report features consistent with a broader family or not otherwise "
            "specified category. Confidence below 40 percent should be insufficient or atypical. "
            "GradCAM regions in pathology should correlate with nuclear pleomorphism, mitotic "
            "figures, and tumor microenvironment architecture. Outputs should include ICD-O-4, "
            "WHO citations, disclaimer, and calibrated confidence band."
        ),
    },
]


def _tokens(text: str) -> list[str]:
    return [token for token in re.split(r"[^a-z0-9]+", text.lower()) if len(token) > 2]


def _embed(text: str, dimensions: int = EMBED_DIMENSIONS) -> list[float]:
    vector = [0.0] * dimensions
    for token in _tokens(text):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dimensions
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign
    norm = sum(value * value for value in vector) ** 0.5 or 1.0
    return [value / norm for value in vector]


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    words = text.split()
    chunks: list[str] = []
    step = max(size - overlap, 1)
    for index in range(0, len(words), step):
        chunks.append(" ".join(words[index : index + size]))
    return chunks


def build_corpus() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    doc_id = "WHO_PATHOLOGY_STD_2024"
    chunk_index = 0
    for section in WHO_PATH_SECTIONS:
        full_text = f"{section['heading']}. {section['text']}"
        for chunk in chunk_text(full_text):
            rows.append(
                {
                    "chunk_id": f"{doc_id}_chunk_{chunk_index:04d}",
                    "text": chunk,
                    "doc_id": doc_id,
                    "chunk_index": chunk_index,
                    "section_heading": section["heading"],
                    "page_number": chunk_index // 3 + 1,
                    "token_count": len(chunk.split()),
                    **SOURCE_META,
                }
            )
            chunk_index += 1
    return rows


def _collection():
    import chromadb
    from chromadb.config import Settings

    try:
        import chromadb.segment.impl.metadata.sqlite as chroma_sqlite
        import chromadb.segment.impl.vector.local_persistent_hnsw as chroma_hnsw

        original_decode_seq_id = chroma_sqlite._decode_seq_id
        original_load_hnsw_metadata = chroma_hnsw.PersistentData.load_from_file

        def decode_seq_id_compat(seq_id):
            if isinstance(seq_id, int):
                return seq_id
            return original_decode_seq_id(seq_id)

        def load_hnsw_metadata_compat(filename):
            data = original_load_hnsw_metadata(filename)
            if isinstance(data, dict):
                dimensionality = data.get("dimensionality")
                if dimensionality is None:
                    sqlite_path = Path(filename).resolve().parent.parent / "chroma.sqlite3"
                    import sqlite3

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

        chroma_sqlite._decode_seq_id = decode_seq_id_compat
        chroma_hnsw.PersistentData.load_from_file = staticmethod(load_hnsw_metadata_compat)
    except Exception:
        pass

    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(CHROMA_PATH),
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(name=COLLECTION, metadata={"hnsw:space": "cosine"})


def ingest_who_pathology() -> int:
    collection = _collection()
    corpus = build_corpus()
    existing_ids = set(collection.get(include=[])["ids"])
    pending = [row for row in corpus if row["chunk_id"] not in existing_ids]

    if not pending:
        print(f"WHO pathology standards already present. Collection count: {collection.count()}")
        return 0

    uploaded = 0
    for index in range(0, len(pending), UPLOAD_BATCH):
        batch = pending[index : index + UPLOAD_BATCH]
        texts = [row["text"] for row in batch]
        collection.upsert(
            ids=[row["chunk_id"] for row in batch],
            embeddings=[_embed(text) for text in texts],
            documents=texts,
            metadatas=[{key: value for key, value in row.items() if key not in {"chunk_id", "text"}} for row in batch],
        )
        uploaded += len(batch)
        print(f"Uploaded {uploaded}/{len(pending)} WHO pathology chunks")

    print(f"Done. Collection '{COLLECTION}' count: {collection.count()}")
    print("WHO pathology Blue Books, cytology tiers, WSI specs, and AI validation standards are ready for RAG.")
    return uploaded


if __name__ == "__main__":
    ingest_who_pathology()
