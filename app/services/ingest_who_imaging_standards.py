"""Ingest WHO imaging standards into the main VaidyaAI RAG collection."""

from __future__ import annotations

import hashlib
import os
import re
import sqlite3
from pathlib import Path
from typing import Any


CHROMA_PATH = Path(os.getenv("CHROMA_PATH", "data/chromadb"))
COLLECTION = os.getenv("CHROMA_COLLECTION", "medical_evidence_week1_clean")
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "100"))
UPLOAD_BATCH = int(os.getenv("RAG_UPLOAD_BATCH_SIZE", "32"))
EMBED_DIMENSIONS = int(os.getenv("RAG_EMBED_DIMENSIONS", "96"))

SOURCE_META = {
    "source_name": "WHO Diagnostic Imaging Standards",
    "source_type": "clinical_guideline",
    "source_file": "WHO_Imaging_Standards.txt",
    "title": "Procedural Frameworks for Interpretation and Management of Diagnostic Imaging",
    "url": "https://www.who.int/publications/diagnostic-imaging",
    "published_date": "2024-01-01",
    "condition": "radiology,imaging,DICOM,GradCAM,chest_xray,CT,tuberculosis,COVID-19",
    "topic": "imaging_standards",
}

WHO_SECTIONS = [
    {
        "heading": "I. Foundational Mandate - WHO Imaging Framework",
        "text": (
            "The World Health Organization Manual of Diagnostic Imaging governs evidence-based "
            "policy for medical imaging globally. Approximately 70-80% of clinically relevant "
            "diagnostic questions can be resolved using fundamental imaging modalities: "
            "radiography and ultrasonography. Governance requires adherence to radiation safety "
            "under Diagnostic Radiological Physicist and Radiation Safety Officer oversight. "
            "The ALARA principle applies to all ionizing radiation procedures."
        ),
    },
    {
        "heading": "II. Technical Acquisition Criteria - Image Quality Standards",
        "text": (
            "WHO specifies minimum technical parameters for diagnostic radiographs. Generator "
            "rating: 300 mA at 125 kVp minimum. Focal spot size: maximum 2 mm. Pixel pitch: "
            "maximum 200 micrometers. Bit depth: 10-bit minimum. Spatial resolution: 2.5 line "
            "pairs per mm. For chest radiography, exposure time must be <=50 ms. Sensor area "
            "for chest screening must be no less than 1505 square cm with minimum width 35 cm "
            "to capture the full thoracic cavity during full inspiration. High-density "
            "structures attenuate the X-ray beam and appear white; low-density structures like "
            "air appear black."
        ),
    },
    {
        "heading": "II. Portable and Mobile Imaging - Technical Criteria",
        "text": (
            "Portable digital radiography systems for emergency departments and ICUs must "
            "function as standalone units for acquisition, review, display, and storage. "
            "Weight: preferably 5-10 kg. Dimensions: approximately 35-45 cm. Battery: minimum "
            "1-2 hours normal operation. Mobile images use anteroposterior projection rather "
            "than standard posteroanterior view, causing inherent magnification of mediastinal "
            "structures. Interpreting clinicians must account for this artifact when assessing "
            "cardiac silhouette size."
        ),
    },
    {
        "heading": "III. Patient Positioning Criteria - Anatomical Inclusion",
        "text": (
            "For a chest radiograph to be considered diagnostic, it must include the "
            "supraclavicular region, lateral margins of the chest wall, and both "
            "hemidiaphragms in their entirety. In musculoskeletal imaging, at least two "
            "radiographs at right angles are required to localize abnormalities in three "
            "dimensions. Maximum inspiration is required for thoracic imaging to allow "
            "expansion of the thoracic cavity and lowering of the diaphragm, increasing "
            "diagnostic acuity."
        ),
    },
    {
        "heading": "IV. Systematic Interpretation Checklist - Pre-Clinical Verification",
        "text": (
            "WHO systematic interpretation checklist before clinical assessment: verify full "
            "patient name, birthdate, and institutional ID to prevent wrong-person procedures. "
            "Check study date and compare with previous exams to determine whether findings are "
            "new or chronic. Anatomical left or right markers must be present. Patient position "
            "and projection such as PA, AP, orthostatism, or decubitus must be noted. Maximum "
            "inspiration is required for thoracic imaging. Failure to verify these steps is a "
            "leading cause of diagnostic error."
        ),
    },
    {
        "heading": "IV. Region-Specific Interpretation - Systematic Radiograph Reading",
        "text": (
            "Musculoskeletal interpretation assesses cortical and trabecular pattern for "
            "disruption and evaluates soft tissues for displacement or obliteration of fat "
            "planes. Chest and pulmonary interpretation should proceed through support devices, "
            "chest wall, heart and mediastinum, hila, lungs, airways, pleura, and diaphragm. "
            "Evaluate lung volumes, mediastinal width, and lung patterns. Key findings include "
            "pneumonia, pneumothorax, tumors, and heart failure. Abdominal interpretation "
            "assesses gas patterns, organ borders, and opacity changes."
        ),
    },
    {
        "heading": "V. CT Interpretation Criteria - Advanced Imaging Standards",
        "text": (
            "Computed tomography offers higher diagnostic accuracy than conventional radiography "
            "for complex pathologies. CT is indicated for characterization of bone tumors, joint "
            "morphology, complex fractures, and cases where X-ray cannot resolve the clinical "
            "question. Technical parameters include 256-slice CT reconstruction slice thickness "
            "0.9 mm and tube voltage 100-120 kVp adjusted for BMI. CT interpretation criteria "
            "include patchy consolidation greater than 1 cm versus multifocal consolidation of "
            "three or more lesions under 1 cm. Ground-glass opacity is common but nonspecific "
            "in viral infections and inflammatory conditions. Distribution should be classified "
            "by lung zone and type."
        ),
    },
    {
        "heading": "VI. DICOM Standards - Digital Imaging and Communications",
        "text": (
            "DICOM is the global clinical messaging syntax ISO 12052 for exchanging, storing, "
            "and managing medical images. DICOM file structure includes information objects, "
            "service classes such as C-STORE and C-FIND, tags and attributes for name, ID, and "
            "exposure settings, and pixel data supporting JPEG, JPEG 2000, and RLE compression. "
            "EHR systems must link to full-resolution DICOM because low-resolution PDF "
            "encapsulation compromises clinical interpretation."
        ),
    },
    {
        "heading": "VII. Teleradiology - Display and Viewing Standards",
        "text": (
            "Teleradiology diagnostic display minimum specifications include display resolution "
            "of at least 3 MP for general radiography and CT, luminance at least 350 cd per "
            "square meter, ambient lighting 25-75 lux at the display surface, contrast ratio "
            "at least 350:1 measured using AAPM TG18 patterns, and network bandwidth at least "
            "100 Mbps sustained. Reports must include formal specialist report or documented "
            "oral communication, retention per local regulations, disclaimer statements, and "
            "interaction logs for quality assurance and reimbursement."
        ),
    },
    {
        "heading": "VIII. Quality Assurance - WHO QA and Radiation Safety",
        "text": (
            "WHO quality assurance ensures quality control activities maintain high-quality "
            "radiographs at minimum radiation exposure. Primary monitor QA includes bi-weekly "
            "cleaning and evaluation using SMPTE test patterns. Grayscale from 0-100% must be "
            "differentiable and contrast resolution sharp in all corners. Reject film analysis "
            "identifies training gaps and reduces repeat exposures. Radiation safety follows "
            "ALARA and ALARP principles. Operators should stand behind barriers during exposure "
            "and wear dosimetry."
        ),
    },
    {
        "heading": "IX. Tuberculosis Screening - CAD and CXR Guidelines",
        "text": (
            "WHO recommends chest radiography as a fundamental tool for early TB detection as "
            "part of the End TB Strategy. WHO endorses Computer-Aided Detection software as an "
            "alternative to human interpretation of digital CXR for TB screening in individuals "
            "aged 15 and older. CAD diagnostic accuracy is similar to human readers but requires "
            "calibration for the specific setting and prevalence. CXR is the most sensitive "
            "tool for identifying survey participants with high probability of TB."
        ),
    },
    {
        "heading": "IX. COVID-19 Imaging - WHO Conditional Recommendations",
        "text": (
            "WHO conditional recommendations for imaging in suspected COVID-19: asymptomatic "
            "contacts should not use imaging because virological testing is primary. Symptomatic "
            "patients with testing available should not use imaging for initial workup. "
            "Symptomatic patients with delayed testing may use chest imaging for rapid triage. "
            "Hospitalized patients may use X-ray or CT to detect complications including "
            "pulmonary artery thrombosis. COVID-19 imaging findings overlap with influenza, "
            "SARS, and MERS and must be interpreted alongside clinical and laboratory data."
        ),
    },
    {
        "heading": "X. Standardized Reporting - Radiologic Report Criteria",
        "text": (
            "The radiologic report is the essential tool radiologists provide to care providers. "
            "Reports must be uniform, comprehensive, and machine-readable. Standardized "
            "terminology such as RadLex facilitates communication and data analysis. Risk "
            "assessment systems such as BI-RADS and LI-RADS categorize disease risk using "
            "structured terms. Coded information such as RadElements allows automated "
            "extraction of dose indices. Reports require a definitive signature and unique "
            "electronic identifier."
        ),
    },
    {
        "heading": "GradCAM and AI Explainability - WHO Imaging Interpretation Context",
        "text": (
            "WHO standards require AI-assisted image analysis outputs to be interpreted "
            "alongside the systematic checklist: verify patient ID, study date, anatomical "
            "markers, and projection type before assessing AI-generated findings. GradCAM "
            "heatmaps highlight regions of highest model attention corresponding to anatomical "
            "regions that influenced classification. Red or orange indicates strongest "
            "activation. Activation maps must be correlated with clinical findings using "
            "systematic interpretation of chest wall, mediastinum, hila, lung zones, and "
            "pleura. All AI outputs are AI-assisted analysis only and not a medical diagnosis."
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


def _collection_embedding_dimensions(default: int = EMBED_DIMENSIONS) -> int:
    sqlite_path = CHROMA_PATH / "chroma.sqlite3"
    if not sqlite_path.exists():
        return default
    try:
        with sqlite3.connect(sqlite_path) as conn:
            row = conn.execute(
                "SELECT vector FROM embeddings_queue WHERE vector IS NOT NULL LIMIT 1"
            ).fetchone()
        if row and row[0]:
            return len(row[0]) // 4
    except Exception:
        pass
    return default


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    words = text.split()
    chunks: list[str] = []
    step = max(size - overlap, 1)
    for index in range(0, len(words), step):
        chunks.append(" ".join(words[index : index + size]))
    return chunks


def build_corpus() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    doc_id = "WHO_IMAGING_STD_2024"
    chunk_index = 0
    for section in WHO_SECTIONS:
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
                    max_seq_id=data.get("max_seq_id") or 0,
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


def ingest_who_standards() -> int:
    collection = _collection()
    dimensions = _collection_embedding_dimensions()
    corpus = build_corpus()
    existing_ids = set(collection.get(include=[])["ids"])
    pending = [row for row in corpus if row["chunk_id"] not in existing_ids]

    if not pending:
        print(f"WHO imaging standards already present. Collection count: {collection.count()}")
        return 0

    uploaded = 0
    for index in range(0, len(pending), UPLOAD_BATCH):
        batch = pending[index : index + UPLOAD_BATCH]
        texts = [row["text"] for row in batch]
        collection.upsert(
            ids=[row["chunk_id"] for row in batch],
            embeddings=[_embed(text, dimensions) for text in texts],
            documents=texts,
            metadatas=[{key: value for key, value in row.items() if key not in {"chunk_id", "text"}} for row in batch],
        )
        uploaded += len(batch)
        print(f"Uploaded {uploaded}/{len(pending)} WHO imaging chunks")

    print(f"Done. Collection '{COLLECTION}' count: {collection.count()}")
    return uploaded


if __name__ == "__main__":
    ingest_who_standards()
