"""VaidyaAI - repair WHO imaging chunks in ChromaDB.

Deletes stale WHO imaging chunks and re-ingests them using the embedding
dimension of the target collection. Fresh 96-dim local KBs stay 96-dim; an
existing 1024-dim main medical_evidence collection stays 1024-dim because
ChromaDB cannot mix vector dimensions inside one collection.

Run from backend root:
    python tools/dev-scripts/fix_who_ingest_dimensions.py
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import sqlite3
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

CHROMA_PATH = Path(os.getenv("CHROMA_PATH", "data/chromadb"))
COLLECTION = os.getenv("CHROMA_COLLECTION", "medical_evidence_week1_clean")
REQUESTED_DIMENSIONS = int(os.getenv("WHO_IMAGE_EMBED_DIMENSIONS", "96"))
UPLOAD_BATCH = int(os.getenv("RAG_UPLOAD_BATCH_SIZE", "32"))


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
    (
        "WHO I Foundational Mandate",
        "The World Health Organization Manual of Diagnostic Imaging governs evidence-based "
        "policy for medical imaging globally. Approximately 70-80% of clinically relevant "
        "diagnostic questions can be resolved using radiography and ultrasonography. "
        "Governance requires adherence to radiation safety under Diagnostic Radiological "
        "Physicist and Radiation Safety Officer oversight. The ALARA principle applies to "
        "all ionizing radiation procedures.",
    ),
    (
        "WHO II Technical Acquisition Criteria",
        "WHO specifies minimum technical parameters for diagnostic radiographs. Generator "
        "rating 300 mA at 125 kVp minimum. Focal spot size maximum 2 mm. Pixel pitch "
        "maximum 200 micrometers. Bit depth 10-bit minimum. Spatial resolution 2.5 line "
        "pairs per mm. For chest radiography exposure time must be 50 ms or less. Sensor "
        "area for chest screening must be no less than 1505 square cm with minimum width "
        "35 cm to capture the full thoracic cavity during full inspiration.",
    ),
    (
        "WHO III Patient Positioning Anatomical Inclusion",
        "For a chest radiograph to be considered diagnostic it must include the "
        "supraclavicular region, lateral margins of the chest wall, and both "
        "hemidiaphragms in their entirety. In musculoskeletal imaging at least two "
        "radiographs at right angles AP and lateral are required to localize abnormalities "
        "in three dimensions. Maximum inspiration is required for thoracic imaging to allow "
        "expansion of the thoracic cavity and lowering of the diaphragm increasing "
        "diagnostic acuity.",
    ),
    (
        "WHO IV Systematic Interpretation Checklist",
        "WHO systematic interpretation checklist before clinical assessment. Patient "
        "Identification verify full name birthdate institutional ID prevents wrong-person "
        "procedures major source of accidental exposure. Study Date check date and compare "
        "with previous exams indispensable for determining if findings are new or chronic. "
        "Anatomical Markers Left L or Right R markers must be present note patient position "
        "and projection PA vs AP. Inspiration Quality maximum inspiration required for "
        "thoracic imaging. Failure to verify these steps is leading cause of diagnostic error.",
    ),
    (
        "WHO IV Chest Pulmonary Interpretation Criteria",
        "Chest pulmonary interpretation orderly assessment of support and monitoring devices "
        "chest wall heart and mediastinum hila lungs airways pleura diaphragm. Evaluate lung "
        "volumes mediastinal width lung patterns. Key clinical findings include pneumonia "
        "pneumothorax pulmonary nodule tumors heart failure emphysema COPD atelectasis "
        "pleural effusion fibrosis infiltration cardiomegaly pleural thickening.",
    ),
    (
        "WHO V CT Interpretation Criteria Advanced Imaging",
        "Computed tomography offers higher diagnostic accuracy than conventional radiography "
        "for complex pathologies. CT indicated for characterization of bone tumors joint "
        "morphology complex fractures. Technical parameters 256-slice CT reconstruction "
        "slice thickness 0.9 mm tube voltage 100-120 kVp adjusted for BMI. CT interpretation "
        "criteria patchy consolidation diameter greater than 1 cm versus multifocal "
        "consolidation three or more lesions less than 1 cm. Ground-glass opacity common "
        "non-specific finding in viral infections and inflammatory conditions.",
    ),
    (
        "WHO VI DICOM Standards Digital Imaging",
        "DICOM Digital Imaging and Communications in Medicine is the global clinical "
        "messaging syntax ISO 12052. DICOM file structure includes information objects "
        "service classes C-STORE C-FIND tags and attributes for name ID and exposure "
        "settings pixel data supporting JPEG JPEG 2000 and RLE compression. EHR must link "
        "to full-resolution DICOM images. DICOM Ontology enables machine-processable "
        "interoperable applications.",
    ),
    (
        "WHO VII Teleradiology Display Standards",
        "Teleradiology diagnostic display minimum specifications. Display resolution minimum "
        "3 MP 2048 by 1536 for general radiography and CT. Luminance minimum 350 cd per "
        "square meter. Ambient lighting 25 to 75 lux at display surface. Contrast ratio "
        "minimum 350 to 1 measured using AAPM TG18 patterns. Network bandwidth minimum "
        "100 Mbps sustained. Display diagonal approximately 80 percent of viewing distance.",
    ),
    (
        "WHO VIII Quality Assurance Radiation Safety",
        "WHO Quality Assurance ensures quality control activities maintain high-quality "
        "radiographs at minimum radiation exposure. Primary monitor QA clean bi-weekly "
        "evaluate using SMPTE test patterns. Radiation safety follows ALARA and ALARP "
        "principles. Dose limits whole-body limit for radiation workers typically 5000 "
        "mrem per year. Operator protection stand behind barriers during exposure wear dosimetry.",
    ),
    (
        "WHO IX Tuberculosis Screening CAD Guidelines",
        "WHO recommends chest radiography CXR as fundamental tool for early TB detection "
        "as part of the End TB Strategy. WHO endorses Computer-Aided Detection CAD software "
        "as alternative to human interpretation of digital CXR for TB screening in "
        "individuals aged 15 and older. CAD diagnostic accuracy is similar to human readers "
        "but requires calibration for the specific setting and prevalence.",
    ),
    (
        "WHO IX COVID-19 Imaging Conditional Recommendations",
        "WHO conditional recommendations for imaging in suspected COVID-19. Asymptomatic "
        "contacts do not use imaging virological testing is primary. Symptomatic with delayed "
        "testing use chest imaging for rapid triage. Hospitalized for monitoring use X-ray "
        "ward or CT ICU to detect complications including pulmonary artery thrombosis. "
        "COVID-19 imaging findings overlap with influenza SARS MERS.",
    ),
    (
        "WHO X Standardized Reporting Radiologic Report",
        "The radiologic report is the essential tool radiologists provide to care providers. "
        "Reports must be uniform comprehensive and machine-readable. Standardized terminology "
        "RadLex facilitates communication and data analysis. Risk assessment BI-RADS LI-RADS "
        "categorize disease risk using structured terms. Coded information RadElements allows "
        "automated extraction of dose indices.",
    ),
    (
        "GradCAM AI Explainability WHO Imaging Interpretation",
        "WHO standards require AI-assisted image analysis outputs interpreted alongside the "
        "systematic checklist. Verify patient ID study date anatomical markers and projection "
        "type before assessing AI-generated findings. GradCAM heatmaps highlight regions of "
        "highest model attention corresponding to anatomical regions that influenced "
        "classification. Red or orange indicates strongest activation. All AI outputs are "
        "AI-assisted analysis only NOT a medical diagnosis. Confidence scores follow "
        "calibrated probability Platt scaling color-coded green above 70 percent amber "
        "40-70 percent red below 40 percent.",
    ),
    (
        "Pulmonary Nodule Chest Xray Detection Clinical Criteria",
        "Pulmonary nodule on chest xray small round opacity typically less than 3 cm "
        "diameter. Solitary pulmonary nodule requires follow-up CT for characterization. "
        "Detection confidence and classification probability guide clinical urgency. Nodules "
        "detected in right upper lobe or left upper lobe require higher clinical suspicion. "
        "WHO recommends systematic follow-up protocol based on nodule size density and "
        "patient risk factors including smoking history.",
    ),
    (
        "Emphysema COPD Chest Xray Radiographic Findings",
        "Emphysema and COPD on chest xray shows air trapping lung hyperinflation flattened "
        "diaphragms increased AP diameter barrel chest appearance. Bullae visible as "
        "hyperlucent areas without lung markings. Pulmonary vasculature may appear pruned "
        "peripherally. GradCAM activation typically bilateral lower zones for emphysema "
        "upper zones for bullous disease. Clinical correlation with spirometry FEV1 FVC "
        "ratio required for COPD diagnosis per WHO GOLD criteria.",
    ),
    (
        "Cardiomegaly Chest Xray Cardiothoracic Ratio Assessment",
        "Cardiomegaly on chest xray defined as cardiothoracic ratio greater than 0.5 on PA "
        "posteroanterior projection. On AP anteroposterior mobile projection cardiac "
        "silhouette appears magnified and ratio threshold adjusted upward. Enlarged heart "
        "may indicate heart failure cardiomyopathy valvular disease pericardial effusion. "
        "Cardiothoracic ratio measured as maximum cardiac width divided by maximum thoracic "
        "width at inner rib margins. Clinical correlation with echocardiography recommended.",
    ),
    (
        "Pleural Thickening Pleural Effusion Chest Xray Findings",
        "Pleural thickening on chest xray indicates past infection exposure asbestos or "
        "inflammatory disease. Blunting of costophrenic angle indicates pleural effusion. "
        "Pleural thickening differs from effusion in that thickening is fixed on decubitus "
        "views while effusion layers. Calcified pleural plaques suggest prior asbestos "
        "exposure or tuberculosis. Clinical follow-up with CT or ultrasound recommended "
        "for new or changing pleural abnormalities.",
    ),
    (
        "Fibrosis Infiltration Lung Findings Chest Xray",
        "Pulmonary fibrosis on chest xray shows bilateral reticular or reticulonodular "
        "pattern predominantly lower zones honeycombing in advanced cases traction "
        "bronchiectasis. Lung scarring from chronic disease infection or exposure. "
        "Infiltration non-specific opacity indicating infection or inflammation. Ground "
        "glass opacity hazy increased opacity without obscuring underlying vessels seen in "
        "viral pneumonia edema patterns. Associated conditions COVID-19 viral pneumonia "
        "pulmonary fibrosis sarcoidosis.",
    ),
]


def _embed(text: str, dimensions: int) -> list[float]:
    vector = [0.0] * dimensions
    for token in re.split(r"[^a-z0-9]+", text.lower()):
        if len(token) <= 2:
            continue
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dimensions
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign
    norm = sum(value * value for value in vector) ** 0.5 or 1.0
    return [value / norm for value in vector]


def _detect_collection_dimension(default: int = REQUESTED_DIMENSIONS) -> int:
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
    except Exception as exc:
        log.warning("Could not detect Chroma vector dimension: %s", exc)
    return default


def _get_collection():
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
    return client.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


def build_corpus() -> list[dict]:
    rows = []
    doc_id = "WHO_IMAGING_STD_2024"
    for index, (heading, text) in enumerate(WHO_SECTIONS):
        full = f"{heading}. {text}"
        rows.append(
            {
                "chunk_id": f"{doc_id}_chunk_{index:04d}",
                "text": full,
                "doc_id": doc_id,
                "chunk_index": index,
                "section_heading": heading,
                "token_count": len(full.split()),
                **SOURCE_META,
            }
        )
    return rows


def fix_who_ingest() -> None:
    collection_dimension = _detect_collection_dimension()
    dimensions = collection_dimension
    if collection_dimension != REQUESTED_DIMENSIONS:
        log.warning(
            "Requested %d-dim WHO embeddings, but collection '%s' is %d-dim. "
            "Using %d to avoid Chroma dimension mismatch. Rebuild the entire "
            "collection to move it to 96 dimensions.",
            REQUESTED_DIMENSIONS,
            COLLECTION,
            collection_dimension,
            dimensions,
        )

    collection = _get_collection()
    log.info("Collection '%s' current count: %d", COLLECTION, collection.count())

    corpus = build_corpus()
    who_ids = [row["chunk_id"] for row in corpus]
    existing = set(collection.get(include=[])["ids"])
    to_delete = [chunk_id for chunk_id in who_ids if chunk_id in existing]
    if to_delete:
        collection.delete(ids=to_delete)
        log.info("Deleted %d stale WHO chunks", len(to_delete))
    else:
        log.info("No existing WHO chunks to delete")

    log.info("Re-ingesting %d WHO chunks at %d dimensions...", len(corpus), dimensions)
    for index in range(0, len(corpus), UPLOAD_BATCH):
        batch = corpus[index : index + UPLOAD_BATCH]
        texts = [row["text"] for row in batch]
        collection.upsert(
            ids=[row["chunk_id"] for row in batch],
            embeddings=[_embed(text, dimensions) for text in texts],
            documents=texts,
            metadatas=[
                {key: value for key, value in row.items() if key not in {"chunk_id", "text"}}
                for row in batch
            ],
        )
        log.info("  Upserted %d/%d chunks", min(index + UPLOAD_BATCH, len(corpus)), len(corpus))

    log.info("Collection count after: %d", collection.count())
    log.info("Verifying - querying 'pulmonary nodule chest xray'...")
    results = collection.query(
        query_embeddings=[_embed("pulmonary nodule chest xray detection follow-up CT", dimensions)],
        n_results=3,
        where={"topic": "imaging_standards"},
        include=["documents", "metadatas", "distances"],
    )
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]
    if not docs:
        direct = collection.get(where={"topic": "imaging_standards"}, include=["documents", "metadatas"], limit=3)
        docs = direct.get("documents", [])
        metas = direct.get("metadatas", [])
        if not docs:
            log.error("Still 0 WHO imaging rows - check CHROMA_PATH")
            return
        log.warning("Vector query returned 0, but metadata lookup found WHO imaging rows.")
        for doc, meta in zip(docs, metas):
            log.info("  [metadata] %s - %s...", meta.get("source_name", "?"), doc[:70])
        return
    for doc, meta, distance in zip(docs, metas, distances):
        log.info(
            "  [%.3f] %s - %s...",
            round(1 - float(distance), 3),
            meta.get("source_name", "?"),
            doc[:70],
        )
    log.info("Fix complete. WHO imaging evidence is queryable for image analysis citations.")


if __name__ == "__main__":
    fix_who_ingest()
