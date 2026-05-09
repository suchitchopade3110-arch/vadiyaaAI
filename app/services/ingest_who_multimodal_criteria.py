"""Ingest WHO multimodal diagnostic criteria into the main RAG collection.

This complements the imaging and pathology standards ingesters with modality-
specific retrieval chunks for MRI brain tumors, CT, chest radiography, skin, and
digital pathology outputs.
"""

from __future__ import annotations

import hashlib
import os
import re
import sqlite3
from pathlib import Path
from typing import Any


CHROMA_PATH = Path(os.getenv("CHROMA_PATH", "data/chromadb"))
COLLECTION = os.getenv("CHROMA_COLLECTION", "medical_evidence_week1_clean")
UPLOAD_BATCH = int(os.getenv("RAG_UPLOAD_BATCH_SIZE", "32"))
EMBED_DIMENSIONS = int(os.getenv("RAG_EMBED_DIMENSIONS", "96"))

SOURCE_META = {
    "source_name": "WHO Multimodal Diagnostic Criteria",
    "source_type": "clinical_guideline",
    "source_file": "WHO_Multimodal_Criteria_Inline.txt",
    "title": "WHO Criteria: CNS5, Fleischner, Cytopathology, MRI, CT, Skin",
    "url": "https://www.iarc.who.int/cards_page/who-classification-of-tumours/",
    "published_date": "2024-01-01",
}

WHO_MULTI_CHUNKS: list[tuple[str, str, str, str]] = [
    (
        "WHO_CNS5_001",
        "mri_brain_tumor",
        "WHO CNS5 Glioblastoma Definition IDH-wildtype",
        "WHO CNS5 2021 defines adult glioblastoma as IDH-wildtype diffuse astrocytic glioma. "
        "Even low-grade morphology without necrosis or microvascular proliferation is classified "
        "as Glioblastoma IDH-wildtype CNS WHO Grade 4 when EGFR amplification, combined whole "
        "chromosome 7 gain and chromosome 10 loss (+7/-10), or TERT promoter mutation is present. "
        "Arabic grade numerals 1-4 are used. IDH-wildtype gliomas with these molecular markers "
        "behave as aggressively as histologically classic necrotic glioblastomas.",
    ),
    (
        "WHO_CNS5_002",
        "mri_brain_tumor",
        "WHO CNS5 Astrocytoma Oligodendroglioma Classification",
        "Astrocytoma IDH-mutant requires IDH1 or IDH2 mutation without 1p/19q codeletion and "
        "is graded 2, 3, or 4. IDH-mutant astrocytoma with necrosis is Astrocytoma IDH-mutant "
        "Grade 4, not secondary glioblastoma. Oligodendroglioma requires both IDH mutation and "
        "1p/19q codeletion, grade 2 or 3. Pediatric diffuse gliomas are classified separately, "
        "including MAPK-altered low-grade gliomas and high-grade H3 K27-altered or H3 G34-mutant tumors.",
    ),
    (
        "WHO_CNS5_003",
        "mri_brain_tumor",
        "WHO CNS5 Meningioma Molecular Grading",
        "Meningioma includes 15 morphological subtypes. TERT promoter mutation or homozygous "
        "CDKN2A/CDKN2B deletion confers Grade 3 behavior regardless of benign histology. "
        "Rhabdoid meningioma may show BAP1 mutation; clear cell variants may show SMARCE1 "
        "mutation. Convexity and spinal tumors often involve NF2 and chromosome 22q deletion. "
        "Anterior skull base tumors may involve AKT1, TRAF7, SMO, or PIK3CA. Grade 2 criteria "
        "include at least 4 mitoses per 10 HPF or brain invasion.",
    ),
    (
        "WHO_CNS5_004",
        "mri_brain_tumor",
        "WHO PitNET Pituitary Neuroendocrine Tumor Classification",
        "Pituitary adenoma terminology is replaced by Pituitary Neuroendocrine Tumor or PitNET. "
        "Lineage classification uses transcription factor immunohistochemistry: PIT1 lineage "
        "for somatotroph, lactotroph, and thyrotroph tumors; TPIT for corticotroph tumors; SF1 "
        "for gonadotroph tumors. Null cell tumors are rare. High-risk histotypes include sparsely "
        "granulated somatotroph, silent corticotroph, and Crooke cell PitNET. MRI must distinguish "
        "neoplasm from mimics such as ADEM, tumefactive MS, subacute infarct, and pyogenic abscess.",
    ),
    (
        "FLEISCH_001",
        "chest_xray",
        "Fleischner Atelectasis Consolidation Infiltration Definitions",
        "Atelectasis is reduced lung lobe or segment volume with opacity and geometric volume-loss "
        "signs such as mediastinal shift toward the opacity, hilar displacement, fissure shift, or "
        "hemidiaphragm elevation. Consolidation is alveolar filling by fluid, pus, blood, cells, or "
        "protein without volume loss; air bronchograms are typical. Infiltration is an imprecise "
        "term discouraged by Fleischner, often used in AI datasets for poorly defined opacity lacking "
        "the density of true consolidation.",
    ),
    (
        "FLEISCH_002",
        "chest_xray",
        "Fleischner Nodule Mass Pneumonia Definitions",
        "Pulmonary nodule is a discrete rounded opacity no greater than 3 cm without pleural attachment "
        "or associated lymphadenopathy. Mass is a solid pulmonary opacity greater than 3 cm and carries "
        "high baseline malignancy probability requiring advanced imaging or biopsy. Pneumonia is a "
        "clinical diagnosis supported radiographically by focal segmental or lobar consolidation; bacterial "
        "pneumonia often shows dense alveolar consolidation with air bronchograms, while viral pneumonia "
        "may show patchy bilateral interstitial opacities.",
    ),
    (
        "FLEISCH_003",
        "chest_xray",
        "Fleischner Pleural Effusion Pneumothorax Emphysema Fibrosis",
        "Pleural effusion is abnormal pleural fluid, visible on frontal chest radiograph when volume "
        "usually exceeds about 175 mL with costophrenic blunting and meniscus. Pneumothorax is air in "
        "the pleural cavity, seen as a visceral pleural line with absent peripheral lung markings; tension "
        "causes mediastinal shift away from the affected side. Emphysema shows hyperlucency, flattened "
        "diaphragms, and increased AP diameter. Fibrosis shows reticular opacities, volume loss, honeycombing, "
        "and architectural distortion.",
    ),
    (
        "FLEISCH_004",
        "chest_xray",
        "Fleischner Cardiomegaly Edema Hernia Definitions",
        "Cardiomegaly is cardiothoracic ratio above 50 percent on a standard well-inspired PA radiograph. "
        "Pulmonary edema from left ventricular failure progresses through cephalization, interstitial edema "
        "with Kerley B lines, and alveolar edema with bilateral bat-wing perihilar consolidation. Hiatal "
        "hernia appears as a retrocardiac soft tissue mass often with an air-fluid level behind the heart.",
    ),
    (
        "FLEISCH_005",
        "CT",
        "Fleischner CT Nodule Attenuation Classification",
        "CT nodule attenuation classification includes solid nodules that obscure underlying lung structures, "
        "ground-glass nodules with hazy increased attenuation while vessels remain visible, and part-solid "
        "nodules with both solid and ground-glass components. Part-solid nodules have the highest statistical "
        "probability of primary pulmonary malignancy, especially adenocarcinoma. Fleischner follow-up depends "
        "on attenuation, size, and clinical risk factors. CT detects small pleural fluid volumes and tree-in-bud "
        "pattern suggests active endobronchial spread such as tuberculosis.",
    ),
    (
        "WHO_CYTO_001",
        "pathology",
        "WHO Pancreaticobiliary Cytopathology 7-Tier System",
        "WHO pancreaticobiliary cytopathology uses seven tiers: insufficient or inadequate, benign or negative, "
        "atypical, pancreaticobiliary neoplasm low-risk, pancreaticobiliary neoplasm high-risk, suspicious "
        "for malignancy, and malignant. Low-risk includes premalignant noninvasive intraductal lesions such "
        "as low-grade IPMN. High-risk includes severe high-grade atypia without definitive invasion. Malignant "
        "requires unequivocal cancer features. PanNETs and solid pseudopapillary neoplasms are categorized "
        "as malignant in this framework.",
    ),
    (
        "WHO_CYTO_002",
        "pathology",
        "WHO Lung Soft Tissue Cytopathology Tiers",
        "WHO lung cytopathology uses insufficient, benign, atypical, suspicious for malignancy, and malignant "
        "tiers. Atypical and suspicious categories should trigger ancillary immunohistochemistry and molecular "
        "studies, especially for non-small cell lung carcinoma. WHO soft tissue cytopathology includes an "
        "additional category for neoplasm of uncertain malignant potential. AI systems must not force binary "
        "benign or malignant output on paucicellular smears when intermediate WHO categories are appropriate.",
    ),
    (
        "WHO_CYTO_003",
        "pathology",
        "WHO Colorectal Cancer Grading Digital Pathology",
        "WHO colorectal adenocarcinoma grading uses gland formation: Grade 1 well differentiated is more "
        "than 95 percent gland formation, Grade 2 is 50-95 percent, Grade 3 is less than 50 percent, and "
        "Grade 4 lacks gland formation or mucin production. This scheme is not applied to medullary carcinoma "
        "because behavior differs from morphology. WHO 5th edition requires mitotic counts in SI units as "
        "mitoses per mm2 rather than subjective high-power fields.",
    ),
    (
        "WHO_CYTO_004",
        "pathology",
        "WHO Neuroendocrine Neoplasm Grading Ki-67",
        "WHO gastrointestinal neuroendocrine tumors are graded by proliferation. NET Grade 1 has Ki-67 under "
        "3 percent and mitotic rate under 2 per 2 mm2. NET Grade 2 has Ki-67 3-20 percent or mitotic rate "
        "2-20 per 2 mm2. NET Grade 3 has Ki-67 greater than 20 percent or mitotic rate greater than 20 per "
        "2 mm2. Poorly differentiated neuroendocrine carcinomas are classified as small-cell or large-cell "
        "type regardless of Ki-67.",
    ),
    (
        "WHO_SKIN_001",
        "skin",
        "WHO Skin Tumor 5th Edition Melanocytic Classification",
        "WHO 2023 skin tumor classification moves beyond a strict benign-malignant binary. Melanocytoma is "
        "an intermediate state between nevus and melanoma. Common nevus often has a single MAPK activating "
        "mutation such as BRAF V600E with oncogene-induced senescence. Melanocytoma may escape senescence "
        "through second hits such as BAP1, WNT pathway alteration, or PRKAR1A inactivation. Low-CSD melanoma "
        "often involves intermittently sun-exposed skin and BRAF V600E, while high-CSD melanoma shows high "
        "UV mutational burden.",
    ),
    (
        "WHO_SKIN_002",
        "skin",
        "WHO BCC SCC Keratoacanthoma Classification",
        "Basal cell carcinoma is stratified into low-risk superficial, nodular, and fibroepithelial variants "
        "and high-risk infiltrative, basosquamous, or sarcomatoid variants. Squamous cell carcinoma often "
        "shows null or diffusely mutant p53, irregular Ki-67 beyond peripheral basal cells, and 9p21 deletion. "
        "Keratoacanthoma shows concordant graded peripheral Ki-67 and p53 with self-resolving potential. "
        "Ambiguous lesions may be reported as squamoproliferative tumor of uncertain malignant potential.",
    ),
    (
        "WHO_SKIN_003",
        "skin",
        "ABCDE Criteria 7-Point Checklist Dermoscopy Skin Cancer Screening",
        "ABCDE criteria include asymmetry, border irregularity, color variation, diameter greater than 6 mm, "
        "and evolving size, shape, color, elevation, bleeding, or symptoms. The weighted 7-point checklist "
        "assigns 2 points each for change in size, irregular shape or border, and irregular color; 1 point "
        "each for diameter at least 7 mm, inflammation, oozing or bleeding, and itch or pain. Score at least "
        "3 is suspicious for melanoma and requires urgent referral.",
    ),
    (
        "WHO_SKIN_004",
        "skin",
        "Dermoscopy Patterns Atypical Network Blue White Veil",
        "Dermoscopy patterns for melanoma include atypical pigment network with irregular thickened lines, "
        "atypical dots and globules of irregular size and color, radial streaks and pseudopods indicating "
        "peripheral growth, blue-white veil, and negative network. AI skin models trained mostly on Fitzpatrick "
        "I-III may underperform on darker skin types IV-VI. Acral lentiginous melanoma on palms, soles, and "
        "nails is a non-UV subtype that can be missed by standard public health heuristics.",
    ),
    (
        "WHO_MRI_001",
        "mri",
        "WHO MRI Stroke DWI FLAIR Mismatch Thrombolysis Criteria",
        "DWI becomes bright within minutes of ischemic stroke due to cytotoxic edema and restricted water "
        "diffusion. FLAIR hyperintensity takes several hours to develop. DWI-FLAIR mismatch, with positive "
        "DWI and negative FLAIR, suggests stroke within about 4.5 hours and supports thrombolysis decisions "
        "for wake-up or unknown-onset stroke. Extended windows may use CT or MRI core-perfusion mismatch "
        "criteria based on ischemic core volume, mismatch volume, and mismatch ratio.",
    ),
    (
        "WHO_MRI_002",
        "mri",
        "BI-RADS MRI Breast Cancer Screening Criteria",
        "BI-RADS MRI categories include 1 negative, 2 benign, 3 probably benign with 2 percent or lower "
        "malignancy risk and 6-month follow-up, 4 suspicious with biopsy required, 5 highly suggestive with "
        "at least 95 percent malignancy risk, and 6 biopsy-proven malignancy. Dynamic contrast-enhanced MRI "
        "malignancy often shows rapid wash-in and wash-out. MRI has high sensitivity for invasive carcinoma "
        "but requires tissue diagnosis before operative plan changes.",
    ),
    (
        "WHO_MRI_003",
        "mri",
        "McDonald Criteria Multiple Sclerosis MRI Diagnosis",
        "McDonald criteria for multiple sclerosis require dissemination in space and time. Dissemination in "
        "space means lesions in at least 2 of 4 CNS locations: periventricular, cortical or juxtacortical, "
        "infratentorial, or spinal cord. Dissemination in time may be simultaneous enhancing and non-enhancing "
        "lesions or a new T2 or gadolinium-enhancing lesion on follow-up. MRI also helps dementia and Parkinson "
        "differential diagnosis through atrophy and substantia nigra markers.",
    ),
    (
        "WHO_CT_001",
        "CT",
        "WHO CT COVID-19 Diagnostic Features Sensitivity Specificity",
        "COVID-19 CT patterns include multifocal bilateral peripheral or posterior ground-glass opacity, "
        "rounded ground-glass opacity, crazy-paving pattern, reversed halo sign, and bilateral involvement. "
        "Reversed halo is relatively specific but less sensitive. CT should not be routine population screening "
        "in high-prevalence settings and is most useful when testing is unavailable, delayed, or discordant "
        "with persistent clinical suspicion.",
    ),
    (
        "WHO_CT_002",
        "CT",
        "WHO CT Technical Specifications 64-Slice Procurement",
        "WHO and IAEA CT deployment specifications include at least 64 detector slices per 360-degree rotation, "
        "gantry rotation speed no more than 0.5 seconds, slice thickness range about 0.625-10 mm, generator "
        "output at least 20 kW, tube voltage up to 120 kVp, axial and spiral scanning, and retrospective raw "
        "data reconstruction. CTDIvol equals weighted CTDI divided by pitch; DLP equals CTDIvol times scan length. "
        "Diagnostic reference levels should be tracked and audited.",
    ),
    (
        "WHO_CT_003",
        "CT",
        "WHO Neurocysticercosis CT Staging Management",
        "Neurocysticercosis CT staging distinguishes viable vesicular cysts with scolex, degenerating enhancing "
        "ring or nodular lesions with edema, non-viable calcified nodules, and extraparenchymal ventricular or "
        "subarachnoid disease. Viable disease may require antiparasitic drugs with corticosteroid cover. Calcified "
        "disease is managed with anti-epileptic therapy rather than antiparasitics. Extraparenchymal disease may "
        "need neurosurgical shunting for hydrocephalus.",
    ),
    (
        "WHO_CT_004",
        "CT",
        "WHO CT Radiation Safety ALARA Dose Justification Pediatric",
        "WHO Bonn Call for Action emphasizes justification, optimization, and audit for CT. Every CT should "
        "provide net clinical benefit. Awareness, appropriateness criteria, and audit reduce unjustified imaging. "
        "Pediatric CT requires child-specific protocols because children have higher radiation sensitivity and "
        "longer lifespan for stochastic risk expression. ALARA requires dose optimization and pediatric-specific "
        "technique rather than adult settings.",
    ),
    (
        "WHO_SKIN_SCREEN_001",
        "skin",
        "WHO Early Diagnosis vs Screening Skin Cancer Three Steps",
        "WHO distinguishes early diagnosis from population screening. Early diagnosis has three steps: awareness "
        "and access, clinical evaluation with diagnosis and staging, and access to treatment. Screening identifies "
        "unrecognized disease in asymptomatic groups. Routine whole-body screening in asymptomatic adults has "
        "insufficient evidence, but targeted annual examination is appropriate for high-risk people with prior "
        "melanoma, dysplastic nevi, many nevi, immunosuppression, CDKN2A carrier status, blistering sunburns, "
        "or indoor tanning exposure.",
    ),
    (
        "WHO_SKIN_SCREEN_002",
        "skin",
        "WHO Augmented Intelligence Skin Cancer AI Performance",
        "WHO augmented intelligence in dermatology is intended to support rather than replace clinicians. "
        "AI assistance can improve sensitivity and specificity for skin cancer triage and reduce unnecessary "
        "benign lesion excision, but output remains AI-assisted analysis only and not a medical diagnosis. "
        "Teledermatology performance depends strongly on image quality; self-captured images may have lower "
        "concordance than clinician-acquired images. Fairness requires diverse training data and monitoring.",
    ),
    (
        "WHO_WSI_001",
        "pathology",
        "WHO WSI Technical Specifications Scanner Resolution Z-Stack",
        "Diagnostic-grade whole slide imaging requires high optical resolution, often 40x and 80x objective "
        "capabilities, about 0.25 um per pixel at 40x and 0.13 um per pixel at 80x, z-stacking for thick "
        "cytology smears or frozen sections, rapid scanning, barcode recognition, calibrated high-luminance "
        "diagnostic displays, adequate workstation RAM and SSD storage, and validation with at least 60 cases, "
        "2-week washout, at least 95 percent concordance, and low major discrepancy rate.",
    ),
    (
        "WHO_WSI_002",
        "pathology",
        "WHO AI Ethics Pathology Good Machine Learning Practice",
        "WHO AI ethics for pathology requires protection of human autonomy, safety, transparency and explainability, "
        "responsibility and accountability, inclusiveness and equity, and sustainable responsive AI. Good machine "
        "learning practice includes project initialization, upstream workflow control such as staining standardization, "
        "downstream continuous monitoring, and long-term maintenance with periodic retraining. Pre-analytical variables "
        "such as fixation, microtomy, and stain formulation can cause algorithmic drift.",
    ),
]


def _tokens(text: str) -> list[str]:
    return [token for token in re.split(r"[^a-z0-9]+", text.lower()) if len(token) > 2]


def _embed(text: str, dimensions: int) -> list[float]:
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


def build_corpus() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, (chunk_id, topic, heading, text) in enumerate(WHO_MULTI_CHUNKS):
        rows.append(
            {
                "chunk_id": chunk_id,
                "text": text,
                "doc_id": "WHO_MULTIMODAL_CRITERIA_2024",
                "chunk_index": index,
                "section_heading": heading,
                "page_number": index // 3 + 1,
                "token_count": len(text.split()),
                "topic": topic,
                **SOURCE_META,
            }
        )
    return rows


def ingest_who_multimodal() -> int:
    collection = _collection()
    dimensions = _collection_embedding_dimensions()
    corpus = build_corpus()
    existing_ids = set(collection.get(include=[])["ids"])
    pending = [row for row in corpus if row["chunk_id"] not in existing_ids]

    print("=" * 60)
    print("VaidyaAI - WHO Multimodal RAG Ingest")
    print(f"Collection: {COLLECTION} at {CHROMA_PATH}")
    print(f"Embedding dimensions: {dimensions}")
    print(f"Chunks: {len(corpus)} | Already present: {len(corpus) - len(pending)} | Pending: {len(pending)}")
    print("=" * 60)

    if not pending:
        print(f"All WHO multimodal chunks already present. Collection count: {collection.count()}")
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
        print(f"Uploaded {uploaded}/{len(pending)} WHO multimodal chunks")

    print(f"Done. Collection '{COLLECTION}' count: {collection.count()}")
    print("Covered topics:")
    for topic in sorted({row["topic"] for row in corpus}):
        print(f"  {topic}: {sum(1 for row in corpus if row['topic'] == topic)} chunks")
    return uploaded


if __name__ == "__main__":
    ingest_who_multimodal()
