<p align="center">
  <img src="docs/architecture/vaidyaai-banner.png" alt="VaidyaAI Banner" width="600"/>
</p>

<h1 align="center">🏥 VaidyaAI — Medical Intelligence Platform</h1>

<p align="center">
  <em>A six-layer AI pipeline for lab report analysis, medical image diagnosis, and clinical claim verification.</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/Celery-5.4-37814A?logo=celery&logoColor=white" />
  <img src="https://img.shields.io/badge/PostgreSQL-15-336791?logo=postgresql&logoColor=white" />
  <img src="https://img.shields.io/badge/Redis-7-DC382D?logo=redis&logoColor=white" />
  <img src="https://img.shields.io/badge/ChromaDB-0.5-FF6F00" />
  <img src="https://img.shields.io/badge/LLaMA--3-Groq-EE4C2C" />
  <img src="https://img.shields.io/badge/React-JSX-61DAFB?logo=react&logoColor=white" />
  <img src="https://img.shields.io/badge/SIH%202025-Team%20Straw%20Hats-gold" />
</p>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Getting Started](#-getting-started)
- [API Endpoints](#-api-endpoints)
- [Working Examples](#-working-examples)
- [Project Structure](#-project-structure)
- [Team](#-team)
- [License](#-license)

---

## 🧠 Overview

**VaidyaAI** is a comprehensive medical intelligence platform built for **Smart India Hackathon 2025** (SIH 2025). It processes lab reports, medical images (X-rays, MRI, CT, Skin, Pathology), and clinical claims through a multi-stage AI pipeline that delivers:

- ✅ **Verdicts** — Verified / Refuted / Uncertain
- 📊 **Confidence Scores** — Platt-scaled 0–100
- ⚠️ **Uncertainty Flags** — when confidence drops below threshold
- 🔍 **Anomaly Detection** — flagging out-of-range values
- 📖 **Citations** — PubMed, WHO standards, clinical guidelines
- ⚕️ **Medical Disclaimers** — on every response

> 🚨 **Disclaimer:** VaidyaAI is a clinical decision-support tool. It does NOT replace professional medical advice.

---

## 🏗️ Architecture

<p align="center">
  <img src="docs/architecture/Phase_2_Architecture.png" alt="VaidyaAI Architecture" width="900"/>
</p>

VaidyaAI follows a **six-layer pipeline architecture**:

```
┌──────────────────────────────────────────────────────────────────┐
│  Layer 1 — Ingestion & Preprocessing                             │
│  OCR (PaddleOCR/Tesseract) · DICOM Parser · Image Normalization  │
├──────────────────────────────────────────────────────────────────┤
│  Layer 2 — Entity Extraction (NER)                               │
│  ClinicalBERT → conditions, meds, lab values, doses              │
├──────────────────────────────────────────────────────────────────┤
│  Layer 3 — Classification & Prediction                           │
│  CheXNet (X-ray) · ViT (MRI/Skin) · Swin (Pathology) · XGBoost │
├──────────────────────────────────────────────────────────────────┤
│  Layer 4 — RAG Retrieval                                         │
│  BioGPT Embeddings → ChromaDB → WHO/PubMed/StatPearls           │
├──────────────────────────────────────────────────────────────────┤
│  Layer 5 — LLM Reasoning                                        │
│  Groq/LLaMA-3 → Verdict + Explanation + Hallucination Check     │
├──────────────────────────────────────────────────────────────────┤
│  Layer 6 — Explainability & Reporting                            │
│  SHAP · GradCAM · Attention Rollout · PDF Report + QR Access    │
└──────────────────────────────────────────────────────────────────┘
```

---

## ✨ Features

| Module | What It Does | Models Used |
|--------|-------------|-------------|
| 🩸 **Lab Report Analysis** | OCR extraction → NER → panel classification → risk prediction → LLM explanation | PaddleOCR, ClinicalBERT, XGBoost, LLaMA-3 |
| 🫁 **Chest X-ray Analysis** | 14-class pathology detection with GradCAM heatmaps + YOLO lung detection | CheXNet (DenseNet121-NIH), YOLOv8, GradCAM |
| 🧠 **Brain MRI Analysis** | Tumor classification with 3-model fallback + test-time augmentation | ViT Brain Tumor (3-model ensemble + TTA) |
| 🔬 **CT Scan Analysis** | Multi-pathology classification | TorchXRayVision |
| 🧬 **Skin Lesion Analysis** | HAM10000-trained classification with ABCDE scoring | Fine-tuned ViT + Attention Rollout |
| 🔎 **Pathology Analysis** | CRC tissue classification | Swin Transformer |
| ✅ **Claim Verification** | Medical claim fact-checking with evidence retrieval + hallucination check | ClinicalBERT, BioGPT, ChromaDB, LLaMA-3 |
| 📄 **PDF Report Generation** | Clinical-grade PDF reports with VAIDYAA watermark + QR code access | ReportLab |
| 🔄 **Real-time Updates** | WebSocket-based job status tracking | Celery + Redis + WebSocket |

---

## 🛠️ Tech Stack

```
Backend:     FastAPI · Celery · Redis · PostgreSQL · Alembic
ML/AI:       PyTorch · TorchXRayVision · HuggingFace Transformers · XGBoost
LLM:         Groq API (LLaMA-3) — all text generation
NER:         ClinicalBERT (entity extraction only)
Embeddings:  BioGPT (vector search only)
RAG:         ChromaDB (multi-collection: radiology, skin, pathology)
OCR:         PaddleOCR · PyTesseract · PyMuPDF
Explain:     SHAP · GradCAM · Attention Rollout
Frontend:    React JSX · HTML/CSS
Infra:       Docker · Docker Compose · Flower (Celery monitoring)
```

---

## 🚀 Getting Started

### Prerequisites

- 🐍 Python 3.11+
- 🐘 PostgreSQL 15+
- 📦 Redis 7+
- 🔑 Groq API key

### 1️⃣ Clone & Install

```bash
git clone https://github.com/suchitchopade3110-arch/vadiyaaAI.git
cd vadiyaaAI

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2️⃣ Environment Variables

Create a `.env` file in root:

```env
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=vaidyaai
POSTGRES_USER=vaidya
POSTGRES_PASSWORD=vaidya123

REDIS_URL=redis://localhost:6380/0
CELERY_BROKER_URL=redis://localhost:6380/0
CELERY_RESULT_BACKEND=redis://localhost:6380/1

GROQ_API_KEY=your_groq_api_key_here

UPLOAD_DIR=./uploads
DEBUG=true
```

### 3️⃣ Database Setup

```bash
# Start PostgreSQL & create database
createdb -U vaidya vaidyaai

# Run migrations
alembic upgrade head
```

### 4️⃣ Start Services (3 terminals)

```bash
# Terminal 1 — Redis
redis-server --port 6380

# Terminal 2 — Celery Workers
source .venv/bin/activate
celery -A app.workers.celery_app worker -Q reports,images,claims --concurrency=2

# Terminal 3 — FastAPI Server
source .venv/bin/activate
uvicorn app.main:app --reload --port 8000
```

### 🐳 Or Use Docker Compose

```bash
docker-compose up --build
```

Services: API (`:8000`), PostgreSQL (`:5433`), Redis (`:6380`), Flower (`:5555`)

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/verify/claim` | 🔍 Submit a medical claim for verification |
| `GET` | `/api/v1/verify/claim/status/{task_id}` | 📊 Poll claim verification result |
| `POST` | `/api/v1/analyze/image` | 🫁 Upload medical image for analysis |
| `GET` | `/api/v1/analyze/image/status/{task_id}` | 📊 Poll image analysis result |
| `POST` | `/api/v1/analyze/report` | 🩸 Upload lab report (PDF/CSV/TXT) |
| `GET` | `/api/v1/analyze/report/status/{task_id}` | 📊 Poll report analysis result |
| `GET` | `/api/v1/jobs/{job_id}` | 📋 Get any job status |
| `GET` | `/api/v1/health` | 💚 Health check |
| `WS` | `/ws/jobs/{job_id}` | 🔄 WebSocket real-time status |

---

## 💡 Working Examples

### Example 1: 🔍 Verify a Medical Claim

```bash
curl -X POST http://localhost:8000/api/v1/verify/claim \
  -H "Content-Type: application/json" \
  -d '{
    "claim_text": "Aspirin reduces heart attack risk by 50%",
    "priority": "high"
  }'
```

**Response (202 Accepted):**
```json
{
  "claim_id": "a3f1e2d4-...",
  "task_id": "b7c9d1e3-...",
  "status": "pending",
  "poll_url": "/api/v1/verify/claim/status/b7c9d1e3-...",
  "estimated_seconds": 15
}
```

**Poll for result:**
```bash
curl http://localhost:8000/api/v1/verify/claim/status/b7c9d1e3-...
```

**Final Result:**
```json
{
  "verdict": "refuted",
  "confidence_score": 82,
  "uncertainty_flag": false,
  "explanation": "While aspirin does reduce cardiovascular risk, clinical trials (NEJM 2018) show a 12-25% relative risk reduction, not 50%...",
  "anomalies": ["Exaggerated risk reduction claim"],
  "citations": [
    {
      "source": "PubMed",
      "pmid": "30152035",
      "title": "Aspirin for Primary Prevention — ARRIVE Trial"
    }
  ],
  "medical_disclaimer": "This analysis is for informational purposes only..."
}
```

---

### Example 2: 🫁 Analyze a Chest X-ray

```bash
curl -X POST http://localhost:8000/api/v1/analyze/image \
  -F "file=@chest_xray.jpg" \
  -F "analysis_type=xray"
```

**Response (202 Accepted):**
```json
{
  "analysis_id": "d4e5f6a7-...",
  "task_id": "c8b7a6e5-...",
  "analysis_type": "xray",
  "status": "pending",
  "poll_url": "/api/v1/analyze/image/status/c8b7a6e5-...",
  "estimated_seconds": 45
}
```

**Final Result:**
```json
{
  "classification": {
    "label": "Cardiomegaly",
    "confidence": 87,
    "all_pathologies": {
      "Cardiomegaly": 0.87,
      "Effusion": 0.23,
      "Atelectasis": 0.15,
      "No Finding": 0.08
    }
  },
  "severity": "moderate",
  "gradcam_overlay_url": "/outputs/gradcam_c8b7a6e5.png",
  "yolo_detections": [
    { "label": "lung_opacity", "confidence": 0.76, "bbox": [120, 80, 340, 290] }
  ],
  "rag_context": {
    "source": "WHO Diagnostic Imaging Protocols",
    "relevant_guideline": "Cardiomegaly: cardiothoracic ratio > 0.5 on PA film..."
  },
  "explanation": "CheXNet detected cardiomegaly with high confidence. The cardiothoracic ratio appears elevated...",
  "confidence_score": 87,
  "uncertainty_flag": false,
  "citations": [...],
  "medical_disclaimer": "..."
}
```

---

### Example 3: 🩸 Analyze a Lab Report

```bash
curl -X POST http://localhost:8000/api/v1/analyze/report \
  -F "file=@blood_report.pdf" \
  -F "report_type=blood"
```

**Final Result (after polling):**
```json
{
  "patient_data": {
    "conditions": ["Type 2 Diabetes"],
    "medications": ["Metformin 500mg"],
    "lab_values": {
      "HbA1c": { "value": 8.2, "unit": "%", "reference": "4.0-5.6", "flag": "HIGH" },
      "Fasting Glucose": { "value": 156, "unit": "mg/dL", "reference": "70-100", "flag": "HIGH" },
      "Creatinine": { "value": 1.1, "unit": "mg/dL", "reference": "0.7-1.3", "flag": "NORMAL" }
    }
  },
  "panel_classification": "Metabolic Panel",
  "risk_prediction": {
    "risk_level": "elevated",
    "confidence_score": 78,
    "shap_factors": [
      { "feature": "HbA1c", "impact": 0.42, "direction": "increases_risk" },
      { "feature": "Fasting Glucose", "impact": 0.31, "direction": "increases_risk" }
    ]
  },
  "anomalies": [
    "HbA1c significantly above target (8.2% vs target <7%)",
    "Fasting glucose elevated at 156 mg/dL"
  ],
  "explanation": "Lab results indicate suboptimal glycemic control. HbA1c of 8.2% suggests...",
  "citations": [...],
  "medical_disclaimer": "..."
}
```

---

### Example 4: 🧠 Analyze a Brain MRI

```bash
curl -X POST http://localhost:8000/api/v1/analyze/image \
  -F "file=@brain_mri.jpg" \
  -F "analysis_type=mri"
```

**Final Result:**
```json
{
  "classification": {
    "label": "Glioma",
    "confidence": 91,
    "model_used": "vit_brain_tumor_primary",
    "tta_applied": true,
    "ensemble_agreement": "3/3 models agree"
  },
  "segmentation": {
    "mask_url": "/outputs/seg_mask_f7a8b9c0.png",
    "overlay_url": "/outputs/seg_overlay_f7a8b9c0.png",
    "roi_bounding_box": { "x": 145, "y": 89, "w": 112, "h": 98 }
  },
  "severity": "high",
  "explanation": "ViT ensemble detected glioma with high confidence. Segmentation shows...",
  "confidence_score": 91,
  "citations": [...],
  "medical_disclaimer": "..."
}
```

---

### Example 5: 🧬 Analyze a Skin Lesion

```bash
curl -X POST http://localhost:8000/api/v1/analyze/image \
  -F "file=@skin_lesion.jpg" \
  -F "analysis_type=skin"
```

**Final Result:**
```json
{
  "classification": {
    "label": "Melanocytic Nevi (nv)",
    "confidence": 94,
    "abcde_score": {
      "asymmetry": "low",
      "border": "regular",
      "color": "uniform",
      "diameter": "<6mm",
      "evolution": "N/A"
    }
  },
  "explainability": {
    "method": "attention_rollout",
    "heatmap_url": "/outputs/attention_skin_e3d4c5.png"
  },
  "severity": "low",
  "confidence_score": 94,
  "explanation": "Skin lesion classified as melanocytic nevi (benign mole) with high confidence...",
  "citations": [...],
  "medical_disclaimer": "..."
}
```

---

## 📁 Project Structure

```
vadiyaaAI/
├── 📂 app/
│   ├── 📂 api/v1/routes/       # FastAPI route handlers
│   │   ├── claims.py           # POST /verify/claim
│   │   ├── images.py           # POST /analyze/image
│   │   ├── reports.py          # POST /analyze/report
│   │   ├── jobs.py             # GET /jobs/{id}
│   │   └── health.py           # GET /health
│   ├── 📂 services/            # Business logic layer
│   │   ├── claim_service.py    # Claim verification pipeline
│   │   ├── image_analysis_service.py  # Image analysis orchestrator
│   │   ├── rag_pipeline.py     # RAG retrieval (ChromaDB + BioGPT)
│   │   ├── clinical_ner_service.py    # ClinicalBERT NER
│   │   ├── ml_predictor.py     # XGBoost tabular predictions
│   │   ├── ocr_service.py      # PaddleOCR/Tesseract
│   │   ├── yolo_detector.py    # YOLO lung detection
│   │   ├── pdf_report.py       # Clinical PDF generation
│   │   └── 📂 explainability/  # SHAP, GradCAM modules
│   ├── 📂 image_pipeline/      # Medical image processing
│   │   ├── classifier_v2.py    # CheXNet 14-class classifier
│   │   ├── segmentor.py        # MedSAM segmentation
│   │   ├── gradcam.py          # GradCAM heatmap generation
│   │   └── quality_gate.py     # Input image validation
│   ├── 📂 ml/                  # ML models & prediction engines
│   │   ├── ml_prediction_engine.py    # Ensemble prediction
│   │   ├── 📂 models/          # Saved model weights
│   │   └── 📂 explainability/  # SHAP explainers
│   ├── 📂 rag/                 # RAG retrieval & reasoning
│   │   ├── retriever.py        # ChromaDB vector search
│   │   └── reasoner.py         # LLM reasoning chain
│   ├── 📂 workers/             # Celery async task workers
│   │   ├── celery_app.py       # Celery configuration
│   │   └── pipeline_tasks.py   # Task definitions
│   ├── 📂 schemas/             # Pydantic request/response models
│   ├── 📂 db/                  # SQLAlchemy models & session
│   ├── 📂 core/                # Config, middleware, logging
│   ├── main.py                 # FastAPI app entry point
│   └── pipeline.py             # Master pipeline orchestrator (1245 lines)
├── 📂 ui/                      # React JSX frontend
│   ├── report-analyzer.jsx     # Lab report upload & results
│   ├── image-analysis.jsx      # Medical image analysis UI
│   ├── claim-verifier.jsx      # Claim verification UI
│   ├── dashboard.jsx           # Main dashboard
│   └── job-tracker.jsx         # Real-time job status
├── 📂 data/                    # ChromaDB stores, YOLO outputs, OCR cache
├── 📂 alembic/                 # Database migrations
├── 📂 tests/                   # E2E + pipeline + RAG tests
├── 📂 docs/                    # Architecture diagrams
├── docker-compose.yml          # Full stack: API + PG + Redis + Celery + Flower
├── Dockerfile                  # Python 3.11-slim container
├── requirements.txt            # All dependencies
└── clinical_reference.py       # WHO range enrichment & blood report analysis
```

---

## 🏆 Team Straw Hats

Built with 🔥 for **Smart India Hackathon 2025** by Team Straw Hats from **Sri Sakthi Institute of Engineering and Technology**.

| Member | Role |
|--------|------|
| 👨‍💻 **Suchit** | Team Leader · Backend Lead |
| 🤖 **Shruthi** | LLM / RAG Engineer |
| 📊 **Subhiksha** | ML / Prediction Engineer |
| 🗃️ **Thaariha** | Data / Preprocessing Engineer |
| 🔬 **Shreekumar** | Domain / Research Lead |

---

## 📝 Key Design Principles

- 🔒 **Strict Model Isolation** — BioGPT = embeddings only, ClinicalBERT = NER only, all generation → Groq/LLaMA-3
- 📏 **Standardized Confidence** — Platt-scaled 0–100 (not 0–1) across all modules
- 📚 **Real RAG** — No stubs; ChromaDB with per-modality collections (radiology, skin, pathology)
- 🧪 **Explainability First** — SHAP for tabular, GradCAM for CNNs, Attention Rollout for ViTs
- ⚡ **Async Everything** — Celery workers with Redis broker, WebSocket status updates
- 🛡️ **Every Response Includes** — verdict, confidence_score, uncertainty_flag, anomalies[], citations[], disclaimer

---

## 📄 License

This project was developed for SIH 2025. All rights reserved by Team Straw Hats.

---

<p align="center">
  <strong>🩺 VaidyaAI — Because every diagnosis deserves a second opinion from AI.</strong>
</p>
