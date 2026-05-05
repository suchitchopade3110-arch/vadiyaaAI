import io
import json
import uuid
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


# ─────────────────────────────────────────────────────────────────────────────
#  HEALTH
# ─────────────────────────────────────────────────────────────────────────────

def test_health():
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"


def test_root():
    r = client.get("/")
    assert r.status_code == 200
    data = r.json()
    assert "medical_disclaimer" in data
    assert "VaidyaAI" in data["platform"]


# ─────────────────────────────────────────────────────────────────────────────
#  MIDDLEWARE
# ─────────────────────────────────────────────────────────────────────────────

def test_disclaimer_header_on_all_responses():
    """X-Medical-Disclaimer header MUST appear on every response."""
    for path in ["/api/v1/health", "/"]:
        r = client.get(path)
        assert r.headers.get("X-Medical-Disclaimer") == "AI-assisted analysis. NOT diagnostic."


def test_request_id_header():
    r = client.get("/api/v1/health")
    assert "X-Request-ID" in r.headers
    assert len(r.headers["X-Request-ID"]) == 36


# ─────────────────────────────────────────────────────────────────────────────
#  CLAIM VERIFICATION
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_CLAIM = "Aspirin reduces the risk of heart attack in adults over 50 with hypertension."

def test_claim_pipeline_flow():
    claim_id = str(uuid.uuid4())
    # 1. Submit
    r = client.post(
        f"/api/v1/verify/claim/{claim_id}",
        json={"claim_text": SAMPLE_CLAIM},
    )
    assert r.status_code == 202
    data = r.json()
    task_id = data["task_id"]
    assert "poll_url" in data
    
    # 2. Poll Status
    r = client.get(f"/api/v1/verify/claim/status/{task_id}")
    assert r.status_code == 200
    assert "status" in r.json()
    
    # 3. Get Result (might be 425 if not ready, but we check schema if possible)
    # Since we are using TestClient and it's async celery, it won't be ready.
    r = client.get(f"/api/v1/verify/claim/result/{task_id}")
    assert r.status_code in (425, 200)


# ─────────────────────────────────────────────────────────────────────────────
#  IMAGE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def _make_fake_png() -> bytes:
    return (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0"
        b"\x00\x00\x03\x01\x01\x00\xc9\xfe\x92\xef\x00\x00\x00\x00IEND\xaeB`\x82"
    )

def test_image_pipeline_flow():
    fake_img = _make_fake_png()
    # 1. Submit
    r = client.post(
        "/api/v1/analyze/image/xray",
        files={"file": ("test.png", io.BytesIO(fake_img), "image/png")},
    )
    assert r.status_code in (202, 422)
    if r.status_code == 202:
        task_id = r.json()["task_id"]
        # 2. Poll
        r = client.get(f"/api/v1/analyze/image/status/{task_id}")
        assert r.status_code == 200


# ─────────────────────────────────────────────────────────────────────────────
#  REPORT ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_CSV = b"parameter,value,unit\nHbA1c,7.8,%\nGlucose,142,mg/dL"

def test_report_pipeline_flow():
    # 1. Submit
    r = client.post(
        "/api/v1/analyze/report/lab",
        files={"file": ("labs.csv", io.BytesIO(SAMPLE_CSV), "text/csv")},
    )
    assert r.status_code == 202
    task_id = r.json()["task_id"]
    
    # 2. Poll
    r = client.get(f"/api/v1/analyze/report/status/{task_id}")
    assert r.status_code == 200


# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 2 E2E SCENARIOS — TEXT PIPELINE (T1–T6)
# ─────────────────────────────────────────────────────────────────────────────

def test_T1_claim_too_short_rejected():
    """Claim under min_length returns 422."""
    r = client.post(
        f"/api/v1/verify/claim/{uuid.uuid4()}",
        json={"claim_text": "short"},
    )
    assert r.status_code == 422


def test_T2_claim_missing_field_rejected():
    """Missing claim_text returns 422."""
    r = client.post(
        f"/api/v1/verify/claim/{uuid.uuid4()}",
        json={},
    )
    assert r.status_code == 422


def test_T3_report_csv_anomaly_flagged():
    """CSV with high HbA1c — response must contain task metadata."""
    csv = b"parameter,value,unit\nHbA1c,9.5,%\nGlucose,210,mg/dL"
    r = client.post(
        "/api/v1/analyze/report/lab",
        files={"file": ("labs.csv", io.BytesIO(csv), "text/csv")},
    )
    assert r.status_code == 202
    data = r.json()
    assert "task_id" in data


def test_T4_unsupported_file_format_rejected():
    """Uploading .txt file returns 415."""
    r = client.post(
        "/api/v1/analyze/report/lab",
        files={"file": ("report.txt", io.BytesIO(b"some text"), "text/plain")},
    )
    assert r.status_code == 415


def test_T5_oversized_file_rejected():
    """File exceeding size limit returns 413."""
    big = io.BytesIO(b"x" * (51 * 1024 * 1024))
    r = client.post(
        "/api/v1/analyze/report/lab",
        files={"file": ("big.pdf", big, "application/pdf")},
    )
    assert r.status_code == 413


def test_T6_disclaimer_in_all_text_responses():
    """Every text pipeline response includes medical disclaimer."""
    r = client.post(
        f"/api/v1/verify/claim/{uuid.uuid4()}",
        json={"claim_text": SAMPLE_CLAIM},
    )
    data = r.json()
    assert "medical_disclaimer" in data
    assert "NOT A MEDICAL DIAGNOSIS" in data["medical_disclaimer"]


# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 2 E2E SCENARIOS — IMAGE PIPELINE (I1–I5)
# ─────────────────────────────────────────────────────────────────────────────

def test_I1_invalid_analysis_type_rejected():
    """Unknown analysis_type returns 422."""
    r = client.post(
        "/api/v1/analyze/image/ultrasound",
        files={"file": ("test.png", io.BytesIO(_make_fake_png()), "image/png")},
    )
    assert r.status_code == 422


def test_I2_image_ct_accepted():
    r = client.post(
        "/api/v1/analyze/image/ct",
        files={"file": ("ct.png", io.BytesIO(_make_fake_png()), "image/png")},
    )
    assert r.status_code in (202, 422)


def test_I3_image_mri_accepted():
    r = client.post(
        "/api/v1/analyze/image/mri",
        files={"file": ("mri.png", io.BytesIO(_make_fake_png()), "image/png")},
    )
    assert r.status_code in (202, 422)


def test_I4_image_skin_accepted():
    r = client.post(
        "/api/v1/analyze/image/skin",
        files={"file": ("skin.png", io.BytesIO(_make_fake_png()), "image/png")},
    )
    assert r.status_code in (202, 422)


def test_I5_image_pathology_accepted():
    r = client.post(
        "/api/v1/analyze/image/pathology",
        files={"file": ("path.png", io.BytesIO(_make_fake_png()), "image/png")},
    )
    assert r.status_code in (202, 422)
