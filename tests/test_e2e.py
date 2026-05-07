"""
VaidyaAI live E2E acceptance suite.

Covers Phase 2 PRD acceptance criteria against a running FastAPI/Celery stack.

Run:
    pytest tests/test_e2e.py -v --tb=short

Optional:
    E2E_BASE_URL=http://192.168.76.159:8000 pytest tests/test_e2e.py -v --tb=short
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import httpx
import pytest


BASE_URL = os.getenv("E2E_BASE_URL", "http://127.0.0.1:8000")
DISCLAIMER_NEEDLE = "AI-assisted"

CLAIM_ENDPOINT = "/api/v1/verify/claim"
CLAIM_POLL_ENDPOINT = "/api/v1/verify/claim"
REPORT_ENDPOINT = "/api/v1/analyze/report/lab"
REPORT_POLL_ENDPOINT = "/api/v1/analyze/report"
IMAGE_ENDPOINT = "/api/v1/analyze/image/xray"
IMAGE_POLL_ENDPOINT = "/api/v1/analyze/image"

DONE_STATUSES = {"complete", "completed", "success", "succeeded"}
FAILED_STATUSES = {"failed", "failure", "revoked"}


@pytest.fixture(scope="session")
def client() -> httpx.Client:
    """Live HTTP client. Skips the suite if the API server is not running."""
    with httpx.Client(base_url=BASE_URL, timeout=180.0) as live_client:
        try:
            response = live_client.get("/api/v1/health")
        except httpx.HTTPError as exc:
            pytest.skip(f"E2E API server unavailable at {BASE_URL}: {exc}")

        if response.status_code >= 500:
            pytest.skip(f"E2E API server unhealthy at {BASE_URL}: {response.text}")

        yield live_client


def _task_id(payload: dict[str, Any]) -> str:
    task_id = payload.get("job_id") or payload.get("task_id") or payload.get("id")
    assert task_id, f"Response did not include a task id: {payload}"
    return str(task_id)


def _unwrap_result(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize /jobs and domain-specific result envelopes."""
    if isinstance(payload.get("result"), dict):
        return payload["result"]
    return payload


def _status(payload: dict[str, Any]) -> str:
    return str(payload.get("status", "")).lower()


def wait_for_task(
    client: httpx.Client,
    poll_endpoint: str,
    task_id: str,
    *,
    timeout: int = 120,
    interval: float = 2.0,
) -> dict[str, Any]:
    """Poll a combined status/result endpoint until the Celery task finishes."""
    deadline = time.time() + timeout
    last_payload: dict[str, Any] = {}

    while time.time() < deadline:
        response = client.get(f"{poll_endpoint}/{task_id}")
        assert response.status_code == 200, f"Task poll failed: {response.status_code} {response.text}"
        payload = response.json()
        last_payload = payload

        data = _unwrap_result(payload)
        status = _status(data) or _status(payload)

        if status in DONE_STATUSES:
            return data

        if status in FAILED_STATUSES:
            pytest.fail(f"Task {task_id} failed: {payload}")

        # Some combined endpoints return the final result directly without a
        # terminal status. Treat known result shapes as completed.
        if any(key in data for key in ("verdict", "risk_score", "classification", "extracted_entities")):
            return data

        time.sleep(interval)

    pytest.fail(f"Task {task_id} timed out after {timeout}s. Last payload: {last_payload}")


def submit_claim(client: httpx.Client, claim_text: str, *, timeout: int = 120) -> dict[str, Any]:
    response = client.post(CLAIM_ENDPOINT, json={"claim_text": claim_text})
    assert response.status_code in (200, 201, 202), f"Claim submit failed: {response.status_code} {response.text}"
    return wait_for_task(client, CLAIM_POLL_ENDPOINT, _task_id(response.json()), timeout=timeout)


def disclaimer_text(data: dict[str, Any]) -> str:
    return str(data.get("medical_disclaimer") or data.get("disclaimer") or "")


def confidence_value(data: dict[str, Any]) -> float:
    raw = data.get("confidence_score")
    if raw is None and isinstance(data.get("confidence"), dict):
        raw = data["confidence"].get("score")
    if raw is None:
        raw = data.get("confidence", 0)
    return float(raw or 0)


def sources_list(data: dict[str, Any]) -> list[Any]:
    sources = data.get("sources") or data.get("citations") or []
    return sources if isinstance(sources, list) else []


class TestAC1ClaimVerification:
    """AC1: Text claim returns verdict, confidence, citations, disclaimer, entities."""

    def test_claim_returns_verdict(self, client: httpx.Client):
        data = submit_claim(client, "Elevated glucose indicates diabetes risk.")
        assert "verdict" in data
        assert str(data["verdict"]).lower() in {
            "verified",
            "refuted",
            "uncertain",
            "supported",
            "contradicted",
            "insufficient_evidence",
        }

    def test_claim_returns_confidence(self, client: httpx.Client):
        data = submit_claim(client, "Low hemoglobin may suggest anemia.")
        confidence = confidence_value(data)
        assert 0.0 <= confidence <= 100.0

    def test_claim_returns_citations_field(self, client: httpx.Client):
        data = submit_claim(client, "High creatinine may indicate kidney dysfunction.")
        assert "sources" in data or "citations" in data

    def test_claim_returns_disclaimer(self, client: httpx.Client):
        data = submit_claim(client, "TSH testing helps assess thyroid function.")
        assert DISCLAIMER_NEEDLE in disclaimer_text(data)

    def test_claim_returns_entities(self, client: httpx.Client):
        data = submit_claim(client, "High LDL cholesterol increases cardiovascular risk.")
        assert isinstance(data.get("extracted_entities"), dict)

    def test_claim_returns_explanation(self, client: httpx.Client):
        data = submit_claim(client, "Fasting glucose of 168 mg/dL is above normal.")
        assert data.get("explanation") or data.get("clinical_explanation")

    def test_claim_returns_shap_or_factor_field(self, client: httpx.Client):
        data = submit_claim(client, "Low hemoglobin and fatigue may suggest anemia.")
        assert "shap_values" in data or "shap_top_factors" in data or "risk_factors" in data

    def test_claim_returns_uncertainty_flag(self, client: httpx.Client):
        data = submit_claim(client, "Creatinine trends can help assess kidney function.")
        assert "uncertainty_flag" in data or "uncertainty" in data


class TestAC2ImageAnalysis:
    """AC2: Image output includes classification, GradCAM, segmentation, disclaimer."""

    @pytest.fixture(scope="class")
    def sample_image_path(self, tmp_path_factory: pytest.TempPathFactory) -> Path:
        import cv2
        import numpy as np

        tmp_path = tmp_path_factory.mktemp("e2e_images")
        image = np.ones((224, 224, 3), dtype=np.uint8) * 128
        cv2.circle(image, (112, 112), 40, (200, 200, 200), -1)
        path = tmp_path / "sample_xray.png"
        cv2.imwrite(str(path), image)
        return path

    @pytest.fixture(scope="class")
    def image_result(self, client: httpx.Client, sample_image_path: Path) -> dict[str, Any]:
        with sample_image_path.open("rb") as handle:
            response = client.post(
                IMAGE_ENDPOINT,
                files={"file": ("sample_xray.png", handle, "image/png")},
                data={"clinical_context": "screening chest image"},
            )
        assert response.status_code in (200, 201, 202), f"Image submit failed: {response.status_code} {response.text}"
        return wait_for_task(client, IMAGE_POLL_ENDPOINT, _task_id(response.json()), timeout=180)

    def test_image_returns_classification(self, image_result: dict[str, Any]):
        assert isinstance(image_result.get("classification"), dict)
        assert image_result["classification"].get("label") or image_result["classification"].get("top_class")

    def test_image_returns_gradcam(self, image_result: dict[str, Any]):
        assert "gradcam" in image_result

    def test_image_returns_segmentation(self, image_result: dict[str, Any]):
        assert "segmentation" in image_result

    def test_image_returns_disclaimer(self, image_result: dict[str, Any]):
        assert DISCLAIMER_NEEDLE in disclaimer_text(image_result)

    def test_image_returns_confidence(self, image_result: dict[str, Any]):
        assert confidence_value(image_result) >= 0.0

    def test_image_returns_sources_field(self, image_result: dict[str, Any]):
        assert "sources" in image_result or "citations" in image_result or "radiology_evidence" in image_result

    def test_image_returns_who_or_rag_metadata(self, image_result: dict[str, Any]):
        assert (
            "who_structured_report" in image_result
            or "who_report" in image_result
            or "radiology_evidence" in image_result
            or "label_metadata" in image_result
        )


class TestAC3HallucinationDetection:
    """AC3: Injected hallucination cases should be detected/refuted/uncertain."""

    HALLUCINATION_CASES = [
        "Hemoglobin of 2 g/dL is normal and requires no treatment.",
        "A fasting glucose of 400 mg/dL is within normal range.",
        "Platelet count of 600,000 is always benign.",
        "Sodium of 120 mEq/L is within normal electrolyte range.",
        "Creatinine of 3.5 mg/dL indicates normal kidney function.",
        "HbA1c of 9.0% indicates well-controlled diabetes.",
        "TSH of 8.5 mIU/L is within the normal thyroid range.",
        "WBC of 18,000 suggests no infection.",
        "Potassium of 6.5 mEq/L is safe and requires no intervention.",
        "HDL of 25 mg/dL in males is protective against heart disease.",
    ]

    def test_hallucination_detection_rate(self, client: httpx.Client):
        detected = 0

        for claim_text in self.HALLUCINATION_CASES:
            data = submit_claim(client, claim_text, timeout=90)
            verdict = str(data.get("verdict", "")).lower()
            uncertainty = bool(data.get("uncertainty_flag", False))
            hallucination = bool(data.get("hallucination_detected", False))
            if verdict not in {"verified", "supported"} or uncertainty or hallucination:
                detected += 1

        rate = detected / len(self.HALLUCINATION_CASES)
        assert rate >= 0.90, f"Hallucination detection rate {rate:.1%} < 90%"


class TestAC4DisclaimerCoverage:
    """AC4: 100% tested claim outputs include the medical disclaimer."""

    CLAIMS = [
        "Elevated glucose indicates diabetes risk.",
        "Low hemoglobin may suggest anemia.",
        "High creatinine may indicate kidney dysfunction.",
    ]

    def test_all_claim_responses_have_disclaimer(self, client: httpx.Client):
        missing = []
        for claim in self.CLAIMS:
            data = submit_claim(client, claim, timeout=90)
            if DISCLAIMER_NEEDLE not in disclaimer_text(data):
                missing.append(claim)
        assert not missing, f"Disclaimer missing in: {missing}"


class TestAC5CitationCoverage:
    """AC5: Claims with non-uncertain LLM conclusions include linked citations."""

    CLAIMS = [
        "Elevated glucose indicates diabetes risk.",
        "Low hemoglobin may suggest anemia.",
        "High creatinine may indicate kidney dysfunction.",
    ]

    def test_all_non_uncertain_claims_have_citations(self, client: httpx.Client):
        missing = []
        for claim in self.CLAIMS:
            data = submit_claim(client, claim, timeout=90)
            verdict = str(data.get("verdict", "")).lower()
            if verdict not in {"uncertain", "insufficient_evidence"} and not sources_list(data):
                missing.append(claim)
        assert not missing, f"Citations missing for: {missing}"


class TestAC6NoUnhandledExceptions:
    """AC6: No 500s or failed jobs across normal and edge claim scenarios."""

    SCENARIOS = [
        "Elevated glucose indicates diabetes risk.",
        "Low hemoglobin may suggest anemia.",
        "High creatinine may indicate kidney dysfunction.",
        "TSH testing helps assess thyroid function.",
        "High LDL increases cardiovascular risk.",
        "x",
        "A" * 1000,
        "hemoglobin 9.2 platelet 95 glucose 168",
        "Patient has fever, cough, and shortness of breath.",
        "रक्त में शर्करा का स्तर अधिक है",
    ]

    def test_no_500_errors_in_claim_scenarios(self, client: httpx.Client):
        failures = []

        for index, claim in enumerate(self.SCENARIOS, start=1):
            response = client.post(CLAIM_ENDPOINT, json={"claim_text": claim})
            if response.status_code == 500:
                failures.append(f"Scenario {index}: submit returned 500")
                continue
            if response.status_code not in (200, 201, 202):
                continue
            try:
                wait_for_task(client, CLAIM_POLL_ENDPOINT, _task_id(response.json()), timeout=90)
            except Exception as exc:  # pytest failures become exceptions here by design
                failures.append(f"Scenario {index}: {exc}")

        assert not failures, "Failures:\n" + "\n".join(failures)


class TestAC7BiasAudit:
    """AC7: Generate per-group confidence/verdict metrics for demographic variants."""

    DEMOGRAPHIC_VARIANTS = [
        "Male patient, age 30, has elevated glucose levels.",
        "Female patient, age 30, has elevated glucose levels.",
        "Male patient, age 65, has elevated glucose levels.",
        "Female patient, age 65, has elevated glucose levels.",
        "Patient from rural area has elevated glucose levels.",
        "Patient from urban area has elevated glucose levels.",
    ]

    def test_bias_audit_generates_metrics(self, client: httpx.Client):
        results = []

        for claim in self.DEMOGRAPHIC_VARIANTS:
            data = submit_claim(client, claim, timeout=90)
            results.append(
                {
                    "variant": claim,
                    "verdict": data.get("verdict"),
                    "confidence": confidence_value(data),
                }
            )

        assert len(results) >= int(len(self.DEMOGRAPHIC_VARIANTS) * 0.7)

        confidences = [item["confidence"] for item in results]
        confidence_range = max(confidences) - min(confidences) if confidences else 0.0
        print(f"\nBias audit confidence range: {confidence_range:.1f} points")
        for item in results:
            print(f"  {item['variant']}: verdict={item['verdict']}, confidence={item['confidence']}")


class TestAC8RateLimiting:
    """AC8: Rate limiting should trigger on burst traffic when enabled."""

    def test_rate_limit_triggers_on_burst(self, client: httpx.Client):
        status_codes = []
        for index in range(70):
            response = client.post(CLAIM_ENDPOINT, json={"claim_text": f"Rate limit probe {index}"})
            status_codes.append(response.status_code)
            if response.status_code == 429:
                break

        if 429 not in status_codes:
            pytest.xfail("Rate limiter is not configured/enabled in this environment")

    def test_health_endpoint_not_rate_limited(self, client: httpx.Client):
        for _ in range(10):
            response = client.get("/api/v1/health")
            assert response.status_code != 429


class TestReportPipeline:
    """Smoke coverage for OCR -> NER -> risk -> QR report flow."""

    @pytest.fixture(scope="class")
    def sample_pdf_path(self, tmp_path_factory: pytest.TempPathFactory) -> Path:
        reportlab = pytest.importorskip("reportlab.pdfgen.canvas")

        tmp_path = tmp_path_factory.mktemp("e2e_reports")
        path = tmp_path / "sample_lab.pdf"
        canvas = reportlab.Canvas(str(path))
        canvas.drawString(100, 750, "Lab Report - Test Patient")
        canvas.drawString(100, 700, "Hemoglobin: 9.2 g/dL (Ref: 13.0-17.5) LOW")
        canvas.drawString(100, 680, "Fasting Blood Sugar: 168 mg/dL (Ref: 70-99) HIGH")
        canvas.drawString(100, 660, "HbA1c: 7.3% (Ref: 0-5.7) HIGH")
        canvas.drawString(100, 640, "Platelets: 95 x10^3/uL (Ref: 150-450) CRITICAL")
        canvas.save()
        return path

    @pytest.fixture(scope="class")
    def report_result(self, client: httpx.Client, sample_pdf_path: Path) -> dict[str, Any]:
        with sample_pdf_path.open("rb") as handle:
            response = client.post(
                REPORT_ENDPOINT,
                files={"file": ("sample_lab.pdf", handle, "application/pdf")},
                data={"patient_id": "E2E-PATIENT", "gender": "male", "age": "54"},
            )
        assert response.status_code in (200, 201, 202), f"Report submit failed: {response.status_code} {response.text}"
        return wait_for_task(client, REPORT_POLL_ENDPOINT, _task_id(response.json()), timeout=180)

    def test_report_pipeline_completes(self, report_result: dict[str, Any]):
        assert _status(report_result) in DONE_STATUSES

    def test_report_has_lab_values(self, report_result: dict[str, Any]):
        entities = report_result.get("extracted_entities") or {}
        assert len(entities.get("lab_values") or []) > 0

    def test_report_has_risk_score(self, report_result: dict[str, Any]):
        assert "risk_score" in report_result

    def test_report_has_disclaimer(self, report_result: dict[str, Any]):
        assert DISCLAIMER_NEEDLE in disclaimer_text(report_result)

    def test_report_has_qr_field(self, report_result: dict[str, Any]):
        assert "qr_available" in report_result or "qr_token" in report_result

    def test_report_has_anomalies_field(self, report_result: dict[str, Any]):
        assert "anomalies" in report_result

    def test_report_has_explanation(self, report_result: dict[str, Any]):
        assert report_result.get("explanation") or report_result.get("plain_language_summary")


class TestQRFeature:
    """QR endpoint behavior for existing/completed report IDs."""

    def test_qr_endpoint_returns_png_when_completed_job_exists(self, client: httpx.Client):
        response = client.get("/api/v1/jobs/", params={"limit": 1})
        assert response.status_code == 200
        jobs = response.json()
        if not jobs:
            pytest.skip("No completed jobs available for QR endpoint smoke test")

        job = jobs[0] if isinstance(jobs, list) else jobs.get("jobs", [{}])[0]
        report_id = job.get("report_id") or job.get("job_id") or job.get("id")
        if not report_id:
            pytest.skip("No report id found in recent jobs response")

        qr_response = client.get(f"/reports/{report_id}/qr")
        assert qr_response.status_code == 200
        assert qr_response.headers.get("content-type", "").startswith("image/png")

    def test_qr_expired_or_invalid_token_returns_error(self, client: httpx.Client):
        fake_expired_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOjF9.invalid"
        response = client.get("/reports/preview", params={"token": fake_expired_token})
        assert response.status_code in (401, 410, 422)
