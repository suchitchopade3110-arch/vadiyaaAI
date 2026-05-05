import pytest


def test_verify_claim_stub_returns_valid_schema():
    from app.services.rag_pipeline import verify_claim

    result = verify_claim("High glucose may indicate diabetes risk.")

    assert "verdict" in result
    assert "summary" in result
    assert "confidence" in result
    assert "citations" in result
    assert "uncertain" in result
    assert "disclaimer" in result
    assert result["verdict"] in {
        "supported",
        "contradicted",
        "insufficient_evidence",
    }
    assert 0.0 <= result["confidence"] <= 1.0
    assert "NOT a medical diagnosis" in result["disclaimer"]


def test_verify_claim_stub_is_insufficient_evidence():
    from app.services.rag_pipeline import verify_claim

    result = verify_claim("some random claim")

    assert result["verdict"] == "insufficient_evidence"
    assert result["uncertain"] is True


def test_verify_claim_task_exists():
    from app.workers.tasks_rag import verify_claim_task

    assert verify_claim_task is not None
    assert verify_claim_task.name == "rag.verify_claim"


@pytest.mark.integration
def test_full_claim_pipeline_roundtrip():
    from app.workers.tasks_rag import verify_claim_task
    import time

    job = verify_claim_task.delay(
        "High glucose may indicate diabetes risk.",
        "test-claim-id-001",
    )

    deadline = time.time() + 90
    while not job.ready():
        assert time.time() < deadline, "Task timed out"
        time.sleep(2)

    result = job.get()

    assert result["verdict"] in {
        "supported",
        "contradicted",
        "insufficient_evidence",
    }
    assert "disclaimer" in result
    assert "NOT a medical diagnosis" in result["disclaimer"]
    assert "completed_at" in result
    assert result["claim_id"] == "test-claim-id-001"

    if result["verdict"] == "supported":
        assert len(result["citations"]) > 0


@pytest.mark.integration
def test_http_claim_submit_and_poll():
    from fastapi.testclient import TestClient
    from app.main import app
    import time

    client = TestClient(app)

    resp = client.post(
        "/api/v1/verify/claim",
        json={
            "claim_text": "Low hemoglobin may suggest anemia.",
            "top_k": 3,
            "prompt_version": "v2",
        },
    )
    assert resp.status_code == 202
    data = resp.json()
    assert "celery_task_id" in data
    assert "id" in data
    claim_id = data["id"]

    deadline = time.time() + 90
    while True:
        poll = client.get(f"/api/v1/verify/claim/{claim_id}/status")
        assert poll.status_code == 200
        poll_data = poll.json()
        if poll_data["status"] in {"verified", "refuted", "uncertain", "failed"}:
            break
        assert time.time() < deadline, "Poll timed out"
        time.sleep(2)

    assert "NOT a medical diagnosis" in poll_data["medical_disclaimer"]
