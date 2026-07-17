"""Lightweight pre-demo load test for VaidyaAI.

Not wired into CI — this is a manual sanity tool you run yourself before a
demo/pilot to see how the stack behaves under concurrent load. It exercises
the same endpoints a real client hits: health, claim verification, report
upload, and image upload.

Usage:
    pip install locust
    locust -f load-tests/locustfile.py --host http://localhost:8000

Then open http://localhost:8089 and set concurrent users / spawn rate.

Every /api/v1/* endpoint here requires auth (see app/core/auth.py). Each
simulated user registers its own throwaway account once, on_start, then
reuses the access token for every request in its session — the same flow
a real client follows (see POST /auth/register, POST /auth/login).
"""
from __future__ import annotations

import io
import random
import string
import uuid

from locust import HttpUser, between, task

MINIMAL_PDF = (
    b"%PDF-1.4\n"
    b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
    b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
    b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 200 200]>>endobj\n"
    b"trailer<</Size 4/Root 1 0 R>>\n"
    b"%%EOF"
)

SAMPLE_CLAIMS = [
    "Aspirin reduces heart attack risk by 50% in all adults over 40.",
    "Metformin is first-line therapy for newly diagnosed type 2 diabetes.",
    "A fasting glucose of 250 mg/dL indicates well-controlled diabetes.",
    "Elevated CRP always indicates a bacterial infection.",
]


def _random_png_bytes() -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (32, 32), color=(200, 30, 30)).save(buf, format="PNG")
    return buf.getvalue()


class VaidyaAIUser(HttpUser):
    """One simulated clinician/uploader session."""

    wait_time = between(1, 3)

    def on_start(self):
        username = "loadtest_" + "".join(random.choices(string.ascii_lowercase, k=12))
        password = "loadtest-password-123"

        self.client.post(
            "/auth/register",
            json={"username": username, "password": password},
            name="/auth/register",
        )
        resp = self.client.post(
            "/auth/login",
            json={"username": username, "password": password},
            name="/auth/login",
        )
        token = resp.json().get("access_token") if resp.status_code == 200 else None
        self.auth_header = {"Authorization": f"Bearer {token}"} if token else {}

    @task(6)
    def health(self):
        self.client.get("/health", name="/health")

    @task(4)
    def verify_claim(self):
        self.client.post(
            "/api/v1/verify/claim",
            json={"claim_text": random.choice(SAMPLE_CLAIMS), "priority": "normal"},
            headers=self.auth_header,
            name="/api/v1/verify/claim",
        )

    @task(2)
    def analyze_report(self):
        files = {"file": (f"{uuid.uuid4()}.pdf", MINIMAL_PDF, "application/pdf")}
        data = {"gender": "male", "age": "40", "explanation_mode": "brief"}
        self.client.post(
            "/api/v1/analyze/report/lab",
            files=files,
            data=data,
            headers=self.auth_header,
            name="/api/v1/analyze/report/[type]",
        )

    @task(2)
    def analyze_image(self):
        files = {"file": (f"{uuid.uuid4()}.png", _random_png_bytes(), "image/png")}
        self.client.post(
            "/api/v1/analyze/image/xray",
            files=files,
            headers=self.auth_header,
            name="/api/v1/analyze/image/[type]",
        )
