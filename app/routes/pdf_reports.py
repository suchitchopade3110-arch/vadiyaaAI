"""
VaidyaAI — PDF Report Generation Route
GET /api/v1/report/{job_id}/pdf -> downloads clinical PDF
"""
from celery.result import AsyncResult
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from app.core.disclaimer import MEDICAL_DISCLAIMER
from app.services.pdf_report import generate_report_pdf
from app.workers.celery_app import celery_app

router = APIRouter()
DISCLAIMER_HEADER = "AI-assisted analysis. NOT diagnostic."


@router.get("/report/{job_id}/pdf")
async def download_pdf_report(job_id: str):
    """
    Fetch a completed Celery job result and stream it as a PDF download.
    Returns 425 if the job is not yet complete.
    """
    result = AsyncResult(job_id, app=celery_app)

    if not result.ready():
        raise HTTPException(
            status_code=425,
            detail={
                "code": "REPORT_NOT_READY",
                "message": "Job still processing. Poll /jobs/{job_id} and retry when status=complete.",
                "job_id": job_id,
                "retryable": True,
            },
        )

    if result.failed():
        raise HTTPException(
            status_code=500,
            detail={
                "code": "JOB_FAILED",
                "message": "Job failed - cannot generate PDF.",
                "job_id": job_id,
                "retryable": False,
            },
        )

    job_result = result.result or {}
    report_data = {
        "report_type": job_result.get("report_type") or job_result.get("pipeline_type") or job_result.get("image_type") or "lab",
        "patient_id": job_result.get("patient_id", "Anonymous"),
        "risk_score": job_result.get("risk_score", 0),
        "risk_label": job_result.get("risk_label", "unknown"),
        "confidence_score": job_result.get(
            "confidence_score",
            job_result.get("confidence", job_result.get("risk_score", 0)),
        ),
        "lab_values": job_result.get("lab_values", []),
        "anomalies": job_result.get("anomalies", []),
        "classification": job_result.get("classification") or job_result.get("image_classification") or {},
        "findings": job_result.get("findings") or (job_result.get("classification") or {}).get("findings") or [],
        "gradcam_path": job_result.get("gradcam_path") or (job_result.get("gradcam") or {}).get("heatmap_path"),
        "gradcam": job_result.get("gradcam") or {},
        "shap_factors": job_result.get("shap_values", {}),
        "explanation": job_result.get("explanation", ""),
        "sources": job_result.get("sources", []) or job_result.get("radiology_evidence", []),
        "citations": job_result.get("sources", []),
        "radiology_evidence": job_result.get("radiology_evidence", []),
        "uncertainty_flag": job_result.get("uncertainty_flag", False),
        "generated_at": job_result.get("completed_at", ""),
    }

    pdf_bytes = generate_report_pdf(report_data)
    filename = f"vaidyaai_report_{job_id[:8]}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "X-Medical-Disclaimer": DISCLAIMER_HEADER,
        },
    )
