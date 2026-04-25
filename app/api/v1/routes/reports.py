from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import UUID

from app.db.session import get_db
from app.models.report import ReportType
from app.schemas.report import ReportResponse, ReportStatusResponse
from app.services.report_service import ReportService

router = APIRouter()


@router.post(
    "/report/{report_type}",
    response_model=ReportResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit lab/clinical report for AI analysis",
)
async def analyze_report(
    report_type: ReportType,
    file: UploadFile = File(...),
    patient_id: UUID | None = None,
    db: AsyncSession = Depends(get_db),
):
    """
    Upload a PDF/CSV medical report for extraction + analysis.

    Pipeline: OCR → ClinicalBERT NER → XGBoost → SHAP → LLM explain → RAG citations.

    Returns async job ID.

    ⚠️ Output is AI-assisted only. NOT a medical diagnosis.
    """
    service = ReportService(db)
    return await service.submit(file=file, report_type=report_type, patient_id=patient_id)


@router.get(
    "/report/{report_id}",
    response_model=ReportResponse,
    summary="Get report analysis result",
)
async def get_report(
    report_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    service = ReportService(db)
    result = await service.get(report_id)
    if not result:
        raise HTTPException(status_code=404, detail="Report not found")
    return result


@router.get(
    "/report/{report_id}/status",
    response_model=ReportStatusResponse,
    summary="Poll async job status for report",
)
async def report_status(
    report_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    service = ReportService(db)
    result = await service.get_status(report_id)
    if not result:
        raise HTTPException(status_code=404, detail="Report not found")
    return result
