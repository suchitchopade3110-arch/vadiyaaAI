from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import UUID

from app.db.session import get_db
from app.models.image_analysis import ImageType
from app.schemas.image import ImageAnalysisResponse, ImageStatusResponse
from app.services.image_service import ImageService

router = APIRouter()


@router.post(
    "/image/{analysis_type}",
    response_model=ImageAnalysisResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit medical image for AI analysis",
)
async def analyze_image(
    analysis_type: ImageType,
    file: UploadFile = File(...),
    patient_id: UUID | None = None,
    db: AsyncSession = Depends(get_db),
):
    """
    Upload a medical image (DICOM / JPG / PNG) for segmentation + classification.

    Pipeline: Normalize + CLAHE → LiteMedSAM segmentation → CheXNet classify → GradCAM → LLM explain.

    Returns async job ID. Long-running GPU task (10–60s).

    ⚠️ Output is AI-assisted only. NOT a medical diagnosis.
    """
    service = ImageService(db)
    return await service.submit(file=file, image_type=analysis_type, patient_id=patient_id)


@router.get(
    "/image/{analysis_id}",
    response_model=ImageAnalysisResponse,
    summary="Get image analysis result",
)
async def get_image_analysis(
    analysis_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    service = ImageService(db)
    result = await service.get(analysis_id)
    if not result:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return result


@router.get(
    "/image/{analysis_id}/status",
    response_model=ImageStatusResponse,
    summary="Poll async job status for image analysis",
)
async def image_status(
    analysis_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    service = ImageService(db)
    result = await service.get_status(analysis_id)
    if not result:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return result
