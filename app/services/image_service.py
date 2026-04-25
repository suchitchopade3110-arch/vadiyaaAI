from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from uuid import UUID
from fastapi import UploadFile

from app.core.file_validator import validate_upload
from app.core.file_storage import save_upload
from app.models.image_analysis import ImageAnalysis, ImageType
from app.schemas.image import ImageAnalysisResponse, ImageStatusResponse


class ImageService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def submit(
        self, file: UploadFile, image_type: ImageType, patient_id: UUID | None
    ) -> ImageAnalysisResponse:
        # 1. Validate (raises HTTPException on failure)
        raw = await validate_upload(file, pipeline="image")

        # 2. Save with UUID filename — no path traversal risk
        file_path = save_upload(raw, file.filename, subfolder="images")

        # 3. DB record
        analysis = ImageAnalysis(
            image_type=image_type,
            patient_id=patient_id,
            file_path=file_path,
            uncertainty_flag=False,
        )
        self.db.add(analysis)
        await self.db.flush()

        # 4. Celery dispatch
        from app.workers.tasks import run_image_pipeline
        task = run_image_pipeline.delay(str(analysis.id))
        analysis.celery_task_id = task.id

        await self.db.commit()
        await self.db.refresh(analysis)
        return ImageAnalysisResponse.model_validate(analysis)

    async def get(self, analysis_id: UUID) -> ImageAnalysisResponse | None:
        result = await self.db.execute(
            select(ImageAnalysis).where(ImageAnalysis.id == analysis_id)
        )
        obj = result.scalar_one_or_none()
        return ImageAnalysisResponse.model_validate(obj) if obj else None

    async def get_status(self, analysis_id: UUID) -> ImageStatusResponse | None:
        result = await self.db.execute(
            select(ImageAnalysis).where(ImageAnalysis.id == analysis_id)
        )
        obj = result.scalar_one_or_none()
        return ImageStatusResponse.model_validate(obj) if obj else None
