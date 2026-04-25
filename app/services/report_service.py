from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from uuid import UUID
from fastapi import UploadFile

from app.core.file_validator import validate_upload
from app.core.file_storage import save_upload
from app.models.report import Report, ReportType
from app.schemas.report import ReportResponse, ReportStatusResponse


class ReportService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def submit(
        self, file: UploadFile, report_type: ReportType, patient_id: UUID | None
    ) -> ReportResponse:
        # 1. Validate
        raw = await validate_upload(file, pipeline="report")

        # 2. Save with UUID filename
        file_path = save_upload(raw, file.filename, subfolder="reports")

        # 3. DB record
        report = Report(
            report_type=report_type,
            patient_id=patient_id,
            file_path=file_path,
            uncertainty_flag=False,
        )
        self.db.add(report)
        await self.db.flush()

        # 4. Celery dispatch
        from app.workers.tasks import run_report_pipeline
        task = run_report_pipeline.delay(str(report.id))
        report.celery_task_id = task.id

        await self.db.commit()
        await self.db.refresh(report)
        return ReportResponse.model_validate(report)

    async def get(self, report_id: UUID) -> ReportResponse | None:
        result = await self.db.execute(
            select(Report).where(Report.id == report_id)
        )
        obj = result.scalar_one_or_none()
        return ReportResponse.model_validate(obj) if obj else None

    async def get_status(self, report_id: UUID) -> ReportStatusResponse | None:
        result = await self.db.execute(
            select(Report).where(Report.id == report_id)
        )
        obj = result.scalar_one_or_none()
        return ReportStatusResponse.model_validate(obj) if obj else None
