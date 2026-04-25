from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from uuid import UUID

from app.models.claim import Claim, ClaimStatus
from app.schemas.claim import ClaimCreateRequest, ClaimResponse, ClaimStatusResponse


class ClaimService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def submit(self, payload: ClaimCreateRequest) -> ClaimResponse:
        claim = Claim(
            claim_text=payload.claim_text,
            patient_id=payload.patient_id,
            status=ClaimStatus.PENDING,
            uncertainty_flag=False,
        )
        self.db.add(claim)
        await self.db.flush()  # get ID before commit

        # Dispatch to Celery
        from app.workers.tasks import run_claim_pipeline
        task = run_claim_pipeline.delay(str(claim.id))
        claim.celery_task_id = task.id
        claim.status = ClaimStatus.PROCESSING

        await self.db.commit()
        await self.db.refresh(claim)
        return ClaimResponse.model_validate(claim)

    async def get(self, claim_id: UUID) -> ClaimResponse | None:
        result = await self.db.execute(select(Claim).where(Claim.id == claim_id))
        claim = result.scalar_one_or_none()
        return ClaimResponse.model_validate(claim) if claim else None

    async def get_status(self, claim_id: UUID) -> ClaimStatusResponse | None:
        result = await self.db.execute(select(Claim).where(Claim.id == claim_id))
        claim = result.scalar_one_or_none()
        return ClaimStatusResponse.model_validate(claim) if claim else None
