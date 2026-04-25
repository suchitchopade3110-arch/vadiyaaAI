from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import UUID

from app.db.session import get_db
from app.schemas.claim import ClaimCreateRequest, ClaimResponse, ClaimStatusResponse
from app.services.claim_service import ClaimService

router = APIRouter()


@router.post(
    "/claim",
    response_model=ClaimResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit medical claim for verification",
)
async def submit_claim(
    payload: ClaimCreateRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Submit a medical claim text for AI-assisted verification.

    Routes to: ClinicalBERT NER → BioGPT → ChromaDB → GPT-4/Llama → Hallucination check.

    Returns async job ID. Poll /verify/claim/{claim_id}/status for result.

    ⚠️ Output is AI-assisted only. NOT a medical diagnosis.
    """
    service = ClaimService(db)
    return await service.submit(payload)


@router.get(
    "/claim/{claim_id}",
    response_model=ClaimResponse,
    summary="Get claim verification result",
)
async def get_claim(
    claim_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    service = ClaimService(db)
    result = await service.get(claim_id)
    if not result:
        raise HTTPException(status_code=404, detail="Claim not found")
    return result


@router.get(
    "/claim/{claim_id}/status",
    response_model=ClaimStatusResponse,
    summary="Poll async job status for claim",
)
async def claim_status(
    claim_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    service = ClaimService(db)
    result = await service.get_status(claim_id)
    if not result:
        raise HTTPException(status_code=404, detail="Claim not found")
    return result
