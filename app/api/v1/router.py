from fastapi import APIRouter
from app.api.v1.routes import claims, images, reports, jobs

api_router = APIRouter()

api_router.include_router(claims.router, prefix="/verify", tags=["Claim Verification"])
api_router.include_router(images.router, prefix="/analyze", tags=["Image Analysis"])
api_router.include_router(reports.router, prefix="/analyze", tags=["Report Analysis"])
api_router.include_router(jobs.router, prefix="/jobs", tags=["Job Status"])
