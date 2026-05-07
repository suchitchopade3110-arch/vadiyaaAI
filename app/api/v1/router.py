from fastapi import APIRouter
from app.api.v1.routes import admin, claims, images, reports, jobs, health

api_router = APIRouter()

api_router.include_router(health.router, prefix="/health", tags=["Health"])
api_router.include_router(claims.router, prefix="/verify", tags=["Claim Verification"])
api_router.include_router(images.router, prefix="/analyze/image", tags=["Image Analysis"])
api_router.include_router(reports.router, prefix="/analyze/report", tags=["Report Analysis"])
api_router.include_router(jobs.router, prefix="/jobs", tags=["Job Status"])
api_router.include_router(admin.router, prefix="/admin", tags=["Admin"])
