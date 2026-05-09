import io
import os

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.session import get_db
from app.services.qr_service import (
    ALGORITHM,
    generate_qr_image,
    generate_report_token,
    get_report_pdf_bytes,
    get_report_summary,
    log_qr_scan,
    validate_report_token,
)

router = APIRouter(prefix="/reports", tags=["QR Report Access"])
security = HTTPBearer(auto_error=False)


def _client_host(request: Request) -> str | None:
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        return forwarded_for.split(",", 1)[0].strip()
    return request.client.host if request.client else None


def _public_base_url(request: Request) -> str:
    configured = (
        os.getenv("PUBLIC_BASE_URL")
        or os.getenv("QR_BASE_URL")
        or os.getenv("FRONTEND_BASE_URL")
    )
    if configured:
        return configured.rstrip("/")

    forwarded_proto = request.headers.get("x-forwarded-proto")
    forwarded_host = request.headers.get("x-forwarded-host")
    scheme = (forwarded_proto or request.url.scheme or "http").split(",", 1)[0].strip()
    host = (forwarded_host or request.headers.get("host") or "").split(",", 1)[0].strip()
    if host:
        return f"{scheme}://{host}".rstrip("/")
    return str(request.base_url).rstrip("/")


def _decode_optional_user(credentials: HTTPAuthorizationCredentials | None) -> dict:
    if not credentials:
        return {}
    try:
        return jwt.decode(credentials.credentials, settings.SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        return {}


@router.get("/{report_id}/qr")
async def get_report_qr(
    report_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
):
    """Generate a signed QR PNG for a completed report or job result."""
    current_user = _decode_optional_user(credentials)
    patient_id = current_user.get("patient_id") or request.query_params.get("patient_id")
    token = await generate_report_token(report_id, db, patient_id=patient_id)
    payload = jwt.decode(token, settings.QR_SECRET_KEY or settings.SECRET_KEY, algorithms=[ALGORITHM])
    base_url = _public_base_url(request)
    qr_bytes = generate_qr_image(token, base_url=base_url)

    await log_qr_scan(
        db,
        token_id=payload.get("token_id"),
        report_id=payload.get("report_id"),
        patient_id=payload.get("patient_id"),
        action="qr_generated",
        ip_address=_client_host(request),
        user_agent=request.headers.get("user-agent"),
    )

    return StreamingResponse(
        io.BytesIO(qr_bytes),
        media_type="image/png",
        headers={
            "Content-Disposition": f"inline; filename=report_{report_id[:8]}_qr.png",
            "Cache-Control": "no-store",
            "X-QR-Base-URL": base_url,
        },
    )


@router.get("/preview")
async def report_preview(
    token: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Validate a QR token without consuming it and return preview-card data."""
    payload = await validate_report_token(token, db, burn=False)
    report_id = payload["report_id"]
    summary = await get_report_summary(report_id, db)

    await log_qr_scan(
        db,
        token_id=payload.get("token_id"),
        report_id=report_id,
        patient_id=payload.get("patient_id"),
        action="preview_accessed",
        ip_address=_client_host(request),
        user_agent=request.headers.get("user-agent"),
    )

    return {
        "report_id": report_id,
        "patient_name": summary.get("patient_name", "Patient"),
        "report_date": summary.get("created_at"),
        "report_type": summary.get("report_type"),
        "key_findings": summary.get("key_findings", []),
        "urgency_flag": summary.get("urgency_flag", "routine"),
        "confidence": summary.get("confidence", 0),
        "download_token": token,
        "expires_at": payload.get("exp"),
        "disclaimer": "AI-assisted analysis. NOT a medical diagnosis.",
    }


@router.get("/download")
async def download_report_pdf(
    token: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Consume a one-time QR token and stream the report PDF."""
    payload = await validate_report_token(token, db, burn=True)
    report_id = payload["report_id"]

    await log_qr_scan(
        db,
        token_id=payload.get("token_id"),
        report_id=report_id,
        patient_id=payload.get("patient_id"),
        action="pdf_downloaded",
        ip_address=_client_host(request),
        user_agent=request.headers.get("user-agent"),
    )

    pdf_bytes = await get_report_pdf_bytes(report_id, db)
    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="vaidya_report_{report_id[:8]}.pdf"',
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )
