from __future__ import annotations

import json
import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import require_role
from app.db.session import get_db
from app.ml.audit.run_bias_audit import JSON_PATH, run_bias_audit, save_report
from app.models.audit_log import AuditLog


router = APIRouter(dependencies=[Depends(require_role("admin"))])


@router.get("/bias-audit")
def get_bias_audit_report():
    """Return the latest bias audit report for the admin dashboard.

    Requires the `admin` role.
    """
    try:
        if not JSON_PATH.exists():
            report = run_bias_audit()
            save_report(report)

        return json.loads(JSON_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unable to load bias audit report: {exc}",
        ) from exc


def _parse_uuid(value: str | None, field: str) -> uuid.UUID | None:
    if value is None:
        return None
    try:
        return uuid.UUID(value)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid {field}: not a UUID") from exc


@router.get("/audit-log", summary="Query the HIPAA access-log audit trail")
async def query_audit_log(
    user_id: str | None = None,
    patient_id: str | None = None,
    resource_type: str | None = None,
    resource_id: str | None = None,
    action: str | None = None,
    status: str | None = None,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    limit: int = 50,
    db: AsyncSession = Depends(get_db),
):
    """Filter the append-only audit_logs table by user, resource, or date range.

    Requires the `admin` role. Returns references (resource IDs) only — this
    table never stores claim text, report contents, or image bytes.
    """
    if limit < 1 or limit > 500:
        raise HTTPException(status_code=422, detail="limit must be between 1 and 500")

    query = select(AuditLog).order_by(AuditLog.timestamp.desc()).limit(limit)

    parsed_user_id = _parse_uuid(user_id, "user_id")
    if parsed_user_id is not None:
        query = query.where(AuditLog.user_id == parsed_user_id)

    parsed_patient_id = _parse_uuid(patient_id, "patient_id")
    if parsed_patient_id is not None:
        query = query.where(AuditLog.patient_id == parsed_patient_id)

    if resource_type is not None:
        query = query.where(AuditLog.resource_type == resource_type)
    if resource_id is not None:
        query = query.where(AuditLog.resource_id == resource_id)
    if action is not None:
        query = query.where(AuditLog.action == action)
    if status is not None:
        query = query.where(AuditLog.status == status)
    if start_date is not None:
        query = query.where(AuditLog.timestamp >= start_date)
    if end_date is not None:
        query = query.where(AuditLog.timestamp <= end_date)

    result = await db.execute(query)
    rows = result.scalars().all()

    return {
        "count": len(rows),
        "entries": [
            {
                "id": str(row.id),
                "timestamp": row.timestamp.isoformat() if row.timestamp else None,
                "user_id": str(row.user_id) if row.user_id else None,
                "action": row.action,
                "resource_type": row.resource_type,
                "resource_id": row.resource_id,
                "patient_id": str(row.patient_id) if row.patient_id else None,
                "ip_address": row.ip_address,
                "user_agent": row.user_agent,
                "status": row.status,
                "details": row.details,
            }
            for row in rows
        ],
    }
