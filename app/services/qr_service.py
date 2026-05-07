import io
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import qrcode
from celery.result import AsyncResult
from fastapi import HTTPException
from jose import JWTError, jwt
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.patient import Patient
from app.models.qr_access import QRAuditLog, QRToken
from app.models.report import Report
from app.services.pdf_report import generate_report_pdf
from app.workers.celery_app import celery_app

QR_EXPIRY_MINUTES = 30
ONE_TIME_SCAN = True
ALGORITHM = "HS256"


def _qr_secret() -> str:
    return settings.QR_SECRET_KEY or settings.SECRET_KEY


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _to_percent(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return round(number * 100, 1) if 0 <= number <= 1 else round(number, 1)


async def find_report(report_id: str, db: AsyncSession) -> Report | None:
    filters = [Report.celery_task_id == report_id]
    try:
        filters.append(Report.id == uuid.UUID(report_id))
    except ValueError:
        pass

    result = await db.execute(select(Report).where(or_(*filters)))
    return result.scalar_one_or_none()


async def generate_report_token(report_id: str, db: AsyncSession, patient_id: str | None = None) -> str:
    report = await find_report(report_id, db)
    resolved_report_id = str(report.id) if report else report_id
    resolved_patient_id = patient_id or (str(report.patient_id) if report and report.patient_id else None)

    token_id = str(uuid.uuid4())
    expires_at = _now() + timedelta(minutes=QR_EXPIRY_MINUTES)
    payload = {
        "token_id": token_id,
        "report_id": resolved_report_id,
        "patient_id": resolved_patient_id,
        "exp": expires_at,
        "iat": _now(),
    }
    token = jwt.encode(payload, _qr_secret(), algorithm=ALGORITHM)

    db.add(QRToken(
        id=token_id,
        report_id=resolved_report_id,
        patient_id=resolved_patient_id,
        token_hash=token[-16:],
        expires_at=expires_at,
        scan_count=0,
        is_active=True,
    ))
    await db.commit()
    return token


async def validate_report_token(token: str, db: AsyncSession, *, burn: bool) -> dict[str, Any]:
    try:
        payload = jwt.decode(token, _qr_secret(), algorithms=[ALGORITHM])
    except JWTError as exc:
        raise HTTPException(status_code=401, detail="Invalid or expired QR token") from exc

    token_id = payload.get("token_id")
    result = await db.execute(select(QRToken).where(QRToken.id == token_id))
    db_token = result.scalar_one_or_none()

    if not db_token:
        raise HTTPException(status_code=404, detail="QR token not found")
    if not db_token.is_active:
        raise HTTPException(status_code=410, detail="QR code already used. Request a new one.")
    if db_token.expires_at < _now():
        db_token.is_active = False
        await db.commit()
        raise HTTPException(status_code=410, detail="QR code expired. Request a new one.")

    if burn and ONE_TIME_SCAN:
        db_token.scan_count += 1
        db_token.is_active = False
        await db.commit()

    return payload


def generate_qr_image(token: str, base_url: str) -> bytes:
    preview_url = f"{base_url}/ui/report-preview.html?token={token}"
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=10,
        border=4,
    )
    qr.add_data(preview_url)
    qr.make(fit=True)
    image = qr.make_image(fill_color="black", back_color="white")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


async def log_qr_scan(
    db: AsyncSession,
    *,
    token_id: str | None,
    report_id: str | None,
    patient_id: str | None,
    action: str,
    ip_address: str | None,
    user_agent: str | None,
) -> None:
    db.add(QRAuditLog(
        token_id=token_id,
        report_id=report_id,
        patient_id=patient_id,
        action=action,
        ip_address=ip_address,
        user_agent=user_agent,
    ))
    await db.commit()


async def get_report_summary(report_id: str, db: AsyncSession) -> dict[str, Any]:
    report = await find_report(report_id, db)
    patient_name = "Patient"
    if report and report.patient_id:
        patient_result = await db.execute(select(Patient).where(Patient.id == report.patient_id))
        patient = patient_result.scalar_one_or_none()
        patient_name = patient.patient_code if patient else str(report.patient_id)

    data = await get_report_payload(report_id, db)
    risk_score = _to_percent(data.get("risk_score", data.get("confidence_score", 0)))
    findings = _key_findings(data)
    return {
        "patient_name": patient_name,
        "created_at": (
            data.get("generated_at")
            or ((report.completed_at or report.created_at).isoformat() if report and (report.completed_at or report.created_at) else None)
        ),
        "report_type": data.get("report_type") or (report.report_type if report else "report"),
        "key_findings": findings[:3],
        "urgency_flag": "urgent" if risk_score >= 80 else "follow_up" if risk_score >= 55 else "routine",
        "confidence": _to_percent(data.get("confidence_score", data.get("confidence", risk_score))),
    }


async def get_report_pdf_bytes(report_id: str, db: AsyncSession) -> bytes:
    return generate_report_pdf(await get_report_payload(report_id, db))


async def get_report_payload(report_id: str, db: AsyncSession) -> dict[str, Any]:
    celery_result = AsyncResult(report_id, app=celery_app)
    if celery_result.ready() and not celery_result.failed() and isinstance(celery_result.result, dict):
        return _coerce_report_data(celery_result.result)

    report = await find_report(report_id, db)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")

    raw = {
        "report_id": str(report.id),
        "report_type": report.report_type,
        "patient_id": str(report.patient_id) if report.patient_id else "Anonymous",
        "risk_score": report.risk_score,
        "risk_label": (report.analysis_result or {}).get("risk_label", "unknown"),
        "confidence_score": report.confidence_score or report.confidence or report.risk_score or 0,
        "lab_values": (report.parsed_data or {}).get("lab_values", []),
        "anomalies": report.anomalies or [],
        "shap_values": report.shap_values or {},
        "risk_factors": report.risk_factors or [],
        "explanation": report.explanation or "",
        "sources": report.source_citations or report.retrieved_sources or [],
        "uncertainty_flag": report.uncertainty_flag,
        "completed_at": (report.completed_at or report.created_at).isoformat() if report.completed_at or report.created_at else "",
    }
    return _coerce_report_data(raw)


def _coerce_report_data(data: dict[str, Any]) -> dict[str, Any]:
    entities = data.get("extracted_entities") or {}
    classification = data.get("image_classification") or data.get("classification") or {}
    lab_values = data.get("lab_values") or entities.get("lab_values") or []
    risk_score = _to_percent(data.get("risk_score", data.get("confidence_score", data.get("confidence", 0))))
    confidence = _to_percent(data.get("confidence_score", data.get("confidence", risk_score)))

    return {
        "report_type": data.get("report_type") or data.get("image_type") or data.get("type") or "report",
        "patient_id": data.get("patient_id", "Anonymous"),
        "risk_score": risk_score,
        "risk_label": data.get("risk_label") or data.get("risk_level") or classification.get("primary_finding") or "unknown",
        "confidence_score": confidence,
        "lab_values": _normalize_lab_values(lab_values),
        "anomalies": data.get("anomalies") or _findings_as_anomalies(data.get("findings") or classification.get("all_findings") or []),
        "shap_factors": data.get("shap_values") or _risk_factors_as_shap(data.get("risk_factors") or []),
        "explanation": data.get("explanation") or data.get("impression") or "",
        "citations": data.get("sources") or data.get("source_citations") or data.get("citations") or [],
        "uncertainty_flag": data.get("uncertainty_flag") or data.get("uncertainty", False),
        "generated_at": data.get("completed_at") or data.get("generated_at") or "",
    }


def _normalize_lab_values(lab_values: Any) -> list[dict[str, Any]]:
    if isinstance(lab_values, dict):
        lab_values = [{"field": key, **value} if isinstance(value, dict) else {"field": key, "value": value} for key, value in lab_values.items()]
    rows = []
    for item in lab_values or []:
        if not isinstance(item, dict):
            continue
        rows.append({
            "field": item.get("field") or item.get("test") or item.get("name") or "",
            "value": item.get("value", item.get("result", "")),
            "unit": item.get("unit", ""),
            "reference": item.get("reference") or item.get("reference_range") or item.get("ref") or "-",
            "flag": item.get("flag") or item.get("severity") or ("ABNORMAL" if item.get("abnormal") else "NORMAL"),
        })
    return rows


def _key_findings(data: dict[str, Any]) -> list[dict[str, Any]]:
    findings = []
    for anomaly in data.get("anomalies") or []:
        if not isinstance(anomaly, dict):
            continue
        findings.append({
            "name": anomaly.get("field") or anomaly.get("test") or anomaly.get("label") or "Finding",
            "value": anomaly.get("value", ""),
            "unit": anomaly.get("unit", ""),
            "reference_range": anomaly.get("reference") or anomaly.get("reference_range") or "-",
            "abnormal": True,
        })
    if findings:
        return findings
    return [
        {
            "name": row.get("field"),
            "value": row.get("value"),
            "unit": row.get("unit"),
            "reference_range": row.get("reference"),
            "abnormal": str(row.get("flag", "")).upper() != "NORMAL",
        }
        for row in _normalize_lab_values(data.get("lab_values"))
    ]


def _findings_as_anomalies(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "field": item.get("label", "Finding"),
            "value": f"{_to_percent(item.get('confidence'))}%",
            "unit": "",
            "reference": "-",
            "severity": item.get("severity", "MODERATE"),
            "explanation": item.get("clinical_meaning") or item.get("clinicalMeaning") or "",
        }
        for item in findings
        if isinstance(item, dict) and (item.get("detected", True) is True)
    ]


def _risk_factors_as_shap(factors: list[Any]) -> dict[str, float]:
    result = {}
    for index, item in enumerate(factors):
        if isinstance(item, dict):
            key = item.get("feature") or item.get("factor") or f"factor_{index + 1}"
            result[str(key)] = float(item.get("shap", item.get("score", 0)) or 0)
        else:
            result[str(item)] = 0.0
    return result
