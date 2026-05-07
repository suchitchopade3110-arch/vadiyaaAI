"""Phase 3 patient history helpers for report comparisons and trends."""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.report import Report


def _coerce_uuid(value: str) -> uuid.UUID | None:
    try:
        return uuid.UUID(str(value))
    except (TypeError, ValueError):
        return None


def _parse_jsonish(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return {}
    return value or {}


def _report_to_dict(report: Report) -> dict[str, Any]:
    parsed_data = _parse_jsonish(report.parsed_data)
    analysis_result = _parse_jsonish(report.analysis_result)
    return {
        "report_id": str(report.id),
        "patient_id": str(report.patient_id) if report.patient_id else None,
        "report_type": report.report_type,
        "status": report.status,
        "risk_score": report.risk_score,
        "risk_factors": report.risk_factors or [],
        "anomalies": report.anomalies or [],
        "extracted_entities": parsed_data,
        "analysis_result": analysis_result,
        "explanation": report.explanation,
        "confidence_score": report.confidence_score or report.confidence,
        "uncertainty_flag": report.uncertainty_flag,
        "completed_at": (report.completed_at or report.created_at).isoformat() if (report.completed_at or report.created_at) else None,
        "created_at": report.created_at.isoformat() if report.created_at else None,
    }


async def get_patient_history(patient_id: str, limit: int, db: AsyncSession) -> list[dict[str, Any]]:
    """Return recent reports for a patient from newest to oldest."""
    patient_uuid = _coerce_uuid(patient_id)
    if patient_uuid is None:
        return []

    result = await db.execute(
        select(Report)
        .where(Report.patient_id == patient_uuid)
        .order_by(Report.created_at.desc())
        .limit(max(1, min(int(limit or 10), 50)))
    )
    return [_report_to_dict(report) for report in result.scalars().all()]


def get_patient_history_sync(patient_id: str, limit: int = 5) -> list[dict[str, Any]]:
    """Best-effort sync history lookup for Celery/report pipeline code."""
    patient_uuid = _coerce_uuid(patient_id)
    if patient_uuid is None:
        return []

    try:
        import psycopg2
        import psycopg2.extras

        from app.core.config import settings

        db_url = (
            f"postgresql://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}"
            f"@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
        )
        with psycopg2.connect(db_url) as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute(
                    """
                    SELECT id, patient_id, report_type, status, parsed_data, risk_score,
                           risk_factors, anomalies, explanation, confidence_score,
                           confidence, uncertainty_flag, analysis_result, created_at,
                           completed_at
                    FROM reports
                    WHERE patient_id = %s
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (str(patient_uuid), max(1, min(int(limit or 5), 50))),
                )
                rows = cursor.fetchall()
    except Exception:
        return []

    history = []
    for row in rows:
        parsed_data = _parse_jsonish(row.get("parsed_data"))
        history.append(
            {
                "report_id": str(row.get("id")),
                "patient_id": str(row.get("patient_id")) if row.get("patient_id") else None,
                "report_type": row.get("report_type"),
                "status": row.get("status"),
                "risk_score": row.get("risk_score"),
                "risk_factors": row.get("risk_factors") or [],
                "anomalies": row.get("anomalies") or [],
                "extracted_entities": parsed_data,
                "analysis_result": _parse_jsonish(row.get("analysis_result")),
                "explanation": row.get("explanation"),
                "confidence_score": row.get("confidence_score") or row.get("confidence"),
                "uncertainty_flag": row.get("uncertainty_flag"),
                "completed_at": _iso(row.get("completed_at") or row.get("created_at")),
                "created_at": _iso(row.get("created_at")),
            }
        )
    return history


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _lab_map(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    entities = report.get("extracted_entities") or {}
    rows = report.get("lab_values") or entities.get("lab_values") or []
    output = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = str(row.get("field") or row.get("test") or row.get("name") or "").lower().strip()
        if name:
            output[name] = row
    return output


def _number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def compare_reports(current: dict[str, Any], previous: dict[str, Any]) -> dict[str, Any]:
    """Compare two report payloads and return risk and lab-value deltas."""
    current_risk = _number(current.get("risk_score"))
    previous_risk = _number(previous.get("risk_score"))
    risk_delta = (
        round(current_risk - previous_risk, 2)
        if current_risk is not None and previous_risk is not None
        else None
    )

    current_labs = _lab_map(current)
    previous_labs = _lab_map(previous)
    lab_deltas = []
    for name, current_row in current_labs.items():
        previous_row = previous_labs.get(name)
        if not previous_row:
            continue
        current_value = _number(current_row.get("value", current_row.get("result")))
        previous_value = _number(previous_row.get("value", previous_row.get("result")))
        if current_value is None or previous_value is None:
            continue
        delta = round(current_value - previous_value, 3)
        lab_deltas.append(
            {
                "test": current_row.get("field") or current_row.get("test") or name,
                "current": current_value,
                "previous": previous_value,
                "delta": delta,
                "direction": "up" if delta > 0 else "down" if delta < 0 else "unchanged",
                "unit": current_row.get("unit") or previous_row.get("unit") or "",
            }
        )

    return {
        "current_report_id": current.get("report_id") or current.get("job_id"),
        "previous_report_id": previous.get("report_id") or previous.get("job_id"),
        "risk_delta": risk_delta,
        "risk_direction": "up" if risk_delta and risk_delta > 0 else "down" if risk_delta and risk_delta < 0 else "unchanged",
        "lab_deltas": lab_deltas,
        "summary": _comparison_summary(risk_delta, lab_deltas),
    }


def _comparison_summary(risk_delta: float | None, lab_deltas: list[dict[str, Any]]) -> str:
    if risk_delta is None and not lab_deltas:
        return "No comparable prior values found."
    pieces = []
    if risk_delta is not None:
        pieces.append(f"Risk changed by {risk_delta:+.1f} points.")
    if lab_deltas:
        pieces.append(f"{len(lab_deltas)} lab values changed since the prior report.")
    return " ".join(pieces)


async def get_longitudinal_trend(patient_id: str, test_name: str, db: AsyncSession) -> dict[str, Any]:
    """Return time-series values for one lab test."""
    history = await get_patient_history(patient_id, limit=50, db=db)
    target = str(test_name).lower().strip()
    points = []
    for report in reversed(history):
        row = _lab_map(report).get(target)
        if not row:
            continue
        value = _number(row.get("value", row.get("result")))
        if value is None:
            continue
        points.append(
            {
                "report_id": report.get("report_id"),
                "date": report.get("completed_at") or report.get("created_at"),
                "value": value,
                "unit": row.get("unit", ""),
            }
        )
    return {"patient_id": patient_id, "test_name": test_name, "points": points, "count": len(points)}


def compare_with_latest_history(current: dict[str, Any], history: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Compare current report to the newest prior history item."""
    if not history:
        return None
    return compare_reports(current, history[0])
