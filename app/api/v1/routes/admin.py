from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException

from app.ml.audit.run_bias_audit import JSON_PATH, run_bias_audit, save_report


router = APIRouter()


@router.get("/bias-audit")
def get_bias_audit_report():
    """Return the latest bias audit report for the admin dashboard."""
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
