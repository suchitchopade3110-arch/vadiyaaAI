"""
VaidyaAI — DB Persistence Helper for Celery Workers
Saves task results to PostgreSQL so data outlives Redis cache (1hr expiry).

Called at the END of every Celery task after all processing is done.
Uses synchronous psycopg2 — Celery workers are sync, not async.
"""

import json
import logging
from datetime import UTC, datetime
from uuid import UUID

logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _get_conn():
    """Sync PostgreSQL connection for Celery workers."""
    import psycopg2
    from app.core.config import settings
    
    # Construct sync URL from settings
    db_url = f"postgresql://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
    return psycopg2.connect(db_url)


def persist_claim(result: dict) -> bool:
    """
    Upsert claim result into claims table.
    Called by: claim_tasks.verify_claim() on success.
    """
    try:
        conn = _get_conn()
        cur  = conn.cursor()

        cur.execute("""
            UPDATE claims SET
                status                 = %s,
                extracted_entities     = %s,
                retrieved_sources      = %s,
                source_count           = %s,
                verdict                = %s,
                explanation            = %s,
                confidence             = %s,
                uncertainty_flag       = %s,
                hallucination_detected = %s,
                hallucination_details  = %s,
                shap_values            = %s,
                disclaimer             = %s,
                completed_at           = %s
            WHERE id = %s
        """, (
            # Map finished status to the verdict for claims
            result.get("verdict", "UNCERTAIN").upper(),
            json.dumps(result.get("extracted_entities", {})),
            json.dumps(result.get("sources", [])),
            result.get("source_count", 0),
            result.get("verdict"),
            result.get("explanation"),
            result.get("confidence_score"),
            result.get("uncertainty_flag", False),
            result.get("hallucination_detected", False),
            json.dumps(result.get("hallucination_details", {})),
            json.dumps(result.get("shap_values", {})),
            result.get("medical_disclaimer"),
            _utc_now(),
            result["claim_id"],
        ))

        conn.commit()
        cur.close()
        conn.close()
        logger.info(f"Claim {result['claim_id']} persisted to DB")
        return True

    except Exception as e:
        logger.error(f"DB persist failed for claim {result.get('claim_id')}: {e}")
        return False   # Non-fatal — Redis result still available


def persist_report(result: dict) -> bool:
    """
    Upsert report result into reports table.
    Called by: report_tasks.analyze_report() on success.
    """
    try:
        conn = _get_conn()
        cur  = conn.cursor()

        cur.execute("""
            UPDATE reports SET
                status             = %s,
                raw_text           = %s,
                parsed_data        = %s,
                risk_score         = %s,
                risk_factors       = %s,
                shap_values        = %s,
                anomalies          = %s,
                explanation        = %s,
                retrieved_sources  = %s,
                confidence_score   = %s,
                uncertainty_flag   = %s,
                medical_disclaimer = %s,
                completed_at       = %s
            WHERE id = %s
        """, (
            "COMPLETE",
            result.get("raw_text_preview"),
            json.dumps(result.get("extracted_entities", {})),
            result.get("risk_score"),
            json.dumps(result.get("risk_factors", [])),
            json.dumps(result.get("shap_values", {})),
            json.dumps(result.get("anomalies", [])),
            result.get("explanation"),
            json.dumps(result.get("sources", [])),
            result.get("confidence_score"),
            result.get("uncertainty_flag", False),
            result.get("medical_disclaimer"),
            _utc_now(),
            result["report_id"],
        ))

        conn.commit()
        cur.close()
        conn.close()
        logger.info(f"Report {result['report_id']} persisted to DB")
        return True

    except Exception as e:
        logger.error(f"DB persist failed for report {result.get('report_id')}: {e}")
        return False


def persist_image_analysis(result: dict) -> bool:
    """
    Upsert image analysis result into image_analyses table.
    Called by: image_tasks.analyze_image() on success.
    """
    try:
        conn = _get_conn()
        cur  = conn.cursor()

        seg = result.get("segmentation", {})
        cls = result.get("classification", {})
        gcam = result.get("gradcam", {})

        cur.execute("""
            UPDATE image_analyses SET
                status                   = %s,
                dicom_metadata           = %s,
                segmentation_mask_path   = %s,
                segmentation_overlay     = %s,
                roi_metadata             = %s,
                segmentation_confidence  = %s,
                classification           = %s,
                classification_probs     = %s,
                gradcam_path             = %s,
                gradcam_regions          = %s,
                explanation              = %s,
                retrieved_sources        = %s,
                confidence               = %s,
                uncertainty_flag         = %s,
                anomaly_detected         = %s,
                medical_disclaimer       = %s,
                completed_at             = %s
            WHERE id = %s
        """, (
            "COMPLETE",
            json.dumps(result.get("dicom_metadata", {})),
            seg.get("mask_path"),
            seg.get("overlay_path"),
            json.dumps(seg.get("roi_bounding_box", {})),
            seg.get("confidence"),
            cls.get("top_class"),
            json.dumps(cls.get("probabilities", {})),
            gcam.get("heatmap_path"),
            json.dumps(gcam.get("top_regions", [])),
            result.get("explanation"),
            json.dumps(result.get("sources", [])),
            result.get("confidence_score"),
            result.get("uncertainty_flag", True),
            result.get("anomaly_detected", False),
            result.get("medical_disclaimer"),
            _utc_now(),
            result["analysis_id"],
        ))

        conn.commit()
        cur.close()
        conn.close()
        logger.info(f"ImageAnalysis {result['analysis_id']} persisted to DB")
        return True

    except Exception as e:
        logger.error(f"DB persist failed for image {result.get('analysis_id')}: {e}")
        return False


def mark_failed(table: str, record_id: str, error: str) -> bool:
    """
    Mark a record as failed in DB when task crashes.
    table: 'claims' | 'reports' | 'image_analyses'
    """
    try:
        conn = _get_conn()
        cur  = conn.cursor()
        cur.execute(
            f"UPDATE {table} SET status = 'FAILED', completed_at = %s WHERE id = %s",
            (_utc_now(), record_id)
        )
        conn.commit()
        cur.close()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"mark_failed error for {table}/{record_id}: {e}")
        return False


# ── INSERT on submission (routes call these) ──────────────────────────────────

def insert_claim(claim_id: str, claim_text: str, task_id: str, patient_id: str = None) -> bool:
    """Insert pending claim row when route receives submission."""
    try:
        conn = _get_conn()
        cur  = conn.cursor()
        cur.execute("""
            INSERT INTO claims (id, claim_text, status, celery_task_id, created_at, uncertainty_flag, hallucination_detected)
            VALUES (%s, %s, 'PENDING', %s, %s, FALSE, FALSE)
            ON CONFLICT (id) DO NOTHING
        """, (claim_id, claim_text, task_id, _utc_now()))
        conn.commit()
        cur.close()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"insert_claim failed: {e}")
        return False


def insert_report(report_id: str, report_type: str, file_path: str,
                  file_format: str, task_id: str, patient_id: str = None) -> bool:
    """Insert pending report row when route receives submission."""
    try:
        conn = _get_conn()
        cur  = conn.cursor()
        cur.execute("""
            INSERT INTO reports (id, report_type, file_path, file_format, status, celery_task_id, created_at, uncertainty_flag)
            VALUES (%s, %s, %s, %s, 'PENDING', %s, %s, FALSE)
            ON CONFLICT (id) DO NOTHING
        """, (report_id, report_type, file_path, file_format, task_id, _utc_now()))
        conn.commit()
        cur.close()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"insert_report failed: {e}")
        return False


def insert_image_analysis(analysis_id: str, image_type: str, file_path: str,
                          file_format: str, task_id: str, patient_id: str = None) -> bool:
    """Insert pending image_analysis row when route receives submission."""
    try:
        conn = _get_conn()
        cur  = conn.cursor()
        cur.execute("""
            INSERT INTO image_analyses (id, image_type, file_path, file_format, status, celery_task_id, created_at, uncertainty_flag, anomaly_detected)
            VALUES (%s, %s, %s, %s, 'PENDING', %s, %s, FALSE, FALSE)
            ON CONFLICT (id) DO NOTHING
        """, (analysis_id, image_type, file_path, file_format, task_id, _utc_now()))
        conn.commit()
        cur.close()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"insert_image_analysis failed: {e}")
        return False
