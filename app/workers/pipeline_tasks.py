"""Celery tasks that dispatch to the Phase 2 pipeline orchestrator."""

import logging
from typing import Optional

from celery import Task

from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


def _retry_delay(retries: int) -> int:
    """Exponential backoff: 10s, 40s, 160s."""
    return 10 * (4 ** retries)


class PipelineTask(Task):
    abstract = True

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        logger.error("[%s] Pipeline task failed: %s", task_id, exc)


@celery_app.task(
    bind=True,
    base=PipelineTask,
    name="app.workers.pipeline_tasks.analyze_report_task",
    max_retries=3,
)
def analyze_report_task(
    self,
    file_path: str,
    patient_id: Optional[str],
    report_type: str,
    gender: str = "male",
    age: int = 40,
    explanation_mode: str = "brief",
):
    """Run report pipeline asynchronously."""
    job_id = self.request.id
    logger.info("[%s] Report task started: type=%s", job_id, report_type)
    try:
        from app.pipeline import run_report_pipeline, to_json_safe

        result = run_report_pipeline(
            file_path=file_path,
            patient_id=patient_id,
            report_type=report_type,
            job_id=job_id,
            gender=gender,
            age=age,
            explanation_mode=explanation_mode,
        )
        return to_json_safe(result.model_dump())
    except Exception as exc:
        logger.error("[%s] Report task failed: %s", job_id, exc)
        raise self.retry(exc=exc, countdown=_retry_delay(self.request.retries))


@celery_app.task(
    bind=True,
    base=PipelineTask,
    name="app.workers.pipeline_tasks.analyze_image_task",
    max_retries=3,
)
def analyze_image_task(
    self,
    file_path: str,
    image_type: str,
    patient_id: Optional[str],
    clinical_context: str = "",
):
    """Run image pipeline asynchronously."""
    job_id = self.request.id
    logger.info("[%s] Image task started: type=%s", job_id, image_type)
    try:
        from app.pipeline import run_image_pipeline, to_json_safe

        result = run_image_pipeline(
            image_path=file_path,
            image_type=image_type,
            patient_id=patient_id,
            job_id=job_id,
            clinical_context=clinical_context,
        )
        return to_json_safe(result.model_dump())
    except Exception as exc:
        logger.error("[%s] Image task failed: %s", job_id, exc)
        raise self.retry(exc=exc, countdown=_retry_delay(self.request.retries))


@celery_app.task(
    bind=True,
    base=PipelineTask,
    name="app.workers.pipeline_tasks.verify_claim_task",
    max_retries=3,
)
def verify_claim_task(self, claim_text: str, patient_id: Optional[str]):
    """Run claim verification pipeline asynchronously."""
    job_id = self.request.id
    logger.info("[%s] Claim task started", job_id)
    try:
        from app.pipeline import run_claim_pipeline, to_json_safe

        result = run_claim_pipeline(
            claim_text=claim_text,
            patient_id=patient_id,
            job_id=job_id,
        )
        return to_json_safe(result.model_dump())
    except Exception as exc:
        logger.error("[%s] Claim task failed: %s", job_id, exc)
        raise self.retry(exc=exc, countdown=_retry_delay(self.request.retries))
