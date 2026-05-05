"""
Celery job status polling.
Routes can use this to return live task state from Redis backend
without hitting the DB for every poll.
"""

from enum import StrEnum
from celery.result import AsyncResult
from app.workers.celery_app import celery_app


class TaskState(StrEnum):
    PENDING = "PENDING"
    STARTED = "STARTED"
    PROCESSING = "PROCESSING"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    RETRY = "RETRY"
    REVOKED = "REVOKED"


def get_task_status(task_id: str) -> dict:
    """
    Poll Celery result backend for live task state.
    Returns dict safe to include in API response.
    """
    result = AsyncResult(task_id, app=celery_app)

    state = result.state
    info = {}

    if state == "SUCCESS":
        info = result.result or {}
    elif state == "FAILURE":
        info = {"error": str(result.result)}
    elif state in ("STARTED", "PROCESSING"):
        info = result.info or {}

    return {
        "task_id": task_id,
        "state": state,
        "info": info,
    }


def revoke_task(task_id: str, terminate: bool = False) -> bool:
    """Cancel a queued or running task."""
    celery_app.control.revoke(task_id, terminate=terminate)
    return True
