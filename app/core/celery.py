"""
VaidyaAI — Celery Worker Configuration
Async job queue for long-running ML tasks.
"""

from celery import Celery
from app.core.config import settings

# ── Celery App ────────────────────────────────────────────────────────────────
celery_app = Celery(
    "vaidyaai",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=[
        "app.workers.claim_tasks",
        "app.workers.image_tasks",
        "app.workers.report_tasks",
    ]
)

# ── Configuration ─────────────────────────────────────────────────────────────
celery_app.conf.update(
    # Serialization
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    
    # Timeouts
    task_soft_time_limit=settings.TASK_SOFT_TIME_LIMIT,
    task_time_limit=settings.TASK_HARD_TIME_LIMIT,
    
    # Retry policy
    task_max_retries=3,
    task_default_retry_delay=5,   # seconds
    
    # Result expiry
    result_expires=3600,   # 1 hour
    
    # Routing — separate queues by task type
    task_routes={
        "app.workers.claim_tasks.*":  {"queue": "claims"},
        "app.workers.image_tasks.*":  {"queue": "images"},   # GPU queue
        "app.workers.report_tasks.*": {"queue": "reports"},
    },
    
    # Concurrency
    worker_prefetch_multiplier=1,   # One task at a time per worker
    
    # Beat schedule (Phase 3: periodic tasks)
    beat_schedule={},
    timezone="UTC",
)
