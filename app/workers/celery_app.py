from celery import Celery
from app.core.config import settings

celery_app = Celery(
    "vaidyaai",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=[
        "app.workers.claim_tasks",
        "app.workers.image_tasks",
        "app.workers.report_tasks",
        "app.workers.pipeline_tasks",
    ]
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_soft_time_limit=settings.TASK_SOFT_TIME_LIMIT,
    task_time_limit=settings.TASK_HARD_TIME_LIMIT,
    task_max_retries=3,
    task_default_retry_delay=5,
    result_expires=86400,
    worker_prefetch_multiplier=1,
    timezone="UTC",
    task_routes={
        "app.workers.claim_tasks.*": {"queue": "claims"},
        "app.workers.image_tasks.*": {"queue": "images"},
        "app.workers.report_tasks.*": {"queue": "reports"},
        "app.workers.pipeline_tasks.analyze_report_task": {"queue": "reports"},
        "app.workers.pipeline_tasks.analyze_image_task": {"queue": "images"},
        "app.workers.pipeline_tasks.verify_claim_task": {"queue": "claims"},
    }
)
