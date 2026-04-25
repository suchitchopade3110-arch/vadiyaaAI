from celery import Celery
from app.core.config import settings

celery_app = Celery("vaidyaai")

celery_app.config_from_object({
    "broker_url": settings.CELERY_BROKER_URL,
    "result_backend": settings.CELERY_RESULT_BACKEND,
    "task_serializer": "json",
    "result_serializer": "json",
    "accept_content": ["json"],
    "timezone": "UTC",
    "enable_utc": True,
    "task_track_started": True,
    "task_acks_late": True,            # ack only after task completes → safe retry on crash
    "worker_prefetch_multiplier": 1,   # one task/worker → GPU-safe
    "task_soft_time_limit": 120,       # warn at 2min
    "task_time_limit": 180,            # hard kill at 3min
    "result_expires": 86400,           # results kept 24h in Redis
})

# Auto-discover tasks
celery_app.autodiscover_tasks(["app.workers"])
