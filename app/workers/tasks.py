"""
VaidyaAI Celery Tasks Base
Shared logic for all pipeline tasks.
"""

import logging
from celery import Task
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from app.db.base import Base  # noqa: F401
from app.workers.celery_app import celery_app
from app.core.config import settings

log = logging.getLogger(__name__)

# Sync engine for Celery (workers are sync; asyncpg won't work here)
_SYNC_DB_URL = settings.DATABASE_URL.replace("+asyncpg", "+psycopg2")
_engine = create_engine(_SYNC_DB_URL, pool_pre_ping=True, pool_size=5)
SyncSession = sessionmaker(bind=_engine, expire_on_commit=False)


def get_sync_db() -> Session:
    return SyncSession()


class VaidyaTask(Task):
    abstract = True

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        log.error(f"Task {self.name}[{task_id}] failed: {exc}")
