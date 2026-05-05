"""Compatibility import for `celery -A workers.celery_app worker`."""

from app.workers.celery_app import celery_app

__all__ = ["celery_app"]

