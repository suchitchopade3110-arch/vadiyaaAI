"""Compatibility DB import for skeleton routes."""

from app.db.session import AsyncSessionLocal, engine, get_db

__all__ = ["AsyncSessionLocal", "engine", "get_db"]

