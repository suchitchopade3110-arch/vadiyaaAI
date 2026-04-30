"""Compatibility settings import for the project skeleton.

The canonical settings live in app.core.config. This module keeps the shorter
`app.config` path from the skeleton available without duplicating config state.
"""

from app.core.config import settings as _settings


class SettingsProxy:
    def __getattr__(self, name: str):
        return getattr(_settings, name)

    @property
    def DB_URL(self) -> str:
        return _settings.DATABASE_URL

    @property
    def MAX_FILE_SIZE_MB(self) -> int:
        return getattr(_settings, "MAX_UPLOAD_SIZE_MB", 50)

    @property
    def DEBUG(self) -> bool:
        return getattr(_settings, "APP_ENV", "development") == "development"


settings = SettingsProxy()

