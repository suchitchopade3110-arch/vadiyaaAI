from dataclasses import dataclass, field
from datetime import datetime, UTC
from typing import Any


@dataclass
class AsyncJob:
    id: str
    status: str = "pending"
    progress: float = 0.0
    result: dict[str, Any] | None = None
    error_message: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
