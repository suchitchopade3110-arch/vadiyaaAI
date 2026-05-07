import uuid

from sqlalchemy import Boolean, DateTime, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base_class import Base


class QRToken(Base):
    """Tracks QR report tokens for expiry and one-time download enforcement."""

    __tablename__ = "qr_tokens"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    report_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    patient_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    token_hash: Mapped[str] = mapped_column(String(16), nullable=False)
    expires_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    scan_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, index=True)
    created_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class QRAuditLog(Base):
    """Immutable audit trail for QR generation, preview, download, and rejection events."""

    __tablename__ = "qr_audit_logs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    token_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    report_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    patient_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    action: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    ip_address: Mapped[str | None] = mapped_column(String(45), nullable=True)
    user_agent: Mapped[str | None] = mapped_column(Text, nullable=True)
    timestamp: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)
