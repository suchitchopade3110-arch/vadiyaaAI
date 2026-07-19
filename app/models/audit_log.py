import uuid

from sqlalchemy import DateTime, JSON, String, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base_class import Base


class AuditLog(Base):
    """Immutable HIPAA-technical-safeguard access log.

    Append-only: no application code path should ever UPDATE or DELETE a row
    here. Stores references (resource IDs), never PHI content — the claim
    text, report contents, and image data live in their own tables; this
    table only records who touched what, when.
    """

    __tablename__ = "audit_logs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    timestamp: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), nullable=True, index=True
    )
    action: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    resource_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    resource_id: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)
    patient_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), nullable=True, index=True
    )
    ip_address: Mapped[str | None] = mapped_column(String(45), nullable=True)
    user_agent: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    details: Mapped[dict | None] = mapped_column(JSON, nullable=True)
