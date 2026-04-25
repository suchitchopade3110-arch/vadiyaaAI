import uuid
from sqlalchemy import String, JSON, DateTime, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID
from app.db.base_class import Base


class Patient(Base):
    __tablename__ = "patients"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    # Anonymized — no real PII stored in v1
    demographics: Mapped[dict] = mapped_column(JSON, nullable=True)
    medical_history: Mapped[dict] = mapped_column(JSON, nullable=True)
    created_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    claims: Mapped[list["Claim"]] = relationship(back_populates="patient")
    reports: Mapped[list["Report"]] = relationship(back_populates="patient")
    image_analyses: Mapped[list["ImageAnalysis"]] = relationship(back_populates="patient")
