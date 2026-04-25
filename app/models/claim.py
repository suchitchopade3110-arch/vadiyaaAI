import uuid
from sqlalchemy import String, Float, JSON, DateTime, ForeignKey, func, Enum
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID
from app.db.base_class import Base
import enum


class ClaimStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    VERIFIED = "verified"
    REFUTED = "refuted"
    UNCERTAIN = "uncertain"
    FAILED = "failed"


class Claim(Base):
    __tablename__ = "claims"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    patient_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("patients.id"), nullable=True
    )
    claim_text: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[ClaimStatus] = mapped_column(
        Enum(ClaimStatus), default=ClaimStatus.PENDING
    )
    confidence: Mapped[float] = mapped_column(Float, nullable=True)  # 0–100 Platt-scaled
    extracted_entities: Mapped[dict] = mapped_column(JSON, nullable=True)   # ClinicalBERT output
    source_citations: Mapped[list] = mapped_column(JSON, nullable=True)     # ChromaDB hits
    shap_values: Mapped[dict] = mapped_column(JSON, nullable=True)
    disclaimer: Mapped[str] = mapped_column(String, nullable=True)
    uncertainty_flag: Mapped[bool] = mapped_column(default=False)
    celery_task_id: Mapped[str] = mapped_column(String, nullable=True)
    created_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    patient: Mapped["Patient"] = relationship(back_populates="claims")
