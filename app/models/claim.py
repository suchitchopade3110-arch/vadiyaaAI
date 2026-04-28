import uuid
from sqlalchemy import String, Float, JSON, DateTime, ForeignKey, func, Enum, Integer, Text, Boolean
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID
from app.db.base_class import Base
import enum


class ClaimStatus(str, enum.Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    VERIFIED = "VERIFIED"
    REFUTED = "REFUTED"
    UNCERTAIN = "UNCERTAIN"
    FAILED = "FAILED"


class Claim(Base):
    __tablename__ = "claims"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    patient_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("patients.id"), nullable=True
    )
    claim_text: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String, default="PENDING")
    confidence: Mapped[float] = mapped_column(Float, nullable=True)  # 0–100 Platt-scaled
    extracted_entities: Mapped[dict] = mapped_column(JSON, nullable=True)   # ClinicalBERT output
    source_citations: Mapped[list] = mapped_column(JSON, nullable=True)     # ChromaDB hits
    source_count: Mapped[int] = mapped_column(Integer, default=0)
    verdict: Mapped[str] = mapped_column(String(20), nullable=True)
    explanation: Mapped[str] = mapped_column(Text, nullable=True)
    shap_values: Mapped[dict] = mapped_column(JSON, nullable=True)
    retrieved_sources: Mapped[list] = mapped_column(JSON, nullable=True)
    disclaimer: Mapped[str] = mapped_column(Text, nullable=True)
    uncertainty_flag: Mapped[bool] = mapped_column(Boolean, default=False)
    hallucination_detected: Mapped[bool] = mapped_column(Boolean, default=False)
    hallucination_details: Mapped[dict] = mapped_column(JSON, nullable=True)
    celery_task_id: Mapped[str] = mapped_column(String, nullable=True)
    created_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    completed_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), nullable=True)
    updated_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    patient: Mapped["Patient"] = relationship(back_populates="claims")
