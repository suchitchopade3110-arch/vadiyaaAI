import uuid
import enum
from sqlalchemy import String, JSON, DateTime, ForeignKey, func, Enum, Float, Text, Boolean
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID
from app.db.base_class import Base


class ReportType(str, enum.Enum):
    LAB = "lab"
    CLINICAL = "clinical"
    DISCHARGE = "discharge"

class AnalysisStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETE = "complete"
    FAILED = "failed"


class Report(Base):
    __tablename__ = "reports"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    patient_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("patients.id"), nullable=True
    )
    report_type: Mapped[ReportType] = mapped_column(Enum(ReportType), nullable=False)
    file_path: Mapped[str] = mapped_column(String(500), nullable=True)
    file_format: Mapped[str] = mapped_column(String(10), nullable=True)
    status: Mapped[AnalysisStatus] = mapped_column(Enum(AnalysisStatus), default=AnalysisStatus.PENDING)
    raw_text: Mapped[str] = mapped_column(Text, nullable=True)       # OCR output
    parsed_data: Mapped[dict] = mapped_column(JSON, nullable=True)     # ClinicalBERT structured
    risk_score: Mapped[float] = mapped_column(Float, nullable=True)
    risk_factors: Mapped[list] = mapped_column(JSON, nullable=True)
    shap_values: Mapped[dict] = mapped_column(JSON, nullable=True)
    anomalies: Mapped[list] = mapped_column(JSON, nullable=True)
    explanation: Mapped[str] = mapped_column(Text, nullable=True)
    retrieved_sources: Mapped[list] = mapped_column(JSON, nullable=True)
    analysis_result: Mapped[dict] = mapped_column(JSON, nullable=True) # XGBoost + LLM output
    source_citations: Mapped[list] = mapped_column(JSON, nullable=True)
    confidence_score: Mapped[float] = mapped_column(Float, nullable=True)
    confidence: Mapped[float] = mapped_column(Float, nullable=True)
    uncertainty_flag: Mapped[bool] = mapped_column(Boolean, default=False)
    medical_disclaimer: Mapped[str] = mapped_column(Text, nullable=True)
    celery_task_id: Mapped[str] = mapped_column(String(255), nullable=True)
    created_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    completed_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), nullable=True)

    patient: Mapped["Patient"] = relationship(back_populates="reports")
