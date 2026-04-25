import uuid
import enum
from sqlalchemy import String, JSON, DateTime, ForeignKey, func, Enum
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID
from app.db.base_class import Base


class ReportType(str, enum.Enum):
    LAB = "lab"
    CLINICAL = "clinical"
    DISCHARGE = "discharge"


class Report(Base):
    __tablename__ = "reports"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    patient_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("patients.id"), nullable=True
    )
    report_type: Mapped[ReportType] = mapped_column(Enum(ReportType), nullable=False)
    file_path: Mapped[str] = mapped_column(String, nullable=True)
    raw_text: Mapped[str] = mapped_column(String, nullable=True)       # OCR output
    parsed_data: Mapped[dict] = mapped_column(JSON, nullable=True)     # ClinicalBERT structured
    analysis_result: Mapped[dict] = mapped_column(JSON, nullable=True) # XGBoost + LLM output
    source_citations: Mapped[list] = mapped_column(JSON, nullable=True)
    confidence: Mapped[float] = mapped_column(nullable=True)
    uncertainty_flag: Mapped[bool] = mapped_column(default=False)
    celery_task_id: Mapped[str] = mapped_column(String, nullable=True)
    created_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    patient: Mapped["Patient"] = relationship(back_populates="reports")
