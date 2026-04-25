import uuid
import enum
from sqlalchemy import String, Float, JSON, DateTime, ForeignKey, func, Enum, Text, Boolean
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID
from app.db.base_class import Base
from app.models.report import AnalysisStatus


class ImageType(str, enum.Enum):
    XRAY = "xray"
    CT = "ct"
    MRI = "mri"
    SKIN = "skin"
    PATHOLOGY = "pathology"


class ImageAnalysis(Base):
    __tablename__ = "image_analyses"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    patient_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("patients.id"), nullable=True
    )
    image_type: Mapped[ImageType] = mapped_column(Enum(ImageType), nullable=False)
    file_path: Mapped[str] = mapped_column(String(500), nullable=True)
    file_format: Mapped[str] = mapped_column(String(10), nullable=True)
    status: Mapped[AnalysisStatus] = mapped_column(Enum(AnalysisStatus), default=AnalysisStatus.PENDING)
    dicom_metadata: Mapped[dict] = mapped_column(JSON, nullable=True)
    segmentation_mask_path: Mapped[str] = mapped_column(String(500), nullable=True)  # MedSAM output
    segmentation_overlay: Mapped[str] = mapped_column(String(500), nullable=True)
    segmentation_confidence: Mapped[float] = mapped_column(Float, nullable=True)
    gradcam_path: Mapped[str] = mapped_column(String(500), nullable=True)             # GradCAM overlay
    gradcam_regions: Mapped[list] = mapped_column(JSON, nullable=True)
    classification: Mapped[dict] = mapped_column(JSON, nullable=True)            # CheXNet output
    classification_probs: Mapped[dict] = mapped_column(JSON, nullable=True)
    roi_metadata: Mapped[dict] = mapped_column(JSON, nullable=True)              # bounding boxes
    explanation: Mapped[str] = mapped_column(Text, nullable=True)
    confidence: Mapped[float] = mapped_column(Float, nullable=True)
    confidence_score: Mapped[float] = mapped_column(Float, nullable=True)
    source_citations: Mapped[list] = mapped_column(JSON, nullable=True)
    retrieved_sources: Mapped[list] = mapped_column(JSON, nullable=True)
    uncertainty_flag: Mapped[bool] = mapped_column(Boolean, default=False)
    anomaly_detected: Mapped[bool] = mapped_column(Boolean, default=False)
    medical_disclaimer: Mapped[str] = mapped_column(Text, nullable=True)
    celery_task_id: Mapped[str] = mapped_column(String(255), nullable=True)
    created_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    completed_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), nullable=True)

    patient: Mapped["Patient"] = relationship(back_populates="image_analyses")
