import uuid
import enum
from sqlalchemy import String, Float, JSON, DateTime, ForeignKey, func, Enum
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID
from app.db.base_class import Base


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
    file_path: Mapped[str] = mapped_column(String, nullable=True)
    segmentation_mask_path: Mapped[str] = mapped_column(String, nullable=True)  # MedSAM output
    gradcam_path: Mapped[str] = mapped_column(String, nullable=True)             # GradCAM overlay
    classification: Mapped[dict] = mapped_column(JSON, nullable=True)            # CheXNet output
    roi_metadata: Mapped[dict] = mapped_column(JSON, nullable=True)              # bounding boxes
    confidence: Mapped[float] = mapped_column(Float, nullable=True)
    source_citations: Mapped[list] = mapped_column(JSON, nullable=True)
    uncertainty_flag: Mapped[bool] = mapped_column(default=False)
    celery_task_id: Mapped[str] = mapped_column(String, nullable=True)
    created_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    patient: Mapped["Patient"] = relationship(back_populates="image_analyses")
