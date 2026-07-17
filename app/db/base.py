# Import all models here so Alembic can discover them
from app.db.base_class import Base  # noqa
from app.models.patient import Patient  # noqa
from app.models.claim import Claim  # noqa
from app.models.report import Report  # noqa
from app.models.image_analysis import ImageAnalysis  # noqa
from app.models.qr_access import QRAuditLog, QRToken  # noqa
from app.models.user import User  # noqa
