import os

from pydantic import Field
from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    # App
    APP_ENV: str = "development"
    SECRET_KEY: str = "change-me-in-production"
    QR_SECRET_KEY: str = ""
    ALLOWED_ORIGINS: List[str] = [
        "*",
        "http://localhost:3000",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "http://localhost:8000",
        "null",  # file:// origin
    ]

    # PostgreSQL
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "vaidyaai"
    POSTGRES_USER: str = "vaidya"
    POSTGRES_PASSWORD: str = "vaidya123"
    DATABASE_URL_ENV: str = Field(default="", alias="DATABASE_URL")

    @property
    def DATABASE_URL(self) -> str:
        if self.DATABASE_URL_ENV:
            if self.DATABASE_URL_ENV.startswith("postgresql://"):
                return self.DATABASE_URL_ENV.replace("postgresql://", "postgresql+asyncpg://", 1)
            return self.DATABASE_URL_ENV
        return (
            f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    # Redis / Celery
    REDIS_URL: str = "redis://localhost:6379/0"
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/1"

    # LLM
    OPENAI_API_KEY: str = ""
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.1-8b-instant"
    CHROMA_PATH: str = os.path.expanduser("~/vaidyaAI_week1_clean/chromadb")
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    LLM_PROVIDER: str = "openai"  # "openai" | "ollama"

    # File upload
    MAX_UPLOAD_SIZE_MB: int = 50
    UPLOAD_DIR: str = "./uploads"

    # Safety
    CONFIDENCE_THRESHOLD: float = 0.65  # below → "Insufficient evidence"
    MIN_CONFIDENCE_THRESHOLD: float = 0.30   # Below → "Insufficient evidence"
    HIGH_CONFIDENCE_THRESHOLD: float = 0.80  # Above → Green
    MIN_SOURCES_REQUIRED: int = 2            # Below → Uncertainty flag
    
    MEDICAL_DISCLAIMER: str = (
        "⚠️ AI-assisted analysis only. NOT A MEDICAL DIAGNOSIS. "
        "Consult a qualified healthcare professional for clinical decisions."
    )

    # Celery Task Timeouts (seconds)
    TASK_SOFT_TIME_LIMIT: int = 120
    TASK_HARD_TIME_LIMIT: int = 300
    IMAGE_TASK_TIMEOUT: int = 120   # MedSAM can be slow
    TEXT_TASK_TIMEOUT: int = 30

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"


settings = Settings()
