import os

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # App
    APP_ENV: str = "development"
    SECRET_KEY: str
    QR_SECRET_KEY: str = ""

    # PostgreSQL
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "vaidyaai"
    POSTGRES_USER: str = "vaidya"
    POSTGRES_PASSWORD: str
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
    CHROMADB_HOST: str = "localhost"
    CHROMADB_PORT: int = 8001
    CHROMADB_URL: str = "http://localhost:8001"
    CHROMA_PATH: str = "data/chromadb"
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

    # Auth
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 20
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # Request-level safety net (long-running work belongs in Celery, not here)
    REQUEST_TIMEOUT_SECONDS: int = 30

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"


settings = Settings()
