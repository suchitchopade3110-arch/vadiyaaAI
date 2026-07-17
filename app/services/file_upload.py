import os
import uuid
import aiofiles
from pathlib import Path
from typing import Tuple
from fastapi import UploadFile, HTTPException, status

from app.core.config import settings


# ── MIME type → extension map ─────────────────────────────────────────────────
ALLOWED_MIME_TYPES = {
    "application/pdf":       "pdf",
    "application/dicom":     "dcm",
    "image/jpeg":            "jpg",
    "image/png":             "png",
    "text/csv":              "csv",
    "application/octet-stream": None,   # DICOM often sent as octet-stream
}

EXTENSION_PIPELINE_MAP = {
    "pdf": "text",
    "csv": "text",
    "dcm": "image",
    "jpg": "image",
    "jpeg": "image",
    "png":  "image",
}


class FileUploadService:
    """Validate + persist uploaded medical files."""

    def __init__(self):
        # Resolve once so API and worker receive an absolute, stable file path.
        self.upload_dir = Path(getattr(settings, "UPLOAD_DIR", "uploads")).expanduser().resolve()
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.max_bytes = getattr(settings, "MAX_FILE_SIZE_MB", 50) * 1024 * 1024
        self.allowed_extensions = getattr(settings, "ALLOWED_EXTENSIONS", ["pdf", "csv", "dcm", "jpg", "jpeg", "png"])

    async def validate_and_save(
        self, file: UploadFile, sub_dir: str = ""
    ) -> Tuple[str, str, str]:
        """
        Validate file + save to disk.
        Returns: (file_path, extension, pipeline_type)
        Raises: HTTPException on invalid file.
        """
        # ── Extension check ────────────────────────────────────────────────
        original_name = file.filename or "upload"
        extension = Path(original_name).suffix.lstrip(".").lower()

        if extension not in self.allowed_extensions:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"File type '.{extension}' not allowed. "
                       f"Supported: {self.allowed_extensions}"
            )

        # ── Size check ─────────────────────────────────────────────────────
        content = await file.read()
        if len(content) > self.max_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="File exceeds limit."
            )

        pipeline_type = EXTENSION_PIPELINE_MAP.get(extension, "unknown")

        # ── MIME and Image decodability check ──────────────────────────────
        import magic
        mime = magic.from_buffer(content[:2048], mime=True)
        
        if pipeline_type == "image":
            if mime not in {"image/jpeg", "image/png", "application/dicom"}:
                raise HTTPException(
                    status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                    detail={"error": {"code": "FILE_FORMAT_UNSUPPORTED", "message": f"Unsupported MIME: {mime}", "retryable": False}}
                )
            if mime in {"image/jpeg", "image/png"}:
                import io
                from PIL import Image
                try:
                    Image.open(io.BytesIO(content)).verify()
                except Exception:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail={"error": {"code": "FILE_FORMAT_UNSUPPORTED", "message": "File is not a valid image", "retryable": False}}
                    )

        # ── DICOM magic bytes check ────────────────────────────────────────
        if extension == "dcm":
            # DICOM files start with 128 bytes preamble + "DICM" at offset 128
            if len(content) < 132 or content[128:132] != b"DICM":
                # Some DICOM files lack the prefix — warn but allow
                pass  # Phase 2: stricter validation

        # ── Save ───────────────────────────────────────────────────────────
        save_dir = self.upload_dir / sub_dir
        save_dir.mkdir(parents=True, exist_ok=True)

        file_id = str(uuid.uuid4())
        safe_name = f"{file_id}.{extension}"
        file_path = save_dir / safe_name

        async with aiofiles.open(file_path, "wb") as f:
            await f.write(content)

        pipeline_type = EXTENSION_PIPELINE_MAP.get(extension, "unknown")

        return str(file_path), extension, pipeline_type

    def quality_score(self, file_path: str, extension: str) -> float:
        """
        Basic quality score (0.0–1.0).
        Phase 2: Replace with real quality metrics.
        """
        try:
            size = os.path.getsize(file_path)
            if extension in ("jpg", "jpeg", "png", "dcm"):
                # Images: penalize very small files (likely corrupt/too small)
                if size < 10_000:   return 0.3
                if size < 100_000:  return 0.6
                return 0.9
            elif extension == "pdf":
                if size < 1000:   return 0.2
                return 0.85
            elif extension == "csv":
                return 0.9
            return 0.5
        except Exception:
            return 0.0

    def cleanup(self, file_path: str):
        """Delete temp file after processing."""
        try:
            os.remove(file_path)
        except Exception:
            pass


file_upload_service = FileUploadService()
