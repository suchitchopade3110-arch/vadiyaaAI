"""
VaidyaAI File Upload Validator
Validates before any disk write or task dispatch.
Checks: extension, MIME, size, magic bytes (DICOM), basic content sanity.
"""

import os
import magic  # python-magic
from fastapi import UploadFile, HTTPException, status
from app.core.config import settings

# ── Allowed formats per pipeline ───────────────────────────────────────────
ALLOWED = {
    "image": {
        "extensions": {".dcm", ".jpg", ".jpeg", ".png"},
        "mimes": {
            "application/dicom",
            "image/jpeg",
            "image/png",
        },
    },
    "report": {
        "extensions": {".pdf", ".csv"},
        "mimes": {
            "application/pdf",
            "text/csv",
            "text/plain",           # some CSVs sniff as text/plain
            "application/octet-stream",  # fallback for some PDF uploads
        },
    },
}

DICOM_MAGIC = b"DICM"          # at offset 128 in valid DICOM files
MAX_BYTES = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024


async def validate_upload(file: UploadFile, pipeline: str) -> bytes:
    """
    Read, validate, return raw bytes.
    Raises HTTPException on any validation failure.
    pipeline: "image" | "report"
    """
    # 1. Extension check
    ext = os.path.splitext(file.filename or "")[1].lower()
    allowed = ALLOWED[pipeline]
    if ext not in allowed["extensions"]:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Extension '{ext}' not allowed for {pipeline} pipeline. "
                   f"Allowed: {sorted(allowed['extensions'])}",
        )

    # 2. Read into memory (size check)
    raw = await file.read()
    if len(raw) > MAX_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds {settings.MAX_UPLOAD_SIZE_MB} MB limit "
                   f"({len(raw) / 1024 / 1024:.1f} MB received).",
        )
    if len(raw) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty file upload rejected.",
        )

    # 3. Magic-byte MIME sniff (not just Content-Type header)
    sniffed_mime = magic.from_buffer(raw[:2048], mime=True)
    if sniffed_mime not in allowed["mimes"]:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"File content detected as '{sniffed_mime}', not valid for {pipeline} pipeline.",
        )

    # 3.5 Image decodability check
    if sniffed_mime in {"image/jpeg", "image/png"}:
        import io
        from PIL import Image
        try:
            Image.open(io.BytesIO(raw)).verify()
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": {"code": "FILE_FORMAT_UNSUPPORTED", "message": "File is not a valid image", "retryable": False}}
            )

    # 4. DICOM-specific: verify magic bytes at offset 128
    if ext == ".dcm":
        _validate_dicom_magic(raw)

    # 5. PDF-specific: verify PDF header
    if ext == ".pdf":
        _validate_pdf_header(raw)

    # Reset so downstream can re-read if needed
    await file.seek(0)
    return raw


def _validate_dicom_magic(raw: bytes):
    """DICOM files must have 'DICM' at byte offset 128."""
    if len(raw) < 132:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File too small to be a valid DICOM file (< 132 bytes).",
        )
    if raw[128:132] != DICOM_MAGIC:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File does not contain valid DICOM magic bytes at offset 128. "
                   "Ensure file is a real DICOM, not a renamed file.",
        )


def _validate_pdf_header(raw: bytes):
    """PDF files must start with %PDF-"""
    if not raw.startswith(b"%PDF-"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File does not appear to be a valid PDF (missing %PDF- header).",
        )
