"""
VaidyaAI File Storage
Saves validated upload bytes to disk with UUID-based filenames.
No original filename used in path — prevents path traversal.
"""

import os
import uuid
from pathlib import Path
from app.core.config import settings


def save_upload(raw: bytes, original_filename: str, subfolder: str) -> str:
    """
    Save raw bytes to UPLOAD_DIR/{subfolder}/{uuid}{ext}.
    Returns absolute file path.
    """
    ext = Path(original_filename).suffix.lower()
    safe_name = f"{uuid.uuid4().hex}{ext}"
    dest_dir = Path(settings.UPLOAD_DIR).expanduser().resolve() / subfolder
    dest_dir.mkdir(parents=True, exist_ok=True)

    dest_path = dest_dir / safe_name
    dest_path.write_bytes(raw)
    return str(dest_path)


def delete_upload(file_path: str) -> bool:
    """Remove file from disk. Safe no-op if not found."""
    try:
        Path(file_path).unlink(missing_ok=True)
        return True
    except Exception:
        return False
