"""Unit tests for app/core/file_validator.py — extension, size, magic-byte,
and content-sanity rejection paths. No live services required.
"""
from __future__ import annotations

import io

import pytest
from fastapi import HTTPException
from starlette.datastructures import UploadFile

from app.core import file_validator
from app.core.file_validator import (
    _validate_dicom_magic,
    _validate_pdf_header,
    validate_upload,
)

PDF_BYTES = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n1 0 obj\n<< /Type /Catalog >>\nendobj\ntrailer\n<< /Root 1 0 R >>\n"


def _png_bytes() -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (8, 8), color=(255, 0, 0)).save(buf, format="PNG")
    return buf.getvalue()


def _upload(data: bytes, filename: str) -> UploadFile:
    return UploadFile(file=io.BytesIO(data), filename=filename)


# ── Extension checks ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_rejects_disallowed_extension_for_pipeline():
    upload = _upload(PDF_BYTES, "report.exe")
    with pytest.raises(HTTPException) as exc:
        await validate_upload(upload, "report")
    assert exc.value.status_code == 415


@pytest.mark.asyncio
async def test_rejects_image_extension_on_report_pipeline():
    upload = _upload(_png_bytes(), "scan.png")
    with pytest.raises(HTTPException) as exc:
        await validate_upload(upload, "report")
    assert exc.value.status_code == 415


# ── Size checks ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_rejects_oversized_file(monkeypatch):
    monkeypatch.setattr(file_validator, "MAX_BYTES", 10)
    upload = _upload(PDF_BYTES, "report.pdf")
    with pytest.raises(HTTPException) as exc:
        await validate_upload(upload, "report")
    assert exc.value.status_code == 413


@pytest.mark.asyncio
async def test_rejects_empty_file():
    upload = _upload(b"", "report.pdf")
    with pytest.raises(HTTPException) as exc:
        await validate_upload(upload, "report")
    assert exc.value.status_code == 400
    assert "Empty file" in exc.value.detail


# ── Magic-byte / content sniffing ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_accepts_valid_pdf():
    upload = _upload(PDF_BYTES, "report.pdf")
    raw = await validate_upload(upload, "report")
    assert raw == PDF_BYTES


@pytest.mark.asyncio
async def test_accepts_valid_png_image():
    png = _png_bytes()
    upload = _upload(png, "scan.png")
    raw = await validate_upload(upload, "image")
    assert raw == png


@pytest.mark.asyncio
async def test_rejects_content_mime_mismatch():
    """A .pdf extension with actual PNG bytes must fail the magic-byte sniff."""
    upload = _upload(_png_bytes(), "fake.pdf")
    with pytest.raises(HTTPException) as exc:
        await validate_upload(upload, "report")
    assert exc.value.status_code == 415
    assert "detected as" in exc.value.detail


@pytest.mark.asyncio
async def test_rejects_corrupt_image_content():
    """Valid image extension + magic bytes, but not a decodable image."""
    # A JPEG-ish magic prefix so it sniffs as image/jpeg, but truncated/garbage
    # body so PIL's verify() fails.
    fake_jpeg = b"\xff\xd8\xff\xe0" + b"\x00" * 50
    upload = _upload(fake_jpeg, "broken.jpg")
    with pytest.raises(HTTPException) as exc:
        await validate_upload(upload, "image")
    assert exc.value.status_code == 400


# ── DICOM magic-byte helper (unit-level, no libmagic dependency) ──────────


def test_validate_dicom_magic_accepts_valid_preamble():
    raw = (b"\x00" * 128) + b"DICM" + b"rest of file"
    _validate_dicom_magic(raw)  # must not raise


def test_validate_dicom_magic_rejects_too_small():
    with pytest.raises(HTTPException) as exc:
        _validate_dicom_magic(b"\x00" * 50)
    assert exc.value.status_code == 400
    assert "too small" in exc.value.detail


def test_validate_dicom_magic_rejects_missing_magic():
    raw = (b"\x00" * 128) + b"NOPE"
    with pytest.raises(HTTPException) as exc:
        _validate_dicom_magic(raw)
    assert exc.value.status_code == 400
    assert "DICOM magic bytes" in exc.value.detail


# ── PDF header helper (unit-level) ────────────────────────────────────────


def test_validate_pdf_header_accepts_valid_header():
    _validate_pdf_header(PDF_BYTES)  # must not raise


def test_validate_pdf_header_rejects_missing_header():
    with pytest.raises(HTTPException) as exc:
        _validate_pdf_header(b"not a pdf")
    assert exc.value.status_code == 400
    assert "%PDF-" in exc.value.detail
