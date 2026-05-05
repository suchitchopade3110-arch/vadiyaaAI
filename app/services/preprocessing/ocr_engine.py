"""OCR helpers for PIL images and report files."""

from PIL import Image


def perform_ocr_on_pil_image(image: Image.Image):
    """Run OCR on an in-memory PIL image."""
    import numpy as np

    from app.services.preprocessor import OcrOutput, _tesseract_on_array

    arr = np.array(image.convert("RGB"))
    text, confidence = _tesseract_on_array(arr)
    return OcrOutput(raw_text=text, confidence=confidence, pages=[text])


def run_ocr(file_path: str):
    """Lazy passthrough to the report OCR engine."""
    from app.services.preprocessor import run_ocr as _run_ocr

    return _run_ocr(file_path)


__all__ = ["perform_ocr_on_pil_image", "run_ocr"]
