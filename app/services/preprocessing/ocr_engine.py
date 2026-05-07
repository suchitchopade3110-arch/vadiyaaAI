"""OCR helpers for PIL images and report files."""

from PIL import Image


def perform_ocr_on_pil_image(image: Image.Image):
    """Run OCR on an in-memory PIL image."""
    import numpy as np

    from app.services.preprocessor import OcrOutput

    arr = np.array(image.convert("RGB"))
    try:
        import cv2
        from app.services.ocr_service import ocr_image, preprocess_image

        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        lines = ocr_image(preprocess_image(bgr))
        text = " ".join(line["text"] for line in lines)
        confidence = round(sum(line["confidence"] for line in lines) / len(lines), 3) if lines else 0.0
    except Exception:
        from app.services.preprocessor import _tesseract_on_array

        text, confidence = _tesseract_on_array(arr)
    return OcrOutput(raw_text=text, confidence=confidence, pages=[text])


def run_ocr(file_path: str):
    """Lazy passthrough to the report OCR engine."""
    from app.services.preprocessor import run_ocr as _run_ocr

    return _run_ocr(file_path)


__all__ = ["perform_ocr_on_pil_image", "run_ocr"]
