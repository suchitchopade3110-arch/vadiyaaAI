"""PaddleOCR-backed OCR service for reports and scanned images."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import fitz
import numpy as np

_ocr_engine = None
MODEL_ROOT = Path(__file__).resolve().parents[2] / "data" / "paddleocr"


def _get_ocr_engine():
    global _ocr_engine
    if _ocr_engine is None:
        from paddleocr import PaddleOCR

        det_dir = MODEL_ROOT / "det"
        rec_dir = MODEL_ROOT / "rec"
        cls_dir = MODEL_ROOT / "cls"
        for directory in (det_dir, rec_dir, cls_dir):
            directory.mkdir(parents=True, exist_ok=True)

        _ocr_engine = PaddleOCR(
            use_angle_cls=True,
            lang="en",
            use_gpu=False,
            show_log=False,
            enable_mkldnn=True,
            det_model_dir=str(det_dir),
            rec_model_dir=str(rec_dir),
            cls_model_dir=str(cls_dir),
        )
    return _ocr_engine


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """CLAHE contrast enhancement + denoising for document OCR."""
    if image is None:
        raise ValueError("Cannot OCR an empty image")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    denoised = cv2.fastNlMeansDenoising(enhanced, h=10)
    return cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)


def ocr_image(image: np.ndarray) -> list[dict[str, Any]]:
    """Run PaddleOCR on a numpy image and return text/confidence/bbox rows."""
    result = _get_ocr_engine().ocr(image, cls=True)
    if not result or not result[0]:
        return []

    extracted = []
    for line in result[0]:
        bbox, (text, confidence) = line
        text = str(text).strip()
        if not text:
            continue
        extracted.append(
            {
                "text": text,
                "confidence": round(float(confidence), 3),
                "bbox": bbox,
            }
        )
    return extracted


def extract_text_from_pdf(pdf_path: str | Path | bytes, dpi: int = 200) -> dict[str, Any]:
    """Extract PDF text using native text plus PaddleOCR page rendering."""
    doc = fitz.open(stream=pdf_path, filetype="pdf") if isinstance(pdf_path, bytes) else fitz.open(str(pdf_path))
    all_pages = []
    full_text = []

    try:
        for page_num, page in enumerate(doc, start=1):
            native_text = page.get_text("text").strip()
            pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72))
            img = cv2.imdecode(np.frombuffer(pix.tobytes("png"), dtype=np.uint8), cv2.IMREAD_COLOR)
            ocr_lines = ocr_image(preprocess_image(img))
            ocr_text = " ".join(line["text"] for line in ocr_lines)
            avg_conf = round(sum(line["confidence"] for line in ocr_lines) / len(ocr_lines), 3) if ocr_lines else 0.0

            use_native = bool(native_text) and len(native_text) > len(ocr_text) * 0.8
            page_text = native_text if use_native else ocr_text
            all_pages.append(
                {
                    "page": page_num,
                    "text": page_text,
                    "ocr_lines": ocr_lines,
                    "avg_confidence": 0.95 if use_native else avg_conf,
                    "method": "native" if use_native else "paddle_ocr",
                }
            )
            full_text.append(page_text)
    finally:
        doc.close()

    return {
        "full_text": "\n\n".join(full_text),
        "pages": all_pages,
        "total_pages": len(all_pages),
        "source_type": "pdf",
        "avg_confidence": _average_page_confidence(all_pages),
    }


def extract_text_from_image(image_path: str | Path | bytes) -> dict[str, Any]:
    """Extract text from JPG/PNG/BMP/TIFF bytes or path."""
    if isinstance(image_path, bytes):
        img = cv2.imdecode(np.frombuffer(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    else:
        img = cv2.imread(str(image_path))

    ocr_lines = ocr_image(preprocess_image(img))
    full_text = " ".join(line["text"] for line in ocr_lines)
    avg_conf = round(sum(line["confidence"] for line in ocr_lines) / len(ocr_lines), 3) if ocr_lines else 0.0
    return {
        "full_text": full_text,
        "ocr_lines": ocr_lines,
        "avg_confidence": avg_conf,
        "total_pages": 1,
        "source_type": "image",
    }


def extract_table_from_image(image: np.ndarray) -> list[list[str]]:
    """Group OCR lines into table-like rows by Y coordinate."""
    ocr_lines = ocr_image(preprocess_image(image))
    if not ocr_lines:
        return []

    ocr_lines.sort(key=lambda line: line["bbox"][0][1])
    rows = []
    current_row = [ocr_lines[0]]

    for line in ocr_lines[1:]:
        y_current = line["bbox"][0][1]
        y_last = current_row[-1]["bbox"][0][1]
        if abs(y_current - y_last) < 15:
            current_row.append(line)
        else:
            current_row.sort(key=lambda item: item["bbox"][0][0])
            rows.append([item["text"] for item in current_row])
            current_row = [line]

    current_row.sort(key=lambda item: item["bbox"][0][0])
    rows.append([item["text"] for item in current_row])
    return rows


def extract_text_from_file(file: str | Path | bytes, file_type: str = "auto") -> dict[str, Any]:
    """Universal PaddleOCR entry point for PDFs and image files."""
    if file_type == "auto" and isinstance(file, (str, Path)):
        suffix = Path(file).suffix.lower()
        file_type = "pdf" if suffix == ".pdf" else "image"

    result = extract_text_from_pdf(file) if file_type == "pdf" else extract_text_from_image(file)
    if "avg_confidence" not in result:
        result["avg_confidence"] = _average_page_confidence(result.get("pages", []))
    return result


def _average_page_confidence(pages: list[dict[str, Any]]) -> float:
    confidences = [page["avg_confidence"] for page in pages if page.get("avg_confidence")]
    return round(sum(confidences) / len(confidences), 3) if confidences else 0.0
