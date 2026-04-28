import os
import io
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class OCRService:

    def extract_text(self, file_path: str, file_format: str) -> dict:
        """
        Main entry point.
        Returns: {raw_text, method, page_count, quality_score, error}
        """
        ext = file_format.lower().strip(".")

        try:
            if ext == "pdf":
                return self._extract_pdf(file_path)
            elif ext == "csv":
                return self._extract_csv(file_path)
            else:
                return self._error(f"Unsupported format: {ext}")
        except Exception as e:
            logger.error(f"OCR failed for {file_path}: {e}")
            return self._error(str(e))

    # ── PDF ───────────────────────────────────────────────────────────────────
    def _extract_pdf(self, file_path: str) -> dict:
        """
        Try PyMuPDF first (fast, text-layer PDFs).
        Fall back to Tesseract OCR for scanned/image PDFs.
        """
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(file_path)
            pages_text = []
            for page in doc:
                pages_text.append(page.get_text("text"))
            doc.close()

            raw_text = "\n".join(pages_text).strip()

            # If PyMuPDF returned almost nothing → scanned PDF → use Tesseract
            if len(raw_text) < 50:
                logger.info(f"PyMuPDF got <50 chars — falling back to Tesseract: {file_path}")
                return self._tesseract_pdf(file_path)

            return {
                "raw_text": raw_text,
                "method": "pymupdf",
                "page_count": len(pages_text),
                "quality_score": min(1.0, len(raw_text) / 1000),
                "error": None,
            }

        except ImportError:
            logger.warning("PyMuPDF not installed — falling back to Tesseract")
            return self._tesseract_pdf(file_path)
        except Exception as e:
            logger.error(f"PyMuPDF failed: {e} — falling back to Tesseract")
            return self._tesseract_pdf(file_path)

    def _tesseract_pdf(self, file_path: str) -> dict:
        """Convert PDF pages to images → Tesseract OCR."""
        try:
            import fitz
            import pytesseract
            from PIL import Image

            doc = fitz.open(file_path)
            all_text = []

            for page_num, page in enumerate(doc):
                mat = fitz.Matrix(3, 3)  # 216 DPI — better for OCR
                pix = page.get_pixmap(matrix=mat)
                img_bytes = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_bytes))
                text = pytesseract.image_to_string(img, lang="eng", config="--psm 6 --oem 3")
                all_text.append(text)

            doc.close()
            raw_text = "\n".join(all_text).strip()

            return {
                "raw_text": raw_text,
                "method": "tesseract",
                "page_count": len(all_text),
                "quality_score": min(1.0, len(raw_text) / 1000),
                "error": None,
            }

        except Exception as e:
            logger.error(f"Tesseract OCR failed: {e}")
            return self._error(f"Tesseract failed: {e}")

    # ── CSV ───────────────────────────────────────────────────────────────────
    def _extract_csv(self, file_path: str) -> dict:
        """
        Parse CSV with pandas.
        Converts to readable key:value text for ClinicalBERT NER.
        """
        try:
            import pandas as pd

            df = pd.read_csv(file_path)

            # Convert DataFrame to human-readable text for NER
            lines = []
            for _, row in df.iterrows():
                for col, val in row.items():
                    if pd.notna(val):
                        lines.append(f"{col}: {val}")

            raw_text = "\n".join(lines)

            return {
                "raw_text": raw_text,
                "method": "pandas-csv",
                "page_count": 1,
                "row_count": len(df),
                "columns": list(df.columns),
                "quality_score": 0.95,
                "error": None,
            }

        except Exception as e:
            logger.error(f"CSV parse failed: {e}")
            return self._error(f"CSV parse failed: {e}")

    def _error(self, msg: str) -> dict:
        return {
            "raw_text": "",
            "method": "failed",
            "page_count": 0,
            "quality_score": 0.0,
            "error": msg,
        }


# ── Singleton ─────────────────────────────────────────────────────────────────
ocr_service = OCRService()
