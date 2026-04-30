from app.services.ocr import ocr_service


def extract_text(file_path: str, file_format: str | None = None) -> dict:
    return ocr_service.extract_text(file_path, file_format or "")

