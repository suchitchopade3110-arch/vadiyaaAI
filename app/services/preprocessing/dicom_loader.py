"""DICOM loading helpers."""

from app.services.preprocessor import DicomOutput, parse_dicom


def load_dicom(path: str):
    """Load one DICOM file, returning a normalized pixel array."""
    from app.image_pipeline.segmentor import load_dicom_slice

    return load_dicom_slice(path)


def load_dicom_series(folder: str):
    """Load and sort a DICOM series folder."""
    from app.image_pipeline.segmentor import load_dicom_series as _load_dicom_series

    return _load_dicom_series(folder)


def parse_dicom_input(path: str) -> DicomOutput:
    """Parse a DICOM file or directory with metadata."""
    return parse_dicom(path)


__all__ = ["DicomOutput", "load_dicom", "load_dicom_series", "parse_dicom_input"]
