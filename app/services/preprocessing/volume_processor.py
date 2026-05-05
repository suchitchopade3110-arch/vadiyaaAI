"""Volume and batch processing wrappers for DICOM/image inputs."""


def process_volume(folder: str, model_predictor, modality: str = "CT", max_slices: int | None = None):
    """Process a DICOM volume/series directory."""
    from app.image_pipeline.segmentor import process_dicom_series

    return process_dicom_series(folder, model_predictor, modality=modality, max_slices=max_slices)


def process_batch(image_paths: list[str], model_predictor, modality: str = "unknown", max_workers: int = 4):
    """Process a batch of standard images."""
    from app.image_pipeline.segmentor import batch_process_images

    return batch_process_images(image_paths, model_predictor, modality=modality, max_workers=max_workers)


__all__ = ["process_batch", "process_volume"]
