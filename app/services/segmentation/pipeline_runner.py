"""Service-level runner for the image segmentation pipeline."""


def pipeline(image, model_predictor=None, source_path: str = "", slice_index=None, modality: str = "unknown"):
    """Run the MedSAM/SAM segmentation pipeline."""
    from app.image_pipeline.segmentor import get_predictor, run_pipeline

    predictor = model_predictor or get_predictor()
    return run_pipeline(image, predictor, source_path=source_path, slice_index=slice_index, modality=modality)


__all__ = ["pipeline"]
