"""Image normalization helpers."""

from typing import Tuple


def enhance_image(image_input, target_size: Tuple[int, int] = (1024, 1024)):
    """Normalize, resize, apply CLAHE, and denoise an image."""
    from app.services.preprocessor import normalize_image

    return normalize_image(image_input, target_size=target_size, apply_clahe=True, apply_denoise=True)


def augment(image_input, flip: bool = False, rotate_degrees: float = 0.0):
    """Apply lightweight deterministic augmentation for preprocessing tests/batches."""
    import cv2
    import numpy as np
    from PIL import Image

    arr = np.array(Image.open(image_input).convert("RGB")) if isinstance(image_input, str) else np.asarray(image_input).copy()
    if flip:
        arr = cv2.flip(arr, 1)
    if rotate_degrees:
        h, w = arr.shape[:2]
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), rotate_degrees, 1.0)
        arr = cv2.warpAffine(arr, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return arr


def is_low_quality(image_input, threshold: float = 0.35) -> bool:
    """Return True when the existing quality score falls below threshold."""
    from app.services.preprocessor import normalize_image

    result = normalize_image(image_input)
    return bool(result.rejected or result.quality_score < threshold)


def normalize(file_path: str):
    """Backward-compatible normalizer entry point."""
    return enhance_image(file_path)


__all__ = ["augment", "enhance_image", "is_low_quality", "normalize"]
