"""Writers for segmentation mask artifacts."""

import os
from uuid import uuid4


def save_mask(mask, output_dir: str = "uploads/masks", filename: str | None = None) -> str:
    """Save a binary/float mask as an 8-bit PNG and return the path."""
    import cv2
    import numpy as np

    os.makedirs(output_dir, exist_ok=True)
    name = filename or f"{uuid4()}.png"
    path = os.path.join(output_dir, name)
    arr = np.asarray(mask)
    if arr.dtype != np.uint8:
        arr = (arr > 0).astype(np.uint8) * 255
    cv2.imwrite(path, arr)
    return path


__all__ = ["save_mask"]
