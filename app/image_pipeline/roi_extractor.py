import numpy as np
import cv2

def extract_roi(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Crops the original image to the mask bounds.
    """
    coords = np.argwhere(mask)
    if coords.size == 0:
        return image
    
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    roi = image[y_min:y_max+1, x_min:x_max+1]
    return roi

def compute_confidence(mask: np.ndarray, bbox: list) -> float:
    """
    Compute confidence based on mask/bbox area ratio.
    """
    mask_area = np.sum(mask > 0)
    x1, y1, x2, y2 = bbox
    bbox_area = (x2 - x1) * (y2 - y1)
    
    if bbox_area == 0:
        return 0.0
    
    confidence = mask_area / bbox_area
    return float(np.clip(confidence, 0.0, 1.0))
