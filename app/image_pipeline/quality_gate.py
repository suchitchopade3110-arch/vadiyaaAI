import cv2
import numpy as np

def is_valid_medical_image(image_path: str) -> tuple[bool, str]:
    """
    Quality gate to reject screenshots or poor quality images.
    Heuristics can include checking aspect ratio, intensity distribution, or UI elements.
    """
    # Simple heuristic: check for large uniform areas (typical of screenshots with UI)
    # or check file metadata if possible.
    
    image = cv2.imread(image_path)
    if image is None:
        return False, "Unable to load image"
    
    # Aspect ratio check (most medical scans have standard ratios)
    h, w = image.shape[:2]
    aspect_ratio = w / h
    if aspect_ratio > 3.0 or aspect_ratio < 0.3:
        return False, f"Suspicious aspect ratio: {aspect_ratio:.2f}"

    # Intensity histogram check (medical images usually have specific distributions)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    
    # If the image is mostly one color (e.g., white or black background of a screenshot)
    max_freq = np.max(hist)
    if max_freq > (w * h * 0.9):
        return False, "Image is mostly uniform (potential screenshot or empty)"

    return True, "Valid"
