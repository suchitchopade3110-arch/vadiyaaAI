import os
import cv2
import numpy as np
from app.image_pipeline.pipeline import run_image_pipeline

def smoke_test():
    # 1. Create a dummy medical-looking image
    dummy_img = np.zeros((512, 512, 3), dtype=np.uint8)
    # Draw a "lung" or "nodule" looking thing
    cv2.circle(dummy_img, (256, 256), 50, (200, 200, 200), -1)
    
    img_path = "scratch/test_medical_image.png"
    os.makedirs("scratch", exist_ok=True)
    cv2.imwrite(img_path, dummy_img)
    
    print(f"Running pipeline on {img_path}...")
    try:
        output = run_image_pipeline(img_path, "Chest X-Ray")
        print("\nPipeline execution SUCCESS!")
        print("-" * 30)
        print(f"Label:      {output.classification.label}")
        print(f"Confidence: {output.classification.confidence:.2f}")
        print(f"Mask path:  {output.segmentation.mask_path}")
        print(f"ROI path:   {output.segmentation.roi_path}")
        print(f"Explanation: {output.explanation[:100]}...")
        print("-" * 30)
    except Exception as e:
        print(f"\nPipeline execution FAILED: {e}")

if __name__ == "__main__":
    smoke_test()
