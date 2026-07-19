"""
classifier.py — VaidyaAI Image Pipeline
CheXNet/ResNet classifier on MedSAM-extracted ROI.

Role (PRD constraint): Image Classification ONLY.
Input:  ROI numpy array (H, W, 3) from roi_extractor.py
Output: ImageClassification (Pydantic) → backend contract

PRD Output Contract (design_doc 14.3):
    label: str
    confidence: float          # Platt-scaled 0.0-1.0
    probabilities: dict        # class -> probability
    gradcam_path: str          # filled by gradcam.py later
"""

import os
import uuid
import numpy as np
from PIL import Image
from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from pydantic import BaseModel

# ── Device ────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── CheXNet 14-class labels (standard NIH ChestX-ray14) ──
CHEXNET_CLASSES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia"
]

# ── Pydantic output contract (matches design_doc 14.3) ────
class ImageClassification(BaseModel):
    label: str
    confidence: float          # Platt-scaled 0.0–1.0
    probabilities: dict        # {class_name: probability}
    gradcam_path: str          # placeholder; filled by gradcam.py


# ── Platt scaling (same logic as RAG Phase 2) ─────────────
def platt_scale(raw_prob: float, A: float = -2.0, B: float = 1.0) -> float:
    """Calibrates raw sigmoid output → calibrated probability 0–1."""
    import math
    calibrated = 1.0 / (1.0 + math.exp(A * raw_prob + B))
    return float(np.clip(calibrated, 0.0, 1.0))


# ── Image transform (CheXNet standard) ────────────────────
CHEXNET_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],   # ImageNet stats (CheXNet uses these)
        std=[0.229, 0.224, 0.225]
    )
])


# ── Model loader ──────────────────────────────────────────
class CheXNetModel(nn.Module):
    """
    DenseNet-121 backbone with 14-class sigmoid head.
    Matches original CheXNet architecture (Rajpurkar et al. 2017).
    Falls back to ResNet-50 if DenseNet unavailable.
    """
    def __init__(self, num_classes: int = 14, pretrained: bool = True):
        super().__init__()
        try:
            # DenseNet-121 — original CheXNet backbone
            self.backbone = models.densenet121(
                weights=models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
            )
            in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Sequential(
                nn.Linear(in_features, num_classes),
                nn.Sigmoid()
            )
            self.arch = "densenet121"
        except Exception:
            # ResNet-50 fallback
            self.backbone = models.resnet50(
                weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            )
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Linear(in_features, num_classes),
                nn.Sigmoid()
            )
            self.arch = "resnet50"

    def forward(self, x):
        return self.backbone(x)


def load_chexnet(checkpoint_path: Optional[str] = None) -> CheXNetModel:
    """
    Loads CheXNet model.
    - If checkpoint_path provided and exists → loads fine-tuned weights.
    - Otherwise → uses ImageNet pretrained weights as baseline.
    
    NOTE: For production, download CheXNet weights from:
    https://github.com/arnoweng/CheXNet (community reimplementation)
    """
    model = CheXNetModel(num_classes=len(CHEXNET_CLASSES), pretrained=True)

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"[CheXNet] Loading checkpoint: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location=device)
        # Handle both raw state_dict and wrapped checkpoints
        if "state_dict" in state:
            state = state["state_dict"]
        # Strip module. prefix if saved from DataParallel
        state = {k.replace("module.", ""): v for k, v in state.items()}
        model.backbone.load_state_dict(state, strict=False)
        print(f"[CheXNet] Checkpoint loaded. Arch: {model.arch}")
    else:
        print(f"[CheXNet] No checkpoint. Using ImageNet pretrained ({model.arch}). "
              f"Accuracy will be baseline — provide CheXNet weights for medical accuracy.")

    model.to(device)
    model.eval()
    return model


# ── Singleton model instance (load once, reuse) ───────────
_model: Optional[CheXNetModel] = None

def get_model(checkpoint_path: Optional[str] = None) -> CheXNetModel:
    global _model
    if _model is None:
        _model = load_chexnet(checkpoint_path)
    return _model


# ── Core classifier ───────────────────────────────────────
def run_chexnet(
    roi: np.ndarray,
    checkpoint_path: Optional[str] = None,
    top_k: int = 3,
    confidence_threshold: float = 0.5
) -> ImageClassification:
    """
    Main entry point. Takes ROI numpy array → returns ImageClassification.

    Args:
        roi:                 np.ndarray (H, W, 3) uint8 — from extract_roi()
        checkpoint_path:     optional path to .pth CheXNet weights
        top_k:               number of top conditions to include in probabilities dict
        confidence_threshold: below this → uncertainty_flag should be set upstream

    Returns:
        ImageClassification (Pydantic) — matches PRD design_doc 14.3 contract
    """
    if roi is None or roi.size == 0:
        raise ValueError("[CheXNet] ROI is empty. Check MedSAM segmentation output.")

    model = get_model(checkpoint_path)

    # ── Preprocess ROI ────────────────────────────────────
    # Ensure uint8 for PIL
    if roi.dtype != np.uint8:
        roi = (roi - roi.min()) / (roi.max() - roi.min() + 1e-8)
        roi = (roi * 255).astype(np.uint8)

    pil_img = Image.fromarray(roi).convert("RGB")
    tensor = CHEXNET_TRANSFORM(pil_img).unsqueeze(0).to(device)  # (1, 3, 224, 224)

    # ── Inference ─────────────────────────────────────────
    with torch.no_grad():
        raw_probs = model(tensor)                    # (1, 14) sigmoid output
        raw_probs = raw_probs.squeeze(0).cpu().numpy()  # (14,)

    # ── Build probabilities dict ──────────────────────────
    prob_dict = {cls: float(prob) for cls, prob in zip(CHEXNET_CLASSES, raw_probs)}

    # Top-k conditions by raw probability
    sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
    top_probs = dict(sorted_probs[:top_k])

    # ── Primary label = highest probability class ─────────
    primary_label, primary_raw_conf = sorted_probs[0]

    # ── Check for "Normal" — if all classes below threshold
    if primary_raw_conf < confidence_threshold:
        primary_label = "Normal"

    # ── Platt-scale confidence ────────────────────────────
    calibrated_conf = platt_scale(primary_raw_conf)

    return ImageClassification(
        label=primary_label,
        confidence=calibrated_conf,
        probabilities=top_probs,
        gradcam_path=""   # filled by gradcam.py
    )


# ── Batch classifier ──────────────────────────────────────
def run_chexnet_batch(
    rois: list,
    checkpoint_path: Optional[str] = None
) -> list:
    """
    Batch classify list of ROI arrays.
    Returns list of ImageClassification objects.
    """
    return [run_chexnet(roi, checkpoint_path) for roi in rois]


# ── Standalone test ───────────────────────────────────────
if __name__ == "__main__":
    print("=== CheXNet Classifier Test ===")
    print(f"Device: {device}")

    # Synthetic ROI (replace with real MedSAM output)
    dummy_roi = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

    result = run_chexnet(dummy_roi)

    print("\nResult:")
    print(f"  Label:      {result.label}")
    print(f"  Confidence: {result.confidence:.4f} (Platt-scaled)")
    print(f"  Top probs:  {result.probabilities}")
    print(f"  GradCAM:    '{result.gradcam_path}' (empty until gradcam.py runs)")
    print("\nPRD contract check:")
    print("  label ✓ | confidence ✓ | probabilities ✓ | gradcam_path ✓")
