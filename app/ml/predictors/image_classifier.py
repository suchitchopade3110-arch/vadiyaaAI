"""CheXNet image classifier adapter."""

import os
from typing import Optional

_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
DEFAULT_CHECKPOINT = os.path.join(_MODELS_DIR, "chexnet.pth")

CHEXNET_CLASSES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia",
]


def _classifier_module():
    from app.image_pipeline import classifier

    return classifier


class CheXNet:
    """Lazy CheXNet class proxy."""

    def __new__(cls, *args, **kwargs):
        return _classifier_module().CheXNetModel(*args, **kwargs)


def classify_image(
    roi,
    checkpoint_path: Optional[str] = None,
    top_k: int = 3,
    confidence_threshold: float = 0.5,
) :
    """Classify a segmented ROI with CheXNet/DenseNet."""
    run_chexnet = _classifier_module().run_chexnet
    checkpoint = checkpoint_path or DEFAULT_CHECKPOINT
    if not os.path.exists(checkpoint):
        checkpoint = None
    return run_chexnet(
        roi,
        checkpoint_path=checkpoint,
        top_k=top_k,
        confidence_threshold=confidence_threshold,
    )


def load_chexnet(checkpoint_path: Optional[str] = None):
    return _classifier_module().load_chexnet(checkpoint_path)


def get_model(checkpoint_path: Optional[str] = None):
    return _classifier_module().get_model(checkpoint_path)


def run_chexnet(roi, checkpoint_path: Optional[str] = None, top_k: int = 3, confidence_threshold: float = 0.5):
    return classify_image(roi, checkpoint_path, top_k, confidence_threshold)


def run_chexnet_batch(rois: list, checkpoint_path: Optional[str] = None) -> list:
    return [classify_image(roi, checkpoint_path=checkpoint_path) for roi in rois]


CheXNetModel = CheXNet
ImageClassification = object


__all__ = [
    "CHEXNET_CLASSES",
    "CheXNet",
    "CheXNetModel",
    "DEFAULT_CHECKPOINT",
    "ImageClassification",
    "classify_image",
    "get_model",
    "load_chexnet",
    "run_chexnet",
    "run_chexnet_batch",
]
