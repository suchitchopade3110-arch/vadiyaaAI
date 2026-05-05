"""Mask IoU utilities."""


def iou(mask_a, mask_b) -> float:
    """Compute intersection-over-union for two binary masks."""
    import numpy as np

    a = np.asarray(mask_a).astype(bool)
    b = np.asarray(mask_b).astype(bool)
    if a.shape != b.shape:
        raise ValueError(f"Mask shapes differ: {a.shape} != {b.shape}")
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(a, b).sum() / union)


def mean_iou(pred_masks, true_masks) -> float:
    """Compute mean IoU over aligned mask lists."""
    import numpy as np

    scores = [iou(pred, true) for pred, true in zip(pred_masks, true_masks)]
    return float(np.mean(scores)) if scores else 0.0


__all__ = ["iou", "mean_iou"]
