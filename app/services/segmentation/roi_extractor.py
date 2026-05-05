"""ROI metadata helpers for segmentation results."""


def create_roi_metadata(result) -> dict:
    """Build JSON-safe ROI metadata from a SegmentationResult-like object."""
    metadata = dict(getattr(result, "metadata", {}) or {})
    metadata.update(
        {
            "bbox": list(getattr(result, "bbox", []) or []),
            "confidence": float(getattr(result, "confidence", 0.0) or 0.0),
            "num_contours": len(getattr(result, "contours", []) or []),
            "roi_crop_shape": (
                list(result.roi_crop.shape)
                if getattr(result, "roi_crop", None) is not None
                else None
            ),
        }
    )
    return metadata


__all__ = ["create_roi_metadata"]

