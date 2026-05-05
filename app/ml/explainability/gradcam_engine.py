"""GradCAM adapter for image explainability."""


def generate_gradcam(roi, label: str) -> str:
    """Generate a GradCAM heatmap path when the image pipeline deps are available."""
    try:
        from app.image_pipeline.gradcam import generate_gradcam as _generate_gradcam
    except Exception:
        return ""
    return _generate_gradcam(roi, label)


class GradCAM:
    """Minimal callable GradCAM wrapper around the current pipeline hook."""

    def __init__(self, model=None):
        self.model = model

    def generate(self, roi, label: str) -> str:
        return generate_gradcam(roi, label)

    def __call__(self, roi, label: str) -> str:
        return self.generate(roi, label)


__all__ = ["GradCAM", "generate_gradcam"]
