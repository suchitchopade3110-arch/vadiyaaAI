"""Simple tabular + image ensemble combiner."""

from typing import Any, Optional

from app.ml.predictors.tabular_predictor import predict as predict_tabular


def _as_dict(value: Any) -> dict:
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    if isinstance(value, dict):
        return value
    return {
        key: getattr(value, key)
        for key in dir(value)
        if not key.startswith("_") and not callable(getattr(value, key))
    }


def ensemble_predict(
    tabular_features: Optional[dict] = None,
    image_result: Optional[Any] = None,
    tabular_weight: float = 0.7,
    image_weight: float = 0.3,
) -> dict:
    """Blend tabular risk with optional image-classifier confidence."""
    tabular = predict_tabular(tabular_features or {}) if tabular_features is not None else {}
    image = _as_dict(image_result)

    tabular_score = float(tabular.get("risk_score", tabular.get("confidence", 0.0)))
    image_score = float(image.get("confidence", image.get("top_confidence", 0.0))) * 100.0

    if not image:
        combined = tabular_score
    else:
        total_weight = tabular_weight + image_weight
        combined = ((tabular_score * tabular_weight) + (image_score * image_weight)) / total_weight

    return {
        "label": "high_risk" if combined >= 50.0 else "low_risk",
        "risk_score": round(combined, 1),
        "tabular": tabular,
        "image": image,
        "weights": {"tabular": tabular_weight, "image": image_weight},
    }


__all__ = ["ensemble_predict"]

