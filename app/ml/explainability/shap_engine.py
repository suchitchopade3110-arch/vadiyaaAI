"""SHAP explanation entry point for tabular predictions."""


def get_shap_explanation(features: dict) -> dict:
    """Return SHAP values and top factors for the same features used by predict()."""
    from app.ml.predictor import predict_safe

    result = predict_safe(features)
    return {
        "shap_values": result.get("shap_values", {}),
        "top_factors": result.get("top_factors", []),
        "risk_score": result.get("risk_score"),
        "label": result.get("label"),
    }


__all__ = ["get_shap_explanation"]
