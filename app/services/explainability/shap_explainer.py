def compute_shap(prediction) -> dict | None:
    return getattr(prediction, "shap_values", None)

