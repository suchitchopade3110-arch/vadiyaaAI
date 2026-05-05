"""Tabular XGBoost predictor adapter."""

FEATURE_ORDER = [
    "age",
    "glucose",
    "hemoglobin",
    "cholesterol",
    "bp_systolic",
    "bp_diastolic",
    "pulse_pressure",
    "tsh",
    "vitamin_d",
    "creatinine",
    "ldl",
    "hdl",
    "crp",
]


def predict(features: dict) -> dict:
    """Run calibrated tabular risk prediction for one patient/report."""
    from app.ml.predictor import predict_safe

    return predict_safe(features)


def predict_safe(features: dict) -> dict:
    """Backward-compatible alias with lazy ML dependency imports."""
    from app.ml.predictor import predict_safe as _predict_safe

    return _predict_safe(features)


def health_check() -> dict:
    """Check whether model artifacts and ML dependencies can load."""
    from app.ml.predictor import health_check as _health_check

    return _health_check()


__all__ = ["FEATURE_ORDER", "health_check", "predict", "predict_safe"]
