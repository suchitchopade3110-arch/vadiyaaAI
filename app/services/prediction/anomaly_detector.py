from app.services.ml_service import ml_predictor


def detect(lab_values: dict) -> list[dict]:
    return ml_predictor.detect_anomalies(lab_values)

