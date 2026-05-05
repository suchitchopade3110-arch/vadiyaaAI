from types import SimpleNamespace

from app.services.ml_service import ml_predictor


def predict(features: dict) -> object:
    """
    XGBoost on ClinicalBERT features -> risk score.

    Current behavior delegates to the existing rule-based predictor while
    preserving the skeleton's object-shaped return contract.
    """
    result = ml_predictor.predict_risk(features)
    risk_score = result["risk_score"]
    if risk_score >= 70:
        risk_label = "high_risk"
    elif risk_score >= 35:
        risk_label = "moderate_risk"
    else:
        risk_label = "low_risk"
    return SimpleNamespace(
        risk_score=risk_score,
        risk_label=risk_label,
        shap_values=result["shap_values"],
        risk_factors=result["risk_factors"],
    )

