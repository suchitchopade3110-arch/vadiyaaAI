import logging
from typing import Any

logger = logging.getLogger(__name__)

# ── Reference ranges for anomaly detection ────────────────────────────────────
REFERENCE_RANGES = {
    "HBA1C":       (0.0, 5.6,  "moderate", "%"),
    "GLUCOSE":     (70,  99,   "moderate", "mg/dL"),
    "CREATININE":  (0.6, 1.2,  "mild",     "mg/dL"),
    "HEMOGLOBIN":  (12,  17,   "mild",     "g/dL"),
    "CHOLESTEROL": (0,   200,  "mild",     "mg/dL"),
    "TSH":         (0.4, 4.0,  "mild",     "mIU/L"),
    "WBC":         (4.5, 11.0, "moderate", "10³/μL"),
    "PLATELETS":   (150, 400,  "moderate", "10³/μL"),
    "SODIUM":      (136, 145,  "moderate", "mEq/L"),
    "POTASSIUM":   (3.5, 5.0,  "moderate", "mEq/L"),
    "ALT":         (7,   56,   "mild",     "U/L"),
    "AST":         (10,  40,   "mild",     "U/L"),
    "INR":         (0.8, 1.1,  "moderate", ""),
}

# ── Risk feature weights (Phase 1 stub) ──────────────────────────────────────
CONDITION_RISK_WEIGHTS = {
    "diabetes": 15, "hypertension": 12, "cancer": 25, "heart failure": 20,
    "copd": 18, "ckd": 22, "stroke": 20, "sepsis": 30, "obesity": 8,
    "anemia": 10, "depression": 5,
}


class MLPredictor:
    """
    Risk prediction + explainability service.
    Phase 1: Rule-based scoring (correct output schema).
    Phase 2: Swap predict_risk() internals with XGBoost + Platt.
    """

    def predict_risk(self, entities: dict) -> dict:
        """
        Main entry. Returns risk_score, risk_factors, shap_values.
        
        Input:  entities dict from ClinicalBERT (conditions, medications, lab_values)
        Output: {risk_score, risk_factors, shap_values}
        """
        conditions  = entities.get("conditions", [])
        lab_values  = entities.get("lab_values", {})

        score = 0.0
        factors = []

        for condition in conditions:
            weight = CONDITION_RISK_WEIGHTS.get(condition.lower(), 5)
            score += weight
            factors.append(condition.title())

        # Lab value anomalies add to score
        anomalies = self.detect_anomalies(lab_values)
        for anomaly in anomalies:
            if anomaly["severity"] == "severe":   score += 15
            elif anomaly["severity"] == "moderate": score += 8
            else:                                   score += 3
            factors.append(f"Abnormal {anomaly['parameter']}: {anomaly['value']}")

        risk_score = min(100.0, round(score, 1))

        # Stub SHAP — proportional to condition weights (Phase 2: real SHAP)
        shap_values = {
            c: round(CONDITION_RISK_WEIGHTS.get(c.lower(), 5) / 100, 3)
            for c in conditions
        }
        shap_values["_source"] = "rule-based-stub-phase1"

        return {
            "risk_score":   risk_score,
            "risk_factors": factors,
            "shap_values":  shap_values,
        }

    def detect_anomalies(self, lab_values: dict) -> list[dict]:
        """
        Flag lab values outside reference ranges.
        Returns list of AnomalyFlag dicts matching API contract schema.
        """
        anomalies = []

        for param, raw_value_dict in lab_values.items():
            key = param.upper().replace(" ", "")
            if key not in REFERENCE_RANGES:
                continue
                
            raw_value = raw_value_dict.get("value", "") if isinstance(raw_value_dict, dict) else str(raw_value_dict)

            low, high, default_severity, unit = REFERENCE_RANGES[key]

            # Parse numeric value from string like "7.8%" or "142 mg/dL"
            numeric = self._parse_numeric(str(raw_value))
            if numeric is None:
                continue

            if numeric < low or numeric > high:
                deviation = max(
                    abs(numeric - low) / max(low, 0.001),
                    abs(numeric - high) / max(high, 0.001),
                )
                severity = (
                    "severe"   if deviation > 0.5 else
                    "moderate" if deviation > 0.2 else
                    "mild"
                )
                
                # Update the original lab value dict if we can
                if isinstance(raw_value_dict, dict):
                    raw_value_dict["abnormal"] = True
                    raw_value_dict["reference_range"] = f"{low}–{high} {unit}".strip()
                    
                anomalies.append({
                    "parameter":       param.upper(),
                    "value":           str(raw_value),
                    "reference_range": f"{low}–{high} {unit}".strip(),
                    "severity":        severity,
                })

        return anomalies

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _parse_numeric(self, value: str) -> float | None:
        """Extract first numeric from string like '7.8%' or '142 mg/dL'."""
        import re
        match = re.search(r"[\d.]+", value)
        if match:
            try:
                return float(match.group())
            except ValueError:
                pass
        return None


# ── Singleton ─────────────────────────────────────────────────────────────────
ml_predictor = MLPredictor()
