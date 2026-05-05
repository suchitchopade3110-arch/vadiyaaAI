"""Reference-range based lab anomaly flagging."""

LAB_RANGES = {
    "glucose": {"all": (70, 110), "unit": "mg/dL"},
    "hemoglobin": {"all": (12, 17.5), "unit": "g/dL"},
    "cholesterol": {"all": (0, 200), "unit": "mg/dL"},
    "creatinine": {"all": (0.6, 1.2), "unit": "mg/dL"},
    "tsh": {"all": (0.4, 4.0), "unit": "mIU/L"},
    "vitamin d": {"all": (20, 50), "unit": "ng/mL"},
}


def _value_from_item(item):
    if hasattr(item, "name") and hasattr(item, "value"):
        return item.name, item.value, item.unit, item.ref_low, item.ref_high
    if isinstance(item, dict):
        return (
            item.get("name") or item.get("test") or item.get("field"),
            item.get("value") if item.get("value") is not None else item.get("result"),
            item.get("unit", ""),
            item.get("ref_low"),
            item.get("ref_high"),
        )
    return None, None, "", None, None


def flag_anomalies(lab_values, gender: str = "male", age: int = 40) -> list[dict]:
    """Flag values outside known or supplied reference ranges."""
    try:
        from app.services.preprocessor import LAB_REGISTRY, resolve_reference
    except Exception:
        LAB_REGISTRY = LAB_RANGES

        def resolve_reference(name: str, gender: str = "male", age: int = 40, pdf_ref_low=None, pdf_ref_high=None):
            if pdf_ref_low is not None and pdf_ref_high is not None:
                return pdf_ref_low, pdf_ref_high, "pdf"
            config = LAB_REGISTRY.get(str(name).lower())
            if not config:
                return None, None, "unknown"
            low, high = config.get(gender, config.get("all"))
            return low, high, "fallback"

    anomalies = []
    for item in lab_values or []:
        name, value, unit, ref_low, ref_high = _value_from_item(item)
        if name is None or value is None:
            continue
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            continue

        low, high, source = resolve_reference(name, gender=gender, age=age, pdf_ref_low=ref_low, pdf_ref_high=ref_high)
        if low is None or high is None:
            continue

        direction = None
        if numeric_value < low:
            direction = "LOW"
        elif numeric_value > high:
            direction = "HIGH"
        if not direction:
            continue

        anomalies.append(
            {
                "field": name,
                "value": numeric_value,
                "unit": unit,
                "reference": f"{low}-{high}",
                "direction": direction,
                "severity": "HIGH",
                "source": source,
            }
        )
    return anomalies


__all__ = ["LAB_RANGES", "flag_anomalies"]
