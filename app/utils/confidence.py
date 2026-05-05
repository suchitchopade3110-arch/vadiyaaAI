import math

def platt_scale(raw: float) -> float:
    """Always returns 0–100. Input: 0–1."""
    raw = max(0.0, min(1.0, float(raw)))
    scaled = 1.0 / (1.0 + math.exp(2.0 * raw - 1.0))
    return round(scaled * 100, 1)
