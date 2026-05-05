from app.core.config import settings

MEDICAL_DISCLAIMER = settings.MEDICAL_DISCLAIMER

UNCERTAINTY_MESSAGE = (
    "Low confidence — findings require clinical review before use."
)

def wrap_with_disclaimer(data: dict) -> dict:
    data["medical_disclaimer"] = MEDICAL_DISCLAIMER
    return data
