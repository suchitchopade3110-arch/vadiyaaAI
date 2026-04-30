import re
from typing import Any

# ── Phase 2: Uncomment when wiring real model ─────────────────────────────────
# from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
# _NER_MODEL = "d4data/biomedical-ner-all"
# _ner_pipeline = pipeline("ner", model=_NER_MODEL, aggregation_strategy="simple")


# ── Keyword banks (Phase 1 stub) ──────────────────────────────────────────────
_CONDITION_KEYWORDS = [
    "diabetes", "hypertension", "pneumonia", "cancer", "asthma", "copd",
    "heart failure", "stroke", "sepsis", "anemia", "obesity", "depression",
    "anxiety", "tuberculosis", "hepatitis", "cirrhosis", "arthritis",
    "hypothyroidism", "hyperthyroidism", "ckd", "chronic kidney disease",
    "fever", "cough", "chest pain"
]

_MED_KEYWORDS = [
    "aspirin", "metformin", "insulin", "lisinopril", "atorvastatin",
    "amlodipine", "metoprolol", "omeprazole", "amoxicillin", "paracetamol",
    "ibuprofen", "warfarin", "clopidogrel", "losartan", "furosemide",
    "prednisolone", "salbutamol", "ciprofloxacin", "azithromycin",
]

_PROCEDURE_KEYWORDS = [
    "ecg", "ekg", "x-ray", "xray", "ct scan", "mri", "ultrasound", "biopsy",
    "colonoscopy", "endoscopy", "catheterization", "dialysis", "surgery",
    "angioplasty", "intubation", "bronchoscopy",
]

# Lab value pattern: "HbA1c 7.8%" / "Glucose: 142 mg/dL" / "BP 130/80"
_LAB_PATTERN = re.compile(
    r'(HbA1c|glucose|hemoglobin|creatinine|urea|potassium|sodium|cholesterol'
    r'|triglycerides|ALT|AST|TSH|T3|T4|WBC|RBC|platelets|BP|blood pressure'
    r'|SpO2|eGFR|INR|PSA)\s*[:\s]*([0-9]+\.?[0-9]*\s*[%a-zA-Z0-9/]*)',
    re.IGNORECASE
)


class ClinicalBERTService:
    """
    NER extraction service.
    Phase 1: regex + keyword matching.
    Phase 2: swap _extract_* methods with real model calls.
    """

    def extract_entities(self, text: str) -> dict:
        """
        Main entry point. Returns entities dict matching API contract schema.
        Called by: claim_tasks, report_tasks Celery workers.
        """
        if not text or len(text.strip()) < 5:
            return self._empty()

        text_lower = text.lower()

        return {
            "conditions":  self._extract_conditions(text_lower),
            "medications": self._extract_medications(text_lower),
            "lab_values":  self._extract_lab_values(text),
            "procedures":  self._extract_procedures(text_lower),
            "_source": "keyword-stub-phase1",   # Remove in Phase 2
        }

    def _extract_conditions(self, text: str) -> list[str]:
        return [kw for kw in _CONDITION_KEYWORDS if kw in text]

    def _extract_medications(self, text: str) -> list[str]:
        return [kw for kw in _MED_KEYWORDS if kw in text]

    def _extract_procedures(self, text: str) -> list[str]:
        return [kw for kw in _PROCEDURE_KEYWORDS if kw in text]

    def _extract_lab_values(self, text: str) -> dict[str, Any]:
        matches = _LAB_PATTERN.findall(text)
        return {name.upper(): {"value": value.strip()} for name, value in matches}

    def _empty(self) -> dict:
        return {
            "conditions": [], "medications": [],
            "lab_values": {}, "procedures": [],
            "_source": "keyword-stub-phase1",
        }

# ── Singleton ─────────────────────────────────────────────────────────────────
clinicalbert_service = ClinicalBERTService()
