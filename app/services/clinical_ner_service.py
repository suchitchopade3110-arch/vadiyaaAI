"""Clinical NER and feature extraction via Groq Llama-3."""

from __future__ import annotations

import json
import os
import re
from typing import Any

from app.services.urinalysis_knowledge_base import URINE_NER_PROMPT
from app.utils.retry import with_retry

MODEL = os.getenv("GROQ_NER_MODEL", os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"))


NER_PROMPT = """You are a clinical NER system. Extract ALL medical entities from the text below.
Return ONLY valid JSON, no markdown, no explanation.

Schema:
{{
  "conditions": ["string"],
  "medications": [{{"name": "string", "dose": "string", "frequency": "string"}}],
  "lab_values": [{{"name": "string", "value": number, "unit": "string", "reference_range": "string", "status": "normal|high|low|critical"}}],
  "vitals": [{{"name": "string", "value": "string", "unit": "string"}}],
  "symptoms": ["string"],
  "procedures": ["string"]
}}

Clinical Text:
{text}"""


FEATURE_PROMPT = """You are a medical feature extraction system for XGBoost.
Convert NER data to flat numeric features. Return ONLY valid JSON, no markdown.
Use -1 for missing lab values. Binary flags: 1=present, 0=absent.

Schema:
{{
  "hemoglobin": float, "wbc": float, "platelets": float,
  "fasting_glucose": float, "hba1c": float, "creatinine": float,
  "alt": float, "bilirubin": float, "cholesterol": float,
  "ldl": float, "hdl": float, "systolic_bp": float,
  "diastolic_bp": float, "heart_rate": float, "spo2": float,
  "has_diabetes": int, "has_hypertension": int, "has_anemia": int,
  "has_dyslipidemia": int, "has_kidney_disease": int,
  "has_liver_disease": int, "has_cardiac_issue": int,
  "on_metformin": int, "on_statin": int,
  "on_antihypertensive": int, "on_aspirin": int,
  "abnormal_count": int, "critical_count": int
}}

NER Data:
{ner_data}"""

_client = None


def _get_client():
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set")
        from groq import Groq

        _client = Groq(api_key=api_key)
    return _client


@with_retry(max_retries=2, backoff_seconds=2.0)
def _call_groq(prompt: str, temperature: float = 0.1) -> str:
    response = _get_client().chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=2048,
    )
    return response.choices[0].message.content.strip()


def _parse_json(raw: str) -> dict[str, Any]:
    clean = raw.strip().replace("```json", "").replace("```", "").strip()
    if not clean.startswith("{"):
        match = re.search(r"\{.*\}", clean, flags=re.DOTALL)
        if match:
            clean = match.group(0)
    return json.loads(clean)


def extract_entities(text: str) -> dict[str, Any]:
    raw = _call_groq(NER_PROMPT.format(text=text[:12000]))
    return _normalize_entities(_parse_json(raw))


def extract_features(ner_data: dict[str, Any]) -> dict[str, Any]:
    raw = _call_groq(FEATURE_PROMPT.format(ner_data=json.dumps(ner_data, indent=2)))
    return _parse_json(raw)


def run_clinical_pipeline(text: str) -> dict[str, Any]:
    report_type = detect_report_type(text)
    if report_type == "urinalysis":
        return run_urinalysis_pipeline(text)

    entities = extract_entities(text)
    features = extract_features(entities)
    abnormal = [
        lab_value
        for lab_value in entities.get("lab_values", [])
        if str(lab_value.get("status", "")).lower() != "normal"
    ]
    critical = [
        lab_value
        for lab_value in entities.get("lab_values", [])
        if str(lab_value.get("status", "")).lower() == "critical"
    ]
    return {
        "entities": entities,
        "features": features,
        "abnormal_count": len(abnormal),
        "critical_count": len(critical),
        "model_used": f"groq/{MODEL}",
    }


def detect_report_type(text: str) -> str:
    """Auto-detect urinalysis vs standard lab report."""
    urine_keywords = [
        "urinalysis",
        "urine",
        "dipstick",
        "specific gravity",
        "nitrite",
        "leukocyte esterase",
        "ketones",
        "hpf",
        "casts",
        "sediment",
        "protein/creatinine",
        "albumin/creatinine",
        "urine culture",
    ]
    lowered = text.lower()
    hits = sum(1 for keyword in urine_keywords if keyword in lowered)
    return "urinalysis" if hits >= 2 else "lab"


def run_urinalysis_pipeline(text: str) -> dict[str, Any]:
    """Urinalysis NER + pattern extraction + RAG evidence."""
    try:
        raw = _call_groq(URINE_NER_PROMPT.format(text=text[:12000]))
        entities = _normalize_urine_entities(_parse_json(raw))
    except Exception:
        entities = _extract_urine_entities_regex(text)

    patterns = _apply_urine_pattern_rules(entities)
    entities["detected_patterns"] = patterns
    evidence_query = " ".join(patterns) + " " + text[:250]
    try:
        from app.services.ingest_urinalysis_kb import query_urinalysis_kb

        evidence = query_urinalysis_kb(evidence_query, n_results=3)
    except Exception:
        evidence = []

    return {
        "entities": entities,
        "features": _urine_to_features(entities),
        "report_type": "urinalysis",
        "rag_evidence": evidence,
        "abnormal_count": _urine_abnormal_count(entities),
        "critical_count": _urine_critical_count(patterns),
        "model_used": f"groq/{MODEL}" if os.getenv("GROQ_API_KEY") else "regex-urinalysis",
    }


def _normalize_entities(data: dict[str, Any]) -> dict[str, Any]:
    data.setdefault("conditions", [])
    data.setdefault("medications", [])
    data.setdefault("lab_values", [])
    data.setdefault("vitals", [])
    data.setdefault("symptoms", [])
    data.setdefault("procedures", [])

    normalized_labs = []
    for item in data.get("lab_values") or []:
        if not isinstance(item, dict):
            continue
        normalized_labs.append(
            {
                "name": str(item.get("name", "")).strip(),
                "value": _coerce_float(item.get("value")),
                "unit": str(item.get("unit", "")).strip(),
                "reference_range": str(item.get("reference_range", item.get("reference", ""))).strip(),
                "status": str(item.get("status", "normal")).lower().strip() or "normal",
            }
        )
    data["lab_values"] = normalized_labs
    return data


def _coerce_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        match = re.search(r"-?\d+(?:\.\d+)?", str(value))
        return float(match.group(0)) if match else None


def _normalize_urine_entities(data: dict[str, Any]) -> dict[str, Any]:
    data.setdefault("report_type", "urinalysis")
    data.setdefault("dipstick", {})
    data.setdefault("microscopy", {})
    data.setdefault("quantitative", {})
    data.setdefault("detected_patterns", [])
    data.setdefault("conditions", [])
    data.setdefault("urgency", "routine")
    data["microscopy"].setdefault("casts", {})
    return data


def _extract_urine_entities_regex(text: str) -> dict[str, Any]:
    lowered = text.lower()

    def value_for(label: str) -> str:
        patterns = [
            rf"{label}\s*[:\-]?\s*([a-z0-9.+/\- ]+)",
            rf"{label.replace('_', ' ')}\s*[:\-]?\s*([a-z0-9.+/\- ]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, lowered)
            if match:
                return match.group(1).split(",")[0].split(";")[0].strip()
        return ""

    def flag(value: str, normal_values: tuple[str, ...] = ("negative", "absent", "normal", "trace")) -> str:
        if not value:
            return "normal"
        value_lower = value.lower()
        if any(marker in value_lower for marker in ("positive", "present", "2+", "3+", "4+", "large", "many", "moderate")):
            return "abnormal"
        if any(marker in value_lower for marker in normal_values):
            return "normal"
        numeric = _coerce_float(value_lower)
        return "abnormal" if numeric and numeric > 0 else "normal"

    dipstick = {}
    for name in [
        "glucose",
        "protein",
        "ketones",
        "blood",
        "nitrite",
        "leukocyte_esterase",
        "bilirubin",
        "urobilinogen",
        "ph",
        "specific_gravity",
    ]:
        raw = value_for(name)
        dipstick[name] = {"value": raw, "flag": flag(raw)}

    wbc = value_for("wbc")
    rbc = value_for("rbc")
    casts_text = value_for("casts") or lowered
    full_cast_text = f"{casts_text} {lowered}"
    microscopy = {
        "wbc": {"value": wbc, "unit": "/HPF", "flag": "abnormal" if (_coerce_float(wbc) or 0) > 5 else "normal"},
        "rbc": {"value": rbc, "unit": "/HPF", "flag": "abnormal" if (_coerce_float(rbc) or 0) > 3 else "normal"},
        "casts": {
            "rbc_casts": "present" if "rbc cast" in full_cast_text or "red blood cell cast" in full_cast_text else "absent",
            "wbc_casts": "present" if "wbc cast" in full_cast_text or "white blood cell cast" in full_cast_text else "absent",
            "granular_casts": "present" if "granular cast" in full_cast_text else "absent",
        },
        "bacteria": "present" if "bacteria" in lowered and "absent" not in lowered else "absent",
        "epithelial_cells": value_for("epithelial cells"),
    }
    return _normalize_urine_entities(
        {
            "report_type": "urinalysis",
            "dipstick": dipstick,
            "microscopy": microscopy,
            "quantitative": {},
            "detected_patterns": [],
            "conditions": [],
            "urgency": "routine",
        }
    )


def _apply_urine_pattern_rules(entities: dict[str, Any]) -> list[str]:
    dipstick = entities.get("dipstick", {})
    microscopy = entities.get("microscopy", {})
    casts = microscopy.get("casts", {})
    patterns = []

    nitrite = _value(dipstick, "nitrite")
    leukocyte_esterase = _value(dipstick, "leukocyte_esterase")
    glucose_flag = _flag(dipstick, "glucose")
    ketones = _value(dipstick, "ketones")
    protein = _value(dipstick, "protein")
    blood = _value(dipstick, "blood")
    bilirubin = _value(dipstick, "bilirubin")
    urobilinogen = _value(dipstick, "urobilinogen")
    urobilinogen_flag = _flag(dipstick, "urobilinogen")
    rbc_casts = str(casts.get("rbc_casts", "")).lower()
    wbc_value = _coerce_float((microscopy.get("wbc") or {}).get("value")) or 0

    if "positive" in nitrite and "positive" in leukocyte_esterase:
        patterns.append("UTI")
    if glucose_flag == "abnormal" and _is_two_plus_or_more(ketones):
        patterns.append("DKA")
    if _is_three_plus_or_more(protein):
        patterns.append("Nephrotic")
    if "present" in rbc_casts or ("positive" in blood and "cast" in rbc_casts):
        patterns.append("Nephritic")
    if "positive" in bilirubin and "absent" in urobilinogen:
        patterns.append("Obstructive_jaundice")
    if (urobilinogen_flag == "abnormal" or (_coerce_float(urobilinogen) or 0) > 1) and "negative" in bilirubin:
        patterns.append("Hemolytic")
    if wbc_value > 5 and "negative" in nitrite:
        patterns.append("Renal_TB_suspected")
    return patterns


def _value(container: dict[str, Any], key: str) -> str:
    return str((container.get(key) or {}).get("value", "")).lower()


def _flag(container: dict[str, Any], key: str) -> str:
    return str((container.get(key) or {}).get("flag", "")).lower()


def _is_two_plus_or_more(value: str) -> bool:
    return any(marker in str(value).lower() for marker in ("2+", "3+", "4+", "large", "moderate"))


def _is_three_plus_or_more(value: str) -> bool:
    return any(marker in str(value).lower() for marker in ("3+", "4+", "large"))


def _urine_to_features(entities: dict[str, Any]) -> dict[str, Any]:
    dipstick = entities.get("dipstick", {})
    microscopy = entities.get("microscopy", {})
    casts = microscopy.get("casts", {})
    patterns = entities.get("detected_patterns", [])

    return {
        "urine_glucose_abnormal": int(_flag(dipstick, "glucose") == "abnormal"),
        "urine_protein_abnormal": int(_flag(dipstick, "protein") == "abnormal"),
        "urine_ketones_abnormal": int(_flag(dipstick, "ketones") == "abnormal"),
        "urine_blood_abnormal": int(_flag(dipstick, "blood") == "abnormal"),
        "urine_nitrite_positive": int("positive" in _value(dipstick, "nitrite")),
        "urine_le_positive": int("positive" in _value(dipstick, "leukocyte_esterase")),
        "urine_bilirubin_positive": int("positive" in _value(dipstick, "bilirubin")),
        "urine_rbc_casts": int("present" in str(casts.get("rbc_casts", "")).lower()),
        "has_uti_pattern": int("UTI" in patterns),
        "has_dka_pattern": int("DKA" in patterns),
        "has_nephrotic_pattern": int("Nephrotic" in patterns),
        "has_nephritic_pattern": int("Nephritic" in patterns),
        "has_renal_tb_suspected": int("Renal_TB_suspected" in patterns),
        "has_obstructive_jaundice": int("Obstructive_jaundice" in patterns),
        "abnormal_count": _urine_abnormal_count(entities),
        "pattern_count": len(patterns),
    }


def _urine_abnormal_count(entities: dict[str, Any]) -> int:
    dipstick = entities.get("dipstick", {})
    return sum(
        1
        for key in ["glucose", "protein", "ketones", "blood", "bilirubin", "nitrite", "leukocyte_esterase"]
        if _flag(dipstick, key) == "abnormal" or "positive" in _value(dipstick, key)
    )


def _urine_critical_count(patterns: list[str]) -> int:
    return sum(1 for pattern in patterns if pattern in {"DKA", "Pre_eclampsia"})


if __name__ == "__main__":
    sample = """
    Patient: Male, 54. Hemoglobin: 9.2 g/dL (LOW). Fasting Blood Sugar: 168 mg/dL (HIGH).
    HbA1c: 7.3% (HIGH). Platelets: 95 (CRITICAL). ALT: 88 (HIGH).
    Medications: Metformin 500mg BD, Atorvastatin 20mg OD.
    Diagnosis: Type 2 Diabetes, Anemia, Thrombocytopenia.
    """
    result = run_clinical_pipeline(sample)
    print(f"Conditions   : {result['entities']['conditions']}")
    print(f"Medications  : {[m.get('name') for m in result['entities']['medications']]}")
    print(f"Lab values   : {len(result['entities']['lab_values'])} extracted")
    print(f"Abnormal     : {result['abnormal_count']} | Critical: {result['critical_count']}")
    print(f"Feature vec  : {len(result['features'])} features")
    print(f"Model        : {result['model_used']}")
