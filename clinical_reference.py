from __future__ import annotations

from typing import Optional, Dict, Any, List, Tuple
import math
import re


def _lab(unit: str, all_range=None, male_range=None, female_range=None, aliases: Optional[List[str]] = None) -> Dict[str, Any]:
    config: Dict[str, Any] = {"unit": unit}
    if all_range is not None:
        config["all"] = all_range
    if male_range is not None:
        config["M"] = male_range
    if female_range is not None:
        config["F"] = female_range
    if aliases:
        config["aliases"] = aliases
    return config


LAB_RANGES: Dict[str, Dict[str, Any]] = {
    "hemoglobin": _lab("g/dL", male_range=(13.5, 17.5), female_range=(12.0, 15.5), aliases=["hb", "haemoglobin"]),
    "hb": _lab("g/dL", male_range=(13.5, 17.5), female_range=(12.0, 15.5), aliases=["hemoglobin", "haemoglobin"]),
    "haemoglobin": _lab("g/dL", male_range=(13.5, 17.5), female_range=(12.0, 15.5), aliases=["hemoglobin", "hb"]),
    "rbc count": _lab("million/µL", male_range=(4.7, 6.1), female_range=(4.2, 5.4), aliases=["rbc"]),
    "rbc": _lab("million/µL", male_range=(4.7, 6.1), female_range=(4.2, 5.4), aliases=["rbc count"]),
    "wbc": _lab("/µL", all_range=(4000, 11000), aliases=["wbc count", "total wbc", "total leukocyte count", "tlc"]),
    "wbc count": _lab("/µL", all_range=(4000, 11000), aliases=["wbc"]),
    "total wbc": _lab("/µL", all_range=(4000, 11000), aliases=["wbc"]),
    "total leukocyte count": _lab("/µL", all_range=(4000, 11000), aliases=["wbc"]),
    "tlc": _lab("/µL", all_range=(4000, 11000), aliases=["wbc"]),
    "platelet count": _lab("/µL", all_range=(150000, 450000), aliases=["platelets", "plt"]),
    "platelets": _lab("/µL", all_range=(150000, 450000), aliases=["platelet count", "plt"]),
    "plt": _lab("/µL", all_range=(150000, 450000), aliases=["platelets"]),
    "hematocrit": _lab("%", male_range=(41, 53), female_range=(36, 46), aliases=["hct", "pcv"]),
    "hct": _lab("%", male_range=(41, 53), female_range=(36, 46), aliases=["hematocrit", "pcv"]),
    "pcv": _lab("%", male_range=(41, 53), female_range=(36, 46), aliases=["hematocrit", "hct"]),
    "mcv": _lab("fL", all_range=(80, 100)),
    "mch": _lab("pg", all_range=(27, 33)),
    "mchc": _lab("g/dL", all_range=(32, 36)),
    "rdw": _lab("%", all_range=(11, 15)),
    "rdw cv": _lab("%", all_range=(11.6, 14)),
    "mpv": _lab("fL", all_range=(7.5, 12.5)),
    "neutrophils": _lab("%", all_range=(40, 70)),
    "lymphocytes": _lab("%", all_range=(20, 40)),
    "monocytes": _lab("%", all_range=(2, 8)),
    "eosinophils": _lab("%", all_range=(1, 4)),
    "basophils": _lab("%", all_range=(0, 1)),
    "neutrophils absolute": _lab("/µL", all_range=(1800, 7700)),
    "lymphocytes absolute": _lab("/µL", all_range=(1000, 4800)),
    "monocytes absolute": _lab("/µL", all_range=(200, 800)),
    "eosinophils absolute": _lab("/µL", all_range=(0, 450)),
    "basophils absolute": _lab("/µL", all_range=(0, 200)),
    "fasting blood sugar": _lab("mg/dL", all_range=(70, 99), aliases=["fbs", "fasting glucose", "fasting plasma glucose", "fpg", "glucose"]),
    "fasting glucose": _lab("mg/dL", all_range=(70, 99), aliases=["fasting blood sugar", "fbs"]),
    "fasting plasma glucose": _lab("mg/dL", all_range=(70, 99), aliases=["fasting blood sugar", "fpg"]),
    "fbs": _lab("mg/dL", all_range=(70, 99), aliases=["fasting blood sugar", "glucose"]),
    "fpg": _lab("mg/dL", all_range=(70, 99), aliases=["fasting blood sugar", "glucose"]),
    "glucose": _lab("mg/dL", all_range=(70, 99), aliases=["fbs", "fasting blood sugar"]),
    "ppbs": _lab("mg/dL", all_range=(0, 140), aliases=["post prandial blood sugar", "2hr plasma glucose"]),
    "post prandial blood sugar": _lab("mg/dL", all_range=(0, 140), aliases=["ppbs"]),
    "2hr plasma glucose": _lab("mg/dL", all_range=(0, 140), aliases=["ppbs"]),
    "random blood sugar": _lab("mg/dL", all_range=(70, 140), aliases=["rbs"]),
    "rbs": _lab("mg/dL", all_range=(70, 140), aliases=["random blood sugar"]),
    "hba1c": _lab("%", all_range=(0, 5.7), aliases=["glycated hemoglobin", "glycosylated hemoglobin"]),
    "glycated hemoglobin": _lab("%", all_range=(0, 5.7), aliases=["hba1c"]),
    "glycosylated hemoglobin": _lab("%", all_range=(0, 5.7), aliases=["hba1c"]),
    "mean blood glucose": _lab("mg/dL", all_range=(70, 140)),
    "cholesterol": _lab("mg/dL", all_range=(0, 200), aliases=["total cholesterol"]),
    "total cholesterol": _lab("mg/dL", all_range=(0, 200), aliases=["cholesterol"]),
    "hdl": _lab("mg/dL", male_range=(40, 999), female_range=(50, 999), aliases=["hdl cholesterol"]),
    "hdl cholesterol": _lab("mg/dL", male_range=(40, 999), female_range=(50, 999), aliases=["hdl"]),
    "ldl": _lab("mg/dL", all_range=(0, 100), aliases=["ldl cholesterol", "direct ldl"]),
    "ldl cholesterol": _lab("mg/dL", all_range=(0, 100), aliases=["ldl"]),
    "direct ldl": _lab("mg/dL", all_range=(0, 100), aliases=["ldl"]),
    "triglycerides": _lab("mg/dL", all_range=(0, 150), aliases=["triglyceride"]),
    "triglyceride": _lab("mg/dL", all_range=(0, 150), aliases=["triglycerides"]),
    "vldl": _lab("mg/dL", all_range=(5, 40)),
    "chol/hdl ratio": _lab("", all_range=(0, 5.0)),
    "ldl/hdl ratio": _lab("", all_range=(0, 3.5)),
    "total bilirubin": _lab("mg/dL", all_range=(0.1, 1.2), aliases=["bilirubin"]),
    "bilirubin": _lab("mg/dL", all_range=(0.1, 1.2), aliases=["total bilirubin", "billrubin"]),
    "direct bilirubin": _lab("mg/dL", all_range=(0, 0.3)),
    "conjugated bilirubin": _lab("mg/dL", all_range=(0, 0.3)),
    "indirect bilirubin": _lab("mg/dL", all_range=(0, 1.1)),
    "sgpt": _lab("U/L", male_range=(7, 30), female_range=(7, 19), aliases=["alt", "alanine aminotransferase"]),
    "alt": _lab("U/L", male_range=(7, 30), female_range=(7, 19), aliases=["sgpt", "alanine aminotransferase"]),
    "alanine aminotransferase": _lab("U/L", male_range=(7, 30), female_range=(7, 19), aliases=["sgpt", "alt"]),
    "sgot": _lab("U/L", all_range=(10, 40), aliases=["ast", "aspartate aminotransferase"]),
    "ast": _lab("U/L", all_range=(10, 40), aliases=["sgot", "aspartate aminotransferase"]),
    "aspartate aminotransferase": _lab("U/L", all_range=(10, 40), aliases=["sgot", "ast"]),
    "alp": _lab("U/L", all_range=(44, 147), aliases=["alkaline phosphatase"]),
    "alkaline phosphatase": _lab("U/L", all_range=(44, 147), aliases=["alp"]),
    "albumin": _lab("g/dL", all_range=(3.5, 5.0)),
    "total protein": _lab("g/dL", all_range=(6.0, 8.3)),
    "globulin": _lab("g/dL", all_range=(2.3, 3.5)),
    "a/g ratio": _lab("", all_range=(1.3, 1.7)),
    "creatinine": _lab("mg/dL", male_range=(0.7, 1.3), female_range=(0.5, 1.1), aliases=["serum creatinine", "creatinine serum"]),
    "serum creatinine": _lab("mg/dL", male_range=(0.7, 1.3), female_range=(0.5, 1.1), aliases=["creatinine"]),
    "creatinine serum": _lab("mg/dL", male_range=(0.7, 1.3), female_range=(0.5, 1.1), aliases=["creatinine"]),
    "urea": _lab("mg/dL", all_range=(15, 45), aliases=["serum urea"]),
    "serum urea": _lab("mg/dL", all_range=(15, 45), aliases=["urea"]),
    "bun": _lab("mg/dL", all_range=(7, 20), aliases=["blood urea nitrogen"]),
    "blood urea nitrogen": _lab("mg/dL", all_range=(7, 20), aliases=["bun"]),
    "uric acid": _lab("mg/dL", male_range=(3.4, 7.0), female_range=(2.4, 6.0), aliases=["serum uric acid"]),
    "serum uric acid": _lab("mg/dL", male_range=(3.4, 7.0), female_range=(2.4, 6.0), aliases=["uric acid"]),
    "egfr": _lab("mL/min/1.73m²", all_range=(60, 999)),
    "sodium": _lab("mEq/L", all_range=(135, 145)),
    "potassium": _lab("mEq/L", all_range=(3.5, 5.0)),
    "chloride": _lab("mEq/L", all_range=(96, 106)),
    "bicarbonate": _lab("mEq/L", all_range=(22, 29)),
    "calcium": _lab("mg/dL", all_range=(8.5, 10.5)),
    "phosphorus": _lab("mg/dL", all_range=(2.5, 4.5)),
    "magnesium": _lab("mg/dL", all_range=(1.7, 2.2)),
    "tsh": _lab("mIU/L", all_range=(0.4, 4.0), aliases=["thyroid stimulating hormone"]),
    "thyroid stimulating hormone": _lab("mIU/L", all_range=(0.4, 4.0), aliases=["tsh"]),
    "t3": _lab("ng/dL", all_range=(80, 200), aliases=["triiodothyronine"]),
    "triiodothyronine": _lab("ng/dL", all_range=(80, 200), aliases=["t3"]),
    "t4": _lab("µg/dL", all_range=(5, 12), aliases=["thyroxine"]),
    "thyroxine": _lab("µg/dL", all_range=(5, 12), aliases=["t4"]),
    "free t3": _lab("pg/mL", all_range=(2.3, 4.2)),
    "free t4": _lab("ng/dL", all_range=(0.8, 1.8)),
    "vitamin d": _lab("ng/mL", all_range=(30, 100), aliases=["25(oh) vitamin d", "25-oh vitamin d"]),
    "25(oh) vitamin d": _lab("ng/mL", all_range=(30, 100), aliases=["vitamin d"]),
    "25-oh vitamin d": _lab("ng/mL", all_range=(30, 100), aliases=["vitamin d"]),
    "vitamin b12": _lab("pg/mL", all_range=(200, 900), aliases=["b12", "cobalamin"]),
    "b12": _lab("pg/mL", all_range=(200, 900), aliases=["vitamin b12"]),
    "cobalamin": _lab("pg/mL", all_range=(200, 900), aliases=["vitamin b12"]),
    "folate": _lab("ng/mL", all_range=(2.7, 17.0), aliases=["folic acid"]),
    "folic acid": _lab("ng/mL", all_range=(2.7, 17.0), aliases=["folate"]),
    "iron": _lab("µg/dL", all_range=(60, 170), aliases=["serum iron"]),
    "serum iron": _lab("µg/dL", all_range=(60, 170), aliases=["iron"]),
    "total iron binding capacity": _lab("µg/dL", all_range=(250, 370), aliases=["tibc"]),
    "tibc": _lab("µg/dL", all_range=(250, 370), aliases=["total iron binding capacity"]),
    "transferrin saturation": _lab("%", all_range=(20, 50), aliases=["tsat", "transferrin sat"]),
    "ferritin": _lab("ng/mL", male_range=(24, 200), female_range=(11, 150), aliases=["serum ferritin"]),
    "serum ferritin": _lab("ng/mL", male_range=(24, 200), female_range=(11, 150), aliases=["ferritin"]),
    "crp": _lab("mg/L", all_range=(0, 10), aliases=["c reactive protein", "c-reactive protein", "hs-crp", "high sensitivity crp"]),
    "c reactive protein": _lab("mg/L", all_range=(0, 10), aliases=["crp"]),
    "c-reactive protein": _lab("mg/L", all_range=(0, 10), aliases=["crp"]),
    "hs-crp": _lab("mg/L", all_range=(0, 3), aliases=["high sensitivity crp"]),
    "high sensitivity crp": _lab("mg/L", all_range=(0, 3), aliases=["hs-crp"]),
    "esr": _lab("mm/hr", male_range=(0, 15), female_range=(0, 20), aliases=["erythrocyte sedimentation rate"]),
    "erythrocyte sedimentation rate": _lab("mm/hr", male_range=(0, 15), female_range=(0, 20), aliases=["esr"]),
    "procalcitonin": _lab("ng/mL", all_range=(0, 0.1)),
    "pt": _lab("sec", all_range=(11, 13.5), aliases=["prothrombin time"]),
    "prothrombin time": _lab("sec", all_range=(11, 13.5), aliases=["pt"]),
    "inr": _lab("", all_range=(0.8, 1.1)),
    "aptt": _lab("sec", all_range=(25, 35), aliases=["activated partial thromboplastin time"]),
    "activated partial thromboplastin time": _lab("sec", all_range=(25, 35), aliases=["aptt"]),
    "troponin i": _lab("ng/mL", all_range=(0, 0.04)),
    "troponin t": _lab("ng/mL", all_range=(0, 0.01)),
    "hs troponin": _lab("ng/L", all_range=(0, 14)),
    "bnp": _lab("pg/mL", all_range=(0, 100)),
    "nt-probnp": _lab("pg/mL", all_range=(0, 125)),
    "d-dimer": _lab("µg/mL", all_range=(0, 0.5)),
    "ck-mb": _lab("U/L", all_range=(0, 25)),
    "microalbumin": _lab("mg/L", all_range=(0, 16.7)),
    "microalbuminuria": _lab("mg/g creatinine", all_range=(0, 30)),
    "psa": _lab("ng/mL", all_range=(0, 4.0)),
    "hb a": _lab("%", all_range=(96.8, 97.8)),
    "hb a2": _lab("%", all_range=(2.2, 3.2)),
    "hb f": _lab("%", all_range=(0, 1.0)),
    "foetal hb": _lab("%", all_range=(0, 1.0)),
    "hb s": _lab("%", all_range=(0, 0)),
    "testosterone": _lab("ng/dL", male_range=(300, 1000), female_range=(15, 70)),
    "cortisol": _lab("µg/dL", all_range=(6, 23)),
    "prolactin": _lab("ng/mL", male_range=(2, 18), female_range=(2, 29)),
    "lh": _lab("mIU/mL", all_range=(1.7, 8.6)),
    "fsh": _lab("mIU/mL", all_range=(1.5, 12.4)),
    "ige": _lab("IU/mL", all_range=(0, 87)),
    "igg": _lab("mg/dL", all_range=(700, 1600)),
    "igm": _lab("mg/dL", all_range=(40, 230)),
    "cd4": _lab("cells/mm³", all_range=(500, 1500)),
    "specific gravity": _lab("", all_range=(1.005, 1.030)),
    "urine ph": _lab("", all_range=(4.6, 8.0)),
    "apri": _lab("", all_range=(0, 1.0)),
}


CRITICAL_THRESHOLDS: List[Dict[str, Any]] = [
    {"test": "hemoglobin", "direction": "below", "value": 7.0, "message": "Severe anemia - transfusion likely required", "priority": 1},
    {"test": "platelets", "direction": "below", "value": 20000, "message": "Critical bleeding risk - possible ITP or dengue", "priority": 1},
    {"test": "platelets", "direction": "above", "value": 1000000, "message": "Extreme thrombocytosis - clotting disorder risk", "priority": 2},
    {"test": "wbc", "direction": "above", "value": 50000, "message": "Extreme leukocytosis - possible leukemia", "priority": 1},
    {"test": "wbc", "direction": "below", "value": 2000, "message": "Severe leukopenia - high infection risk", "priority": 1},
    {"test": "glucose", "direction": "above", "value": 300, "message": "Diabetic crisis - hyperglycemic emergency", "priority": 1},
    {"test": "glucose", "direction": "below", "value": 50, "message": "Severe hypoglycemia - immediate intervention", "priority": 1},
    {"test": "fbs", "direction": "above", "value": 300, "message": "Diabetic crisis - hyperglycemic emergency", "priority": 1},
    {"test": "potassium", "direction": "above", "value": 6.0, "message": "Severe hyperkalemia - cardiac arrhythmia risk", "priority": 1},
    {"test": "potassium", "direction": "below", "value": 3.0, "message": "Severe hypokalemia - cardiac and muscle risk", "priority": 1},
    {"test": "sodium", "direction": "below", "value": 125, "message": "Severe hyponatremia - seizure/coma risk", "priority": 1},
    {"test": "sodium", "direction": "above", "value": 155, "message": "Severe hypernatremia - dehydration/neurological", "priority": 1},
    {"test": "calcium", "direction": "above", "value": 13.0, "message": "Hypercalcemic crisis - cardiac arrest risk", "priority": 1},
    {"test": "calcium", "direction": "below", "value": 6.5, "message": "Severe hypocalcemia - tetany/seizure risk", "priority": 1},
    {"test": "creatinine", "direction": "above", "value": 10.0, "message": "Renal failure - dialysis likely required", "priority": 1},
    {"test": "troponin i", "direction": "above", "value": 0.04, "message": "Myocardial infarction - EMERGENCY", "priority": 1},
    {"test": "troponin t", "direction": "above", "value": 0.01, "message": "Myocardial injury - cardiac emergency", "priority": 1},
    {"test": "inr", "direction": "above", "value": 3.0, "message": "Severe bleeding risk - medication overdose", "priority": 1},
    {"test": "triglycerides", "direction": "above", "value": 500, "message": "Pancreatitis risk - immediate intervention", "priority": 1},
    {"test": "cd4", "direction": "below", "value": 200, "message": "Advanced HIV Disease - opportunistic infection risk", "priority": 1},
    {"test": "apri", "direction": "above", "value": 2.0, "message": "Cirrhosis likely (APRI score) - hepatology referral", "priority": 2},
    {"test": "bilirubin", "direction": "above", "value": 10.0, "message": "Severe jaundice - liver failure or hemolysis", "priority": 1},
    {"test": "inr", "direction": "below", "value": 0.5, "message": "Critically low INR - hypercoagulable state", "priority": 2},
]


HIGH_MEANINGS: Dict[str, str] = {
    "hemoglobin": "Dehydration, polycythemia, smoking",
    "rbc": "Polycythemia vera, dehydration",
    "wbc": "Bacterial infection, inflammation, leukemia",
    "platelets": "Reactive thrombocytosis, inflammation, clotting risk",
    "neutrophils": "Bacterial infection, stress response",
    "lymphocytes": "Viral infection, chronic lymphocytic leukemia",
    "eosinophils": "Allergy, asthma, parasitic infection",
    "monocytes": "Chronic infection, inflammatory disease",
    "glucose": "Diabetes mellitus, stress hyperglycemia",
    "hba1c": "Uncontrolled diabetes",
    "fbs": "Diabetes mellitus",
    "cholesterol": "Cardiovascular disease risk",
    "ldl": "Atherosclerosis risk",
    "triglycerides": "Metabolic syndrome, pancreatitis risk",
    "sgpt": "Liver damage, hepatitis",
    "sgot": "Liver damage, heart attack, muscle injury",
    "alp": "Liver/bile duct disease, bone disease",
    "bilirubin": "Jaundice, liver dysfunction, hemolysis",
    "creatinine": "Kidney dysfunction, reduced filtration",
    "urea": "Kidney disease, dehydration",
    "uric acid": "Gout, kidney stones, metabolic syndrome",
    "tsh": "Hypothyroidism (underactive thyroid)",
    "t3": "Hyperthyroidism",
    "t4": "Hyperthyroidism",
    "crp": "Active inflammation, infection, autoimmune disease",
    "esr": "Inflammation, infection, rheumatic disease",
    "ferritin": "Iron overload, hemochromatosis, inflammation",
    "inr": "Blood thinner overdose, bleeding risk",
    "psa": "Prostate cancer risk, prostatitis",
    "potassium": "Kidney disease, medication side effect",
    "d-dimer": "Clotting disorder, DVT, pulmonary embolism",
}

LOW_MEANINGS: Dict[str, str] = {
    "hemoglobin": "Anemia, blood loss, nutritional deficiency",
    "rbc": "Anemia, blood loss",
    "wbc": "Weak immunity, viral infection, bone marrow issue",
    "platelets": "Bleeding risk, ITP, dengue",
    "neutrophils": "Weak immunity, viral infection",
    "lymphocytes": "Immune suppression, HIV",
    "glucose": "Hypoglycemia",
    "hba1c": "Good diabetic control or hypoglycemia risk",
    "hdl": "Cardiovascular risk (good cholesterol too low)",
    "albumin": "Liver disease, kidney disease, malnutrition",
    "creatinine": "Low muscle mass, malnutrition",
    "tsh": "Hyperthyroidism (overactive thyroid)",
    "t3": "Hypothyroidism",
    "t4": "Hypothyroidism",
    "vitamin d": "Bone weakness, immune deficiency, rickets",
    "vitamin b12": "Nerve damage, megaloblastic anemia, fatigue",
    "iron": "Iron deficiency anemia",
    "ferritin": "Iron depletion, anemia developing",
    "calcium": "Hypoparathyroidism, vitamin D deficiency, bone disease",
    "potassium": "Muscle weakness, heart arrhythmia",
    "sodium": "Hyponatremia, confusion, seizures",
    "cd4": "HIV immunosuppression",
    "apri": "Low fibrosis burden",
}


def _normalize(value: Any) -> str:
    text = str(value or "").lower().strip()
    text = re.sub(r"[_\-]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _get(d: dict, keys: list) -> Optional[float]:
    for k in keys:
        if k in d:
            try:
                return float(d[k])
            except (ValueError, TypeError):
                pass
        k_lower = k.lower()
        for dk in d:
            if dk.lower() == k_lower:
                try:
                    return float(dk and d[dk])
                except (ValueError, TypeError):
                    pass
    return None


def _find_config(key: str) -> tuple[Optional[str], Optional[Dict[str, Any]]]:
    key_norm = _normalize(key)
    if key_norm in LAB_RANGES:
        return key_norm, LAB_RANGES[key_norm]
    for reg_key, reg_config in LAB_RANGES.items():
        aliases = [reg_key] + list(reg_config.get("aliases", []))
        for alias in aliases:
            alias_norm = _normalize(alias)
            if alias_norm == key_norm or alias_norm in key_norm or key_norm in alias_norm:
                return reg_key, reg_config
    return None, None


NOISE_PREFIXES = [
    "t normal range",
    "normal range",
    "status",
    "result",
    "test name",
    "bio ref",
    "biological ref",
    "reference",
    "high ",
    "low ",
    "h ",
    "l ",
]


def clean_test_name(name: str) -> str:
    """Strip OCR/table-header artifacts from a lab test name."""
    cleaned = str(name or "").replace("_", " ").replace("\n", " ").strip()
    lowered = cleaned.lower()

    for prefix in NOISE_PREFIXES:
        if lowered.startswith(prefix):
            cleaned = cleaned[len(prefix):].strip()
            lowered = cleaned.lower()

    parts = cleaned.split()
    if len(parts) > 1 and len(parts[0]) <= 2:
        cleaned = " ".join(parts[1:])

    return cleaned.strip()


def enrich_lab_with_who_ranges(lab_values: List[Any], gender: str = "M") -> List[Any]:
    """Replace PDF reference ranges with WHO canonical ranges where available."""
    enriched: List[Any] = []
    gender_key = _resolve_gender(gender)

    for lv in lab_values:
        if hasattr(lv, "name"):
            raw_name = clean_test_name(getattr(lv, "name", ""))
            setattr(lv, "name", raw_name)
            test_key = _normalize(raw_name)
            current_ref_low = getattr(lv, "ref_low", None)
            current_ref_high = getattr(lv, "ref_high", None)
            value = getattr(lv, "value", None)
        else:
            raw_name = clean_test_name((lv or {}).get("test") or (lv or {}).get("name") or "")
            test_key = _normalize(raw_name)
            current_ref_low = (lv or {}).get("ref_low")
            current_ref_high = (lv or {}).get("ref_high")
            value = (lv or {}).get("value")

        reg_key, who_config = _find_config(test_key)
        if who_config:
            if gender_key in who_config:
                lo, hi = who_config[gender_key]
            elif "all" in who_config:
                lo, hi = who_config["all"]
            else:
                lo, hi = None, None

            if lo is not None:
                lo, hi = _scale_if_thousand_unit(float(lo), float(hi), float(value or 0), who_config.get("unit", ""))
                if hasattr(lv, "name"):
                    setattr(lv, "ref_low", lo)
                    setattr(lv, "ref_high", hi)
                    setattr(lv, "unit", who_config.get("unit", getattr(lv, "unit", "")))
                    if getattr(lv, "value", None) is not None:
                        setattr(lv, "flag", None)
                        setattr(lv, "status", None)
                else:
                    lv = dict(lv or {})
                    lv["test"] = raw_name
                    lv["name"] = raw_name
                    lv["ref_low"] = lo
                    lv["ref_high"] = hi
                    lv["unit"] = who_config.get("unit", lv.get("unit", ""))
        elif hasattr(lv, "name"):
            # Still clean the name for downstream matching even without a WHO hit.
            setattr(lv, "name", raw_name)
        else:
            lv = dict(lv or {})
            lv["test"] = raw_name
            lv["name"] = raw_name

        enriched.append(lv)

    return enriched


def _resolve_gender(value: Any) -> str:
    if value is None:
        return "M"
    if isinstance(value, (int, float)):
        return "F" if int(value) == 0 else "M"
    txt = str(value).strip().upper()
    if txt in {"F", "FEMALE", "0"}:
        return "F"
    return "M"


def _scale_if_thousand_unit(lo: float, hi: float, value: float, unit: str) -> tuple[float, float]:
    unit_l = _normalize(unit)
    if hi >= 1000 and abs(value) < 1000 and any(x in unit_l for x in ["/µl", "/ul", "10^3", "thousand", "cells/µl", "cells/ul"]):
        return lo / 1000.0, hi / 1000.0
    return lo, hi


def get_hemoglobin_threshold(gender: str = "M", age: int = 40, pregnant: bool = False) -> float:
    if pregnant:
        return 11.0
    if age < 5:
        return 11.0
    if age < 12:
        return 11.5
    if age < 15:
        return 12.0
    if gender.upper() == "F":
        return 12.0
    return 13.0


def get_ferritin_threshold(crp_value: Optional[float] = None, gender: str = "M", age_group: str = "adult") -> Dict[str, float]:
    base_deficiency = 15.0 if age_group == "adult" else 12.0
    overload_threshold = 200.0 if gender == "M" else 150.0
    if crp_value is not None and crp_value > 10:
        return {
            "deficiency_threshold": 70.0 if age_group == "adult" else 30.0,
            "overload_threshold": overload_threshold,
            "adjusted_for_inflammation": True,
            "crp_used": crp_value,
        }
    return {
        "deficiency_threshold": base_deficiency,
        "overload_threshold": overload_threshold,
        "adjusted_for_inflammation": False,
        "crp_used": crp_value,
    }


def classify_diabetes(hba1c: Optional[float] = None, fbs: Optional[float] = None, rbs: Optional[float] = None) -> Optional[Dict[str, str]]:
    if hba1c is not None:
        if hba1c >= 6.5:
            return {"status": "Diabetes Mellitus", "severity": "HIGH", "who_criterion": "HbA1c ≥ 6.5%", "action": "Initiate diabetes management"}
        if hba1c >= 5.7:
            return {"status": "Pre-Diabetes", "severity": "MODERATE", "who_criterion": "HbA1c 5.7-6.4%", "action": "Lifestyle modification, monitor annually"}
    if fbs is not None:
        if fbs >= 126:
            return {"status": "Diabetes Mellitus", "severity": "HIGH", "who_criterion": "FPG ≥ 126 mg/dL", "action": "Confirm with repeat test, initiate management"}
        if fbs >= 100:
            return {"status": "Impaired Fasting Glucose", "severity": "MODERATE", "who_criterion": "FPG 100-125 mg/dL", "action": "Lifestyle changes, repeat in 3 months"}
    return None


def classify_vitamin_d(value: float) -> Dict[str, str]:
    if value < 10:
        return {"status": "Deficiency", "severity": "HIGH", "message": "Severe deficiency - rickets/osteomalacia risk"}
    if value < 20:
        return {"status": "Insufficiency", "severity": "MODERATE", "message": "Insufficient - bone health affected"}
    if value < 30:
        return {"status": "Low Normal", "severity": "LOW", "message": "Borderline - supplementation recommended"}
    if value <= 100:
        return {"status": "Sufficient", "severity": "none", "message": "Optimal vitamin D level"}
    return {"status": "Toxicity", "severity": "HIGH", "message": "Vitamin D toxicity - hypercalcemia risk"}


def _critical_override(test_key: str, value: float) -> Optional[str]:
    key = _normalize(test_key)
    for entry in CRITICAL_THRESHOLDS:
        if _normalize(entry["test"]) != key:
            continue
        if entry["direction"] == "below" and value < entry["value"]:
            return f"⚠️ CRITICAL ALERT: {entry['message']}"
        if entry["direction"] == "above" and value > entry["value"]:
            return f"⚠️ CRITICAL ALERT: {entry['message']}"
    return None


def _meaning(key: str, direction: str) -> str:
    if direction == "HIGH":
        return HIGH_MEANINGS.get(key, "Value above normal range")
    if direction == "LOW":
        return LOW_MEANINGS.get(key, "Value below normal range")
    return ""


def calculate_derived_markers(lab_dict: Dict[str, Any]) -> Dict[str, Any]:
    derived: Dict[str, Any] = {}
    total_chol = _get(lab_dict, ["cholesterol", "total cholesterol"])
    hdl = _get(lab_dict, ["hdl", "hdl cholesterol"])
    tg = _get(lab_dict, ["triglycerides", "triglyceride"])
    if all(v is not None for v in [total_chol, hdl, tg]):
        if tg < 400:
            derived["ldl_calculated"] = round(total_chol - hdl - (tg / 5), 2)
            derived["ldl_formula_valid"] = True
        else:
            derived["ldl_calculated"] = None
            derived["ldl_formula_valid"] = False
            derived["ldl_note"] = "TG >400: Friedewald invalid - measure LDL directly"

    creatinine = _get(lab_dict, ["creatinine", "serum creatinine"])
    age = _get(lab_dict, ["age"])
    gender_raw = lab_dict.get("gender", lab_dict.get("sex", "M"))
    gender = "M" if str(gender_raw) in ["1", "M", "Male", "male"] else "F"
    if creatinine and age and creatinine > 0:
        egfr = ((140 - float(age)) * 72) / (float(creatinine) * 72)
        if gender == "F":
            egfr *= 0.85
        derived["egfr"] = round(egfr, 1)
        derived["ckd_stage"] = _classify_ckd(derived["egfr"])

    ast = _get(lab_dict, ["sgot", "ast", "aspartate aminotransferase"])
    platelets = _get(lab_dict, ["platelets", "platelet count", "plt"])
    if ast and platelets and platelets > 0:
        apri = (ast / 40.0) / (platelets / 100)
        derived["apri"] = round(apri, 3)
        derived["apri_interpretation"] = "Cirrhosis likely (APRI >2.0)" if apri > 2.0 else "Significant fibrosis (APRI >1.0)" if apri > 1.0 else "No significant fibrosis"

    sbp = _get(lab_dict, ["bp_systolic"])
    dbp = _get(lab_dict, ["bp_diastolic"])
    if sbp and dbp:
        derived["pulse_pressure"] = round(float(sbp) - float(dbp), 1)

    sodium = _get(lab_dict, ["sodium"])
    chloride = _get(lab_dict, ["chloride"])
    bicarb = _get(lab_dict, ["bicarbonate"])
    if sodium and chloride and bicarb:
        derived["anion_gap"] = round(float(sodium) - float(chloride) - float(bicarb), 1)

    bun = _get(lab_dict, ["bun", "blood urea nitrogen"])
    creat = _get(lab_dict, ["creatinine"])
    if bun and creat and creat > 0:
        derived["bun_creatinine_ratio"] = round(float(bun) / float(creat), 1)

    return derived


def _classify_ckd(egfr: float) -> Dict[str, Any]:
    if egfr >= 90:
        return {"stage": 1, "label": "Normal or High", "severity": "normal", "action": "Monitor if other markers present"}
    if egfr >= 60:
        return {"stage": 2, "label": "Mildly Decreased", "severity": "low", "action": "Address risk factors"}
    if egfr >= 45:
        return {"stage": "3a", "label": "Mild-Moderate Decrease", "severity": "moderate", "action": "Nephrology monitoring recommended"}
    if egfr >= 30:
        return {"stage": "3b", "label": "Moderate-Severe Decrease", "severity": "high", "action": "Nephrology referral recommended"}
    if egfr >= 15:
        return {"stage": 4, "label": "Severely Decreased", "severity": "high", "action": "Prepare for renal replacement therapy"}
    return {"stage": 5, "label": "Kidney Failure", "severity": "critical", "action": "URGENT - dialysis or transplant required"}


def _parse_numeric(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"[\d.]+", str(value))
    if match:
        try:
            return float(match.group())
        except ValueError:
            return None
    return None


def flag_anomalies(lab_dict: Dict[str, Any], gender: str = "M", age: int = 40, crp_for_ferritin: Optional[float] = None) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    gender_key = _resolve_gender(gender if gender is not None else lab_dict.get("gender"))

    for raw_key, raw_value in lab_dict.items():
        key = _normalize(raw_key)
        if key in {"gender", "gender_encoded", "age", "sex"}:
            continue

        reg_key, config = _find_config(key)
        if not config:
            continue

        try:
            value = float(raw_value)
        except (ValueError, TypeError):
            continue

        if reg_key in {"hemoglobin", "hb", "haemoglobin"}:
            lo = get_hemoglobin_threshold(gender_key, age)
            hi = 17.5 if gender_key == "M" else 15.5
        elif reg_key in {"ferritin", "serum ferritin"}:
            ft = get_ferritin_threshold(crp_for_ferritin, gender_key)
            lo = ft["deficiency_threshold"]
            hi = ft["overload_threshold"]
        else:
            if gender_key in config:
                lo, hi = config[gender_key]
            elif "all" in config:
                lo, hi = config["all"]
            else:
                continue

        unit = config.get("unit", "")
        lo, hi = _scale_if_thousand_unit(lo, hi, value, unit)
        reference = f"{lo} - {hi}" if hi is not None else f">{lo}"

        if value < lo:
            pct = round(((lo - value) / lo) * 100, 1) if lo > 0 else 0.0
            severity = "CRITICAL" if _critical_override(reg_key or key, value) or pct > 50 else "HIGH" if pct > 25 else "MODERATE"
            direction = "LOW"
            flag = f"⬇️ {severity}"
            meaning = _meaning(reg_key or key, "LOW")
        elif value > hi:
            pct = round(((value - hi) / hi) * 100, 1) if hi > 0 else 0.0
            severity = "CRITICAL" if _critical_override(reg_key or key, value) or pct > 50 else "HIGH" if pct > 25 else "MODERATE"
            direction = "HIGH"
            flag = f"⬆️ {severity}"
            meaning = _meaning(reg_key or key, "HIGH")
        else:
            pct = 0.0
            severity = "none"
            direction = "NORMAL"
            flag = "✓ NORMAL"
            meaning = ""

        results[raw_key] = {
            "status": "ABNORMAL" if direction != "NORMAL" else "NORMAL",
            "direction": direction,
            "severity": severity,
            "value": value,
            "ref_low": lo,
            "ref_high": hi if hi is not None else None,
            "unit": unit,
            "reference": reference,
            "pct_deviation": pct,
            "flag": flag,
            "clinical_meaning": meaning,
        }
        alert = _critical_override(reg_key or key, value)
        if alert:
            results[raw_key]["critical_alert"] = alert

    return results


def check_critical_thresholds(lab_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    alerts = []
    for threshold in CRITICAL_THRESHOLDS:
        test = threshold["test"]
        direction = threshold["direction"]
        limit = threshold["value"]
        message = threshold["message"]
        priority = threshold.get("priority", 2)

        value = None
        for k, v in lab_dict.items():
            if _normalize(k) == _normalize(test):
                value = _parse_numeric(v)
                break

        if value is None:
            continue
        triggered = (direction == "below" and value < limit) or (direction == "above" and value > limit)
        if triggered:
            alerts.append({
                "test": test.title(),
                "value": value,
                "limit": limit,
                "direction": direction,
                "message": message,
                "priority": priority,
                "alert_type": "CRITICAL" if priority == 1 else "WARNING",
            })

    alerts.sort(key=lambda x: x["priority"])
    return alerts


CLINICAL_PATTERNS = [
    {"name": "Bacterial Infection", "requires": [("wbc", "HIGH"), ("neutrophils", "HIGH")], "supports": [("crp", "HIGH"), ("procalcitonin", "HIGH")], "severity": "HIGH", "action": "Blood culture recommended, antibiotic review"},
    {"name": "Viral Infection", "requires": [("lymphocytes", "HIGH")], "supports": [("wbc", "LOW"), ("esr", "HIGH")], "severity": "MODERATE", "action": "Supportive management, viral serology if indicated"},
    {"name": "Severe Sepsis", "requires": [("crp", "HIGH"), ("wbc", "HIGH")], "supports": [("procalcitonin", "HIGH")], "severity": "CRITICAL", "action": "URGENT - sepsis protocol, blood cultures, IV antibiotics"},
    {"name": "Allergy / Parasitic Infection", "requires": [("eosinophils", "HIGH")], "supports": [("ige", "HIGH")], "severity": "MODERATE", "action": "Allergy evaluation, stool examination for parasites"},
    {"name": "Iron Deficiency Anemia", "requires": [("hemoglobin", "LOW"), ("mcv", "LOW")], "supports": [("ferritin", "LOW"), ("iron", "LOW"), ("rdw", "HIGH")], "severity": "HIGH", "action": "Iron supplementation, investigate source of blood loss"},
    {"name": "Severe Anemia", "requires": [("hemoglobin", "LOW")], "supports": [], "severity": "CRITICAL", "threshold_check": ("hemoglobin", "below", 7.0), "action": "URGENT - transfusion evaluation required"},
    {"name": "Vitamin B12 / Folate Deficiency", "requires": [("hemoglobin", "LOW"), ("mcv", "HIGH")], "supports": [("vitamin b12", "LOW")], "severity": "MODERATE", "action": "B12 / folate supplementation, dietary review"},
    {"name": "Hemolytic Anemia", "requires": [("hemoglobin", "LOW"), ("bilirubin", "HIGH")], "supports": [("rdw", "HIGH")], "severity": "HIGH", "action": "Hematology referral, peripheral smear"},
    {"name": "Diabetes Mellitus", "requires": [("hba1c", "HIGH")], "supports": [("glucose", "HIGH"), ("fbs", "HIGH")], "severity": "HIGH", "action": "Initiate diabetes management, dietary counseling"},
    {"name": "Uncontrolled Diabetes", "requires": [("hba1c", "HIGH"), ("glucose", "HIGH")], "supports": [], "severity": "HIGH", "action": "Urgent diabetes review"},
    {"name": "Diabetic Dyslipidemia", "requires": [("hba1c", "HIGH"), ("triglycerides", "HIGH")], "supports": [("hdl", "LOW"), ("ldl", "HIGH")], "severity": "HIGH", "action": "Cardiovascular risk reduction"},
    {"name": "Cardiovascular Risk", "requires": [("cholesterol", "HIGH")], "supports": [("ldl", "HIGH"), ("triglycerides", "HIGH")], "severity": "MODERATE", "action": "Dietary modification, statin consideration"},
    {"name": "High Cardiovascular Risk", "requires": [("ldl", "HIGH"), ("cholesterol", "HIGH")], "supports": [("hdl", "LOW"), ("triglycerides", "HIGH")], "severity": "HIGH", "action": "Statin therapy, lifestyle modification"},
    {"name": "Hypertriglyceridemia - Pancreatitis Risk", "requires": [("triglycerides", "HIGH")], "supports": [], "threshold_check": ("triglycerides", "above", 500), "severity": "CRITICAL", "action": "URGENT - immediate triglyceride-lowering therapy"},
    {"name": "Acute Myocardial Infarction", "requires": [("troponin i", "HIGH")], "supports": [("ck-mb", "HIGH")], "severity": "CRITICAL", "action": "EMERGENCY - immediate cardiology evaluation"},
    {"name": "Kidney Dysfunction", "requires": [("creatinine", "HIGH")], "supports": [("urea", "HIGH"), ("bun", "HIGH")], "severity": "HIGH", "action": "Nephrology evaluation, eGFR calculation"},
    {"name": "Chronic Kidney Disease", "requires": [("creatinine", "HIGH"), ("urea", "HIGH")], "supports": [("potassium", "HIGH")], "severity": "HIGH", "action": "Nephrology referral, eGFR staging"},
    {"name": "Liver Damage / Hepatitis", "requires": [("sgpt", "HIGH")], "supports": [("bilirubin", "HIGH"), ("sgot", "HIGH")], "severity": "HIGH", "action": "Hepatology review"},
    {"name": "Liver Cirrhosis (APRI)", "requires": [("apri", "HIGH")], "supports": [("platelets", "LOW"), ("albumin", "LOW")], "severity": "HIGH", "action": "Hepatology referral"},
    {"name": "Cholestatic Liver Disease", "requires": [("alp", "HIGH"), ("bilirubin", "HIGH")], "supports": [("sgpt", "HIGH")], "severity": "HIGH", "action": "Biliary obstruction evaluation"},
    {"name": "Hypothyroidism", "requires": [("tsh", "HIGH")], "supports": [("t4", "LOW"), ("t3", "LOW")], "severity": "MODERATE", "action": "Levothyroxine initiation"},
    {"name": "Hyperthyroidism", "requires": [("tsh", "LOW")], "supports": [("t3", "HIGH"), ("t4", "HIGH")], "severity": "MODERATE", "action": "Thyroid scan, anti-thyroid therapy"},
    {"name": "Vitamin D Deficiency", "requires": [("vitamin d", "LOW")], "supports": [], "severity": "MODERATE", "action": "Vitamin D3 supplementation"},
    {"name": "Vitamin B12 Deficiency", "requires": [("vitamin b12", "LOW")], "supports": [], "severity": "MODERATE", "action": "B12 supplementation or injection"},
    {"name": "Gout / Hyperuricemia", "requires": [("uric acid", "HIGH")], "supports": [], "severity": "MODERATE", "action": "Low-purine diet, hydration"},
    {"name": "Advanced HIV Disease", "requires": [("cd4", "LOW")], "supports": [], "threshold_check": ("cd4", "below", 200), "severity": "CRITICAL", "action": "URGENT - ART initiation/optimization"},
    {"name": "Thrombocytopenia", "requires": [("platelets", "LOW")], "supports": [], "severity": "HIGH", "action": "Hematology evaluation"},
    {"name": "Severe Thrombocytopenia", "requires": [("platelets", "LOW")], "threshold_check": ("platelets", "below", 20000), "supports": [], "severity": "CRITICAL", "action": "URGENT - platelet transfusion evaluation"},
    {"name": "Active Systemic Inflammation", "requires": [("crp", "HIGH"), ("esr", "HIGH")], "supports": [], "severity": "MODERATE", "action": "Identify underlying cause"},
    {"name": "Metabolic Syndrome", "requires": [("triglycerides", "HIGH"), ("glucose", "HIGH")], "supports": [("hdl", "LOW"), ("cholesterol", "HIGH")], "severity": "HIGH", "action": "Weight management, dietary changes"},
]


def detect_clinical_patterns(anomalies: Dict[str, Any], critical_alerts: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    abnormal_map = {
        _normalize(k): (v.get("direction", "").upper() if isinstance(v, dict) else str(v).upper())
        for k, v in anomalies.items()
        if isinstance(v, dict) and v.get("status") == "ABNORMAL"
    }

    detected = []
    for pattern in CLINICAL_PATTERNS:
        threshold = pattern.get("threshold_check")
        if threshold:
            test, direction, limit = threshold
            current = None
            for k, v in anomalies.items():
                if _normalize(k) == _normalize(test) and isinstance(v, dict):
                    current = v.get("value")
                    break
            if current is None:
                continue
            if direction == "below" and not (float(current) < limit):
                continue
            if direction == "above" and not (float(current) > limit):
                continue
        else:
            if not all(abnormal_map.get(_normalize(test)) == dir_ for test, dir_ in pattern["requires"]):
                continue

        support_count = sum(1 for test, dir_ in pattern.get("supports", []) if abnormal_map.get(_normalize(test)) == dir_)
        total_support = len(pattern.get("supports", []))
        support_ratio = support_count / max(total_support, 1)
        confidence = round(0.65 + 0.35 * support_ratio, 2)
        evidence = [t for t, d in (pattern["requires"] + pattern.get("supports", [])) if abnormal_map.get(_normalize(t)) == d]
        detected.append({
            "condition": pattern["name"],
            "severity": pattern["severity"],
            "confidence": confidence,
            "supporting_tests": evidence,
            "action": pattern.get("action", "Consult physician"),
        })

    order = {"CRITICAL": 0, "HIGH": 1, "MODERATE": 2, "LOW": 3}
    unique = []
    seen = set()
    for d in detected:
        if d["condition"] not in seen:
            seen.add(d["condition"])
            unique.append(d)
    unique.sort(key=lambda x: order.get(x["severity"], 3))
    return unique[:10]


def analyze_blood_report(lab_dict: Dict[str, Any], gender: str = "M", age: int = 40) -> Dict[str, Any]:
    crp = None
    for k, v in lab_dict.items():
        if _normalize(k) in ("crp", "c reactive protein", "c-reactive protein"):
            crp = _parse_numeric(v)
            break

    anomalies = flag_anomalies(lab_dict, gender, age, crp)
    critical_alerts = check_critical_thresholds(lab_dict)
    conditions = detect_clinical_patterns(anomalies, critical_alerts)
    derived = calculate_derived_markers({**lab_dict, "age": age, "gender": gender})

    hba1c = _get(lab_dict, ["hba1c", "glycated hemoglobin"])
    fbs = _get(lab_dict, ["fbs", "fasting blood sugar", "fasting glucose"])
    diabetes_status = classify_diabetes(hba1c, fbs)
    vit_d = _get(lab_dict, ["vitamin d", "25(oh) vitamin d"])
    vit_d_status = classify_vitamin_d(vit_d) if vit_d is not None else None

    if critical_alerts:
        risk_level = "critical"
    elif any(c["severity"] == "CRITICAL" for c in conditions):
        risk_level = "critical"
    elif any(c["severity"] == "HIGH" for c in conditions):
        risk_level = "high"
    elif any(v.get("status") == "ABNORMAL" for v in anomalies.values()):
        risk_level = "moderate"
    else:
        risk_level = "normal"

    active_conditions = [c["condition"] for c in conditions]
    return {
        "anomalies": anomalies,
        "critical_alerts": critical_alerts,
        "conditions": conditions,
        "derived": derived,
        "diabetes_status": diabetes_status,
        "vitamin_d_status": vit_d_status,
        "risk_level": risk_level,
        "active_conditions": active_conditions,
    }


__all__ = [
    "LAB_RANGES",
    "CRITICAL_THRESHOLDS",
    "HIGH_MEANINGS",
    "LOW_MEANINGS",
    "CLINICAL_PATTERNS",
    "flag_anomalies",
    "detect_clinical_patterns",
    "check_critical_thresholds",
    "calculate_derived_markers",
    "classify_diabetes",
    "classify_vitamin_d",
    "get_ferritin_threshold",
    "get_hemoglobin_threshold",
    "analyze_blood_report",
]
