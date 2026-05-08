"""Column-aware lab PDF parser for multi-panel reports."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class LabValue:
    test: str
    result: Optional[str] = None
    unit: Optional[str] = None
    ref_low: Optional[float] = None
    ref_high: Optional[float] = None
    ref_raw: Optional[str] = None
    flag: str = "NORMAL"
    section: str = "GENERAL"


HEADER_RE = re.compile(
    "|".join(
        [
            r"SID\s*No\.?\s*:.*",
            r"Ph\s*:\s*\d{10}.*",
            r"Patient\s*ID\s*:.*",
            r"Name\s*:.*",
            r"Age\s*/\s*Sex\s*:.*",
            r"Ref\.\s*By\s*:.*",
            r"Registered\s*Date\s*:.*",
            r"Report\s*Date\s*:.*",
            r"^\s*\d+\s*/\s*\d+\s*$",
        ]
    ),
    re.IGNORECASE,
)

SECTION_RE = re.compile(
    r"^\s*(HAEMATOLOGY|DIFFERENTIAL\s*COUNT|SEROLOGY|BIOCHEMISTRY|"
    r"ELECTROLYTES|URINE\s*ANALYSIS|MACROSCOPIC|MICROSCOPIC|"
    r"BLOOD\s*GROUPING|DENGUE|ESR)\b",
    re.IGNORECASE,
)

REF_RANGE_RE = re.compile(
    r"(?P<lt>[<>])\s*(?P<single>\d+\.?\d*)"
    r"|(?P<low>\d+\.?\d*)\s*(?:-|–|to)\s*(?P<high>\d+\.?\d*)",
    re.IGNORECASE,
)
NUMERIC_RE = re.compile(r"^[:\s]*(?P<val>\d+(?:\.\d+)?)")
UNIT_RE = re.compile(
    r"\b(cells/cumm|cells/mm3|mg/dl|g/dl|mEq/L|mmol/L|ng/mL|%|mm|IU/L|grams?)\b",
    re.IGNORECASE,
)

COL_BOUNDS = {
    "test": (0, 300),
    "result": (300, 480),
    "unit": (480, 580),
    "ref": (580, 1000),
}

PANEL_KEYWORDS = {
    "urinalysis": ["urine", "pus cells", "epithelial", "specific gravity"],
    "haematology": ["haemoglobin", "wbc", "rbc", "platelet", "esr", "differential"],
    "biochemistry": ["sodium", "potassium", "creatinine", "bilirubin", "glucose", "bicarbonate"],
    "serology": ["crp", "dengue", "hiv", "hbsag", "vdrl", "widal"],
    "microbiology": ["culture", "sensitivity", "organism"],
}


def _in_col(x0: float, col: str) -> bool:
    lo, hi = COL_BOUNDS[col]
    return lo <= x0 < hi


def _clean_result(raw: str) -> str:
    return re.sub(r"^[:\s]+", "", raw or "").strip()


def _parse_ref_range(ref_raw: str) -> tuple[Optional[float], Optional[float]]:
    match = REF_RANGE_RE.search(ref_raw or "")
    if not match:
        return None, None
    if match.group("lt"):
        value = float(match.group("single"))
        return (None, value) if match.group("lt") == "<" else (value, None)
    return float(match.group("low")), float(match.group("high"))


def _determine_flag(result_str: str, ref_low: Optional[float], ref_high: Optional[float]) -> str:
    match = NUMERIC_RE.match(result_str or "")
    if not match:
        return "NORMAL"
    value = float(match.group("val"))
    if ref_low is not None and value < ref_low:
        return "ABNORMAL_LOW"
    if ref_high is not None and value > ref_high:
        return "ABNORMAL_HIGH"
    return "NORMAL"


def _extract_words_by_column(page) -> dict[int, dict[str, list[str]]]:
    lines: dict[int, dict[str, list[str]]] = {}
    for x0, y0, _x1, _y1, word, *_rest in page.get_text("words"):
        y_bucket = round(y0 / 8) * 8
        lines.setdefault(y_bucket, {"test": [], "result": [], "unit": [], "ref": []})
        for col in COL_BOUNDS:
            if _in_col(float(x0), col):
                lines[y_bucket][col].append(str(word))
                break
    return lines


def parse_lab_pdf(pdf_path: str) -> list[LabValue]:
    import fitz

    results: list[LabValue] = []
    current_section = "GENERAL"
    doc = fitz.open(pdf_path)
    try:
        for page in doc:
            col_lines = _extract_words_by_column(page)
            for y_bucket in sorted(col_lines):
                row = col_lines[y_bucket]
                test_str = " ".join(row["test"]).strip()
                res_str = _clean_result(" ".join(row["result"]).strip())
                unit_str = " ".join(row["unit"]).strip()
                ref_str = " ".join(row["ref"]).strip()

                if not test_str:
                    continue
                if HEADER_RE.search(test_str) or HEADER_RE.search(res_str):
                    continue
                if re.match(
                    r"^\s*(Method|Technical\s*Note|Test|Result|Reference|Units|Verified|End\s*Of\s*Report|Done\s*by)",
                    test_str,
                    re.IGNORECASE,
                ):
                    continue

                section_match = SECTION_RE.match(test_str)
                if section_match:
                    current_section = section_match.group(1).upper().replace(" ", "_")
                    continue
                if not res_str or res_str in {"-", "--", "—"}:
                    continue

                ref_low, ref_high = _parse_ref_range(ref_str)
                unit = unit_str
                if not unit:
                    unit_match = UNIT_RE.search(ref_str)
                    if unit_match:
                        unit = unit_match.group(0)

                results.append(
                    LabValue(
                        test=test_str,
                        result=res_str,
                        unit=unit,
                        ref_low=ref_low,
                        ref_high=ref_high,
                        ref_raw=ref_str[:120] if ref_str else None,
                        flag=_determine_flag(res_str, ref_low, ref_high),
                        section=current_section,
                    )
                )
    finally:
        doc.close()
    return results


def detect_panel_type(lab_values: list[LabValue | dict]) -> str:
    section_to_panel = {
        "haematology": "haematology",
        "differential_count": "haematology",
        "serology": "serology",
        "biochemistry": "biochemistry",
        "electrolytes": "biochemistry",
        "urine_analysis": "urinalysis",
        "macroscopic": "urinalysis",
        "microscopic": "urinalysis",
        "dengue": "serology",
        "esr": "haematology",
    }
    counts: dict[str, int] = {}
    for item in lab_values:
        if isinstance(item, dict):
            section = str(item.get("section") or "").lower()
            name = str(item.get("name") or item.get("test") or "").lower()
        else:
            section = str(item.section or "").lower()
            name = str(item.test or "").lower()
        panel = section_to_panel.get(section)
        if panel is None:
            panel = next((key for key, words in PANEL_KEYWORDS.items() if any(word in name for word in words)), "general")
        counts[panel] = counts.get(panel, 0) + 1
    non_general = {key: value for key, value in counts.items() if key != "general" and value > 0}
    if len(non_general) > 2:
        return "comprehensive"
    if non_general:
        return max(non_general, key=non_general.get)
    return "general"


def parse_lab_report_for_pipeline(pdf_path: str) -> dict:
    parsed = parse_lab_pdf(pdf_path)
    panel_type = detect_panel_type(parsed)
    lab_values = []
    for item in parsed:
        if item.ref_low is not None and item.ref_high is not None:
            reference = f"{item.ref_low} - {item.ref_high}"
        elif item.ref_high is not None:
            reference = f"< {item.ref_high}"
        elif item.ref_low is not None:
            reference = f"> {item.ref_low}"
        else:
            reference = item.ref_raw or "--"
        lab_values.append(
            {
                "name": item.test,
                "test": item.test,
                "value": item.result,
                "result": item.result,
                "unit": item.unit or "",
                "reference": reference,
                "reference_range": reference,
                "ref_low": item.ref_low,
                "ref_high": item.ref_high,
                "flag": item.flag,
                "section": item.section,
            }
        )

    abnormal_items = [
        {"name": item.test, "value": item.result, "flag": item.flag, "section": item.section}
        for item in parsed
        if item.flag != "NORMAL"
    ]
    return {
        "lab_values": lab_values,
        "panel_type": panel_type,
        "abnormal_count": len(abnormal_items),
        "normal_count": len(parsed) - len(abnormal_items),
        "abnormal_items": abnormal_items,
    }
