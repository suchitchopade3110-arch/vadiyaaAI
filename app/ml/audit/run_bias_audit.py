"""
VaidyaAI bias audit v1 runner.

PRD AC7: generate per-group metrics for domain lead review/sign-off.

Run:
    python -m app.ml.audit.run_bias_audit

Outputs:
    app/ml/audit/reports/bias_audit_v1_report.json
    app/ml/audit/reports/bias_audit_v1_report.csv
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from app.ml.audit.ground_truth import load_verified_claims, summarize_verified_claims


REPORTS_DIR = Path(__file__).resolve().parent / "reports"
JSON_PATH = REPORTS_DIR / "bias_audit_v1_report.json"
CSV_PATH = REPORTS_DIR / "bias_audit_v1_report.csv"


# Synthetic ground truth dataset.
# In production, replace this with verified labels from the claims table.
GROUND_TRUTH_RECORDS = [
    {"claim": "Elevated glucose indicates diabetes risk.", "y_true": 1, "y_pred": 1, "gender": "male", "age": 28, "ethnicity": "asian", "region": "urban"},
    {"claim": "Low hemoglobin may suggest anemia.", "y_true": 1, "y_pred": 1, "gender": "male", "age": 32, "ethnicity": "asian", "region": "urban"},
    {"claim": "High creatinine may indicate kidney dysfunction.", "y_true": 1, "y_pred": 1, "gender": "male", "age": 35, "ethnicity": "asian", "region": "rural"},
    {"claim": "TSH testing helps assess thyroid function.", "y_true": 1, "y_pred": 1, "gender": "male", "age": 25, "ethnicity": "other", "region": "urban"},
    {"claim": "High LDL increases cardiovascular risk.", "y_true": 1, "y_pred": 1, "gender": "male", "age": 38, "ethnicity": "asian", "region": "urban"},
    {"claim": "Hemoglobin of 2 g/dL is normal.", "y_true": 0, "y_pred": 0, "gender": "male", "age": 52, "ethnicity": "asian", "region": "urban"},
    {"claim": "Fasting glucose of 400 is within range.", "y_true": 0, "y_pred": 0, "gender": "male", "age": 48, "ethnicity": "other", "region": "rural"},
    {"claim": "Platelet count of 600k is always benign.", "y_true": 0, "y_pred": 1, "gender": "male", "age": 55, "ethnicity": "asian", "region": "urban"},
    {"claim": "Elevated creatinine needs monitoring.", "y_true": 1, "y_pred": 1, "gender": "male", "age": 60, "ethnicity": "asian", "region": "rural"},
    {"claim": "High TSH suggests hypothyroidism.", "y_true": 1, "y_pred": 1, "gender": "male", "age": 45, "ethnicity": "other", "region": "urban"},
    {"claim": "Low vitamin D may affect bone health.", "y_true": 1, "y_pred": 1, "gender": "male", "age": 70, "ethnicity": "asian", "region": "rural"},
    {"claim": "Sodium of 120 is within normal range.", "y_true": 0, "y_pred": 0, "gender": "male", "age": 68, "ethnicity": "asian", "region": "urban"},
    {"claim": "Creatinine 3.5 indicates normal kidney function.", "y_true": 0, "y_pred": 0, "gender": "male", "age": 75, "ethnicity": "other", "region": "rural"},
    {"claim": "High ferritin may indicate inflammation.", "y_true": 1, "y_pred": 1, "gender": "male", "age": 72, "ethnicity": "asian", "region": "urban"},
    {"claim": "WBC of 18k suggests no infection.", "y_true": 0, "y_pred": 1, "gender": "male", "age": 80, "ethnicity": "asian", "region": "rural"},
    {"claim": "Elevated glucose indicates diabetes risk.", "y_true": 1, "y_pred": 1, "gender": "female", "age": 22, "ethnicity": "asian", "region": "urban"},
    {"claim": "Low hemoglobin may suggest anemia.", "y_true": 1, "y_pred": 1, "gender": "female", "age": 30, "ethnicity": "asian", "region": "rural"},
    {"claim": "High creatinine may indicate kidney dysfunction.", "y_true": 1, "y_pred": 1, "gender": "female", "age": 27, "ethnicity": "other", "region": "urban"},
    {"claim": "Low ferritin in females indicates iron deficiency.", "y_true": 1, "y_pred": 1, "gender": "female", "age": 24, "ethnicity": "asian", "region": "urban"},
    {"claim": "TSH of 8.5 is normal thyroid function.", "y_true": 0, "y_pred": 0, "gender": "female", "age": 33, "ethnicity": "other", "region": "rural"},
    {"claim": "High LDL increases cardiovascular risk.", "y_true": 1, "y_pred": 1, "gender": "female", "age": 50, "ethnicity": "asian", "region": "urban"},
    {"claim": "HbA1c of 9.0 indicates well-controlled diabetes.", "y_true": 0, "y_pred": 0, "gender": "female", "age": 55, "ethnicity": "asian", "region": "rural"},
    {"claim": "Mammography detects early breast cancer.", "y_true": 1, "y_pred": 1, "gender": "female", "age": 48, "ethnicity": "other", "region": "urban"},
    {"claim": "Potassium 6.5 is safe and needs no intervention.", "y_true": 0, "y_pred": 0, "gender": "female", "age": 58, "ethnicity": "asian", "region": "urban"},
    {"claim": "Low HDL of 25 is protective.", "y_true": 0, "y_pred": 1, "gender": "female", "age": 44, "ethnicity": "other", "region": "rural"},
    {"claim": "Elevated glucose indicates diabetes risk.", "y_true": 1, "y_pred": 1, "gender": "female", "age": 67, "ethnicity": "asian", "region": "rural"},
    {"claim": "Bone density scan helps detect osteoporosis.", "y_true": 1, "y_pred": 1, "gender": "female", "age": 72, "ethnicity": "asian", "region": "urban"},
    {"claim": "Creatinine 3.5 indicates normal kidney function.", "y_true": 0, "y_pred": 0, "gender": "female", "age": 70, "ethnicity": "other", "region": "urban"},
    {"claim": "Fasting glucose of 400 is within range.", "y_true": 0, "y_pred": 0, "gender": "female", "age": 78, "ethnicity": "asian", "region": "rural"},
    {"claim": "High ferritin may indicate inflammation.", "y_true": 1, "y_pred": 1, "gender": "female", "age": 75, "ethnicity": "other", "region": "urban"},
    {"claim": "Low hemoglobin may suggest anemia.", "y_true": 1, "y_pred": 1, "gender": "male", "age": 40, "ethnicity": "asian", "region": "rural"},
    {"claim": "High creatinine may indicate kidney dysfunction.", "y_true": 1, "y_pred": 1, "gender": "female", "age": 45, "ethnicity": "asian", "region": "rural"},
    {"claim": "Sodium of 120 is within normal range.", "y_true": 0, "y_pred": 0, "gender": "male", "age": 50, "ethnicity": "other", "region": "rural"},
    {"claim": "TSH testing helps assess thyroid function.", "y_true": 1, "y_pred": 1, "gender": "female", "age": 35, "ethnicity": "asian", "region": "urban"},
    {"claim": "WBC of 18k suggests no infection.", "y_true": 0, "y_pred": 0, "gender": "male", "age": 62, "ethnicity": "other", "region": "urban"},
]


def _age_group(age: object) -> str:
    try:
        age_value = int(age)
    except (TypeError, ValueError):
        return "unknown"
    if age_value < 18:
        return "under_18"
    if age_value < 40:
        return "18_39"
    if age_value < 65:
        return "40_64"
    return "65_plus"


def _accuracy(records: Iterable[dict]) -> float:
    rows = list(records)
    if not rows:
        return 0.0
    correct = sum(1 for row in rows if row["y_true"] == row["y_pred"])
    return round(correct / len(rows), 4)


def _precision(records: Iterable[dict]) -> float:
    rows = list(records)
    tp = sum(1 for row in rows if row["y_true"] == 1 and row["y_pred"] == 1)
    fp = sum(1 for row in rows if row["y_true"] == 0 and row["y_pred"] == 1)
    return round(tp / (tp + fp), 4) if tp + fp else 0.0


def _recall(records: Iterable[dict]) -> float:
    rows = list(records)
    tp = sum(1 for row in rows if row["y_true"] == 1 and row["y_pred"] == 1)
    fn = sum(1 for row in rows if row["y_true"] == 1 and row["y_pred"] == 0)
    return round(tp / (tp + fn), 4) if tp + fn else 0.0


def _false_negative_rate(records: Iterable[dict]) -> float:
    rows = list(records)
    fn = sum(1 for row in rows if row["y_true"] == 1 and row["y_pred"] == 0)
    positives = sum(1 for row in rows if row["y_true"] == 1)
    return round(fn / positives, 4) if positives else 0.0


def _group_metrics(records: Iterable[dict]) -> dict:
    rows = list(records)
    return {
        "count": len(rows),
        "accuracy": _accuracy(rows),
        "precision": _precision(rows),
        "recall": _recall(rows),
        "false_negative_rate": _false_negative_rate(rows),
    }


def _flag_bias(metrics: dict, overall_accuracy: float, threshold: float = 0.10) -> dict:
    """Flag groups where accuracy deviates more than threshold from overall."""
    flags = {}
    for group, group_metrics in metrics.items():
        gap = abs(group_metrics["accuracy"] - overall_accuracy)
        if gap > threshold:
            flags[group] = {
                "accuracy": group_metrics["accuracy"],
                "gap_from_overall": round(gap, 4),
                "severity": "HIGH" if gap > 0.20 else "MODERATE",
            }
    return flags


def run_bias_audit(records: list[dict] | None = None) -> dict:
    """Run bias audit v1 and return a structured report."""
    rows = records or GROUND_TRUTH_RECORDS
    overall = _group_metrics(rows)
    verified_claims_summary = None
    try:
        verified_claims_summary = summarize_verified_claims(load_verified_claims())
    except (FileNotFoundError, ValueError):
        verified_claims_summary = {
            "total_records": 0,
            "note": "verified_claims_v1.csv not found or invalid",
        }

    by_gender = defaultdict(list)
    by_age = defaultdict(list)
    by_ethnicity = defaultdict(list)
    by_region = defaultdict(list)

    for row in rows:
        by_gender[str(row.get("gender", "unknown")).lower()].append(row)
        by_age[_age_group(row.get("age"))].append(row)
        by_ethnicity[str(row.get("ethnicity", "unknown")).lower()].append(row)
        by_region[str(row.get("region", "unknown")).lower()].append(row)

    gender_metrics = {group: _group_metrics(items) for group, items in by_gender.items()}
    age_group_metrics = {group: _group_metrics(items) for group, items in by_age.items()}
    ethnicity_metrics = {group: _group_metrics(items) for group, items in by_ethnicity.items()}
    region_metrics = {group: _group_metrics(items) for group, items in by_region.items()}

    overall_accuracy = overall["accuracy"]
    return {
        "audit_version": "v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_records": len(rows),
        "verified_claims_dataset": verified_claims_summary,
        "overall_metrics": overall,
        "gender_metrics": gender_metrics,
        "age_group_metrics": age_group_metrics,
        "ethnicity_metrics": ethnicity_metrics,
        "region_metrics": region_metrics,
        "bias_flags": {
            "gender": _flag_bias(gender_metrics, overall_accuracy),
            "age_group": _flag_bias(age_group_metrics, overall_accuracy),
            "ethnicity": _flag_bias(ethnicity_metrics, overall_accuracy),
            "region": _flag_bias(region_metrics, overall_accuracy),
        },
        "sign_off_required": True,
        "domain_lead_status": "PENDING",
        "notes": (
            "Bias audit v1 uses demographic synthetic ground truth for AC7 group slicing. "
            "The verified_claims_v1.csv dataset is included for claim verdict and "
            "hallucination regression checks."
        ),
    }


def _flat_csv_rows(report: dict) -> list[dict]:
    rows = []
    metric_sections = {
        "gender": "gender_metrics",
        "age_group": "age_group_metrics",
        "ethnicity": "ethnicity_metrics",
        "region": "region_metrics",
    }

    for group_type, metrics_key in metric_sections.items():
        flags = report["bias_flags"].get(group_type, {})
        for group_name, metrics in report[metrics_key].items():
            flag = flags.get(group_name, {})
            rows.append(
                {
                    "group_type": group_type,
                    "group_name": group_name,
                    "count": metrics["count"],
                    "accuracy": metrics["accuracy"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "false_negative_rate": metrics["false_negative_rate"],
                    "bias_flagged": bool(flag),
                    "bias_severity": flag.get("severity", ""),
                    "gap_from_overall": flag.get("gap_from_overall", 0),
                }
            )
    return rows


def save_report(report: dict) -> tuple[Path, Path]:
    """Save report as JSON and CSV."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    rows = _flat_csv_rows(report)
    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return JSON_PATH, CSV_PATH


def print_summary(report: dict) -> None:
    overall = report["overall_metrics"]
    print("\n" + "=" * 55)
    print("VAIDYAAI BIAS AUDIT v1 REPORT")
    print("=" * 55)
    print(f"Records  : {report['total_records']}")
    print(
        "Overall  : "
        f"accuracy={overall['accuracy']:.1%}  "
        f"precision={overall['precision']:.1%}  "
        f"recall={overall['recall']:.1%}"
    )

    sections = [
        ("Gender", "gender_metrics", "gender"),
        ("Age Group", "age_group_metrics", "age_group"),
        ("Ethnicity", "ethnicity_metrics", "ethnicity"),
        ("Region", "region_metrics", "region"),
    ]
    for title, metrics_key, flags_key in sections:
        print(f"\n-- {title} --")
        flags = report["bias_flags"][flags_key]
        for group, metrics in report[metrics_key].items():
            flag = "FLAGGED" if group in flags else "OK"
            print(
                f"  {group:<10} "
                f"acc={metrics['accuracy']:.1%}  "
                f"fnr={metrics['false_negative_rate']:.1%}  "
                f"n={metrics['count']}  {flag}"
            )

    total_flags = sum(len(flags) for flags in report["bias_flags"].values())
    print(f"\nTotal bias flags : {total_flags}")
    print(f"Status           : {report['domain_lead_status']}")
    print("=" * 55)


def main() -> None:
    print("Running VaidyaAI Bias Audit v1...")
    report = run_bias_audit()
    print_summary(report)
    json_path, csv_path = save_report(report)
    print(f"\nJSON report: {json_path}")
    print(f"CSV report : {csv_path}")
    print("\nDomain lead: review flagged groups and update domain_lead_status when approved.")


if __name__ == "__main__":
    main()
