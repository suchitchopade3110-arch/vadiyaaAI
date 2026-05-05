"""Bias audit helpers for demographic slices."""

from collections import defaultdict
from typing import Iterable


def _age_group(age) -> str:
    try:
        age = int(age)
    except (TypeError, ValueError):
        return "unknown"
    if age < 18:
        return "under_18"
    if age < 40:
        return "18_39"
    if age < 65:
        return "40_64"
    return "65_plus"


def _accuracy(rows: Iterable[dict]) -> float:
    rows = list(rows)
    if not rows:
        return 0.0
    correct = sum(1 for row in rows if row.get("y_true") == row.get("y_pred"))
    return round(correct / len(rows), 4)


def audit_demographic_accuracy(records: list[dict]) -> dict:
    """Compute gender and age-group accuracy from rows with y_true/y_pred."""
    by_gender = defaultdict(list)
    by_age = defaultdict(list)
    for row in records:
        by_gender[str(row.get("gender", "unknown")).lower()].append(row)
        by_age[_age_group(row.get("age"))].append(row)
    return {
        "overall_accuracy": _accuracy(records),
        "gender_accuracy": {group: _accuracy(rows) for group, rows in by_gender.items()},
        "age_group_accuracy": {group: _accuracy(rows) for group, rows in by_age.items()},
    }


gender_age_accuracy = audit_demographic_accuracy

__all__ = ["audit_demographic_accuracy", "gender_age_accuracy"]

