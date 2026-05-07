"""Ground-truth claim dataset utilities for Phase 2 audit/regression checks."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Iterable


DATA_DIR = Path(__file__).resolve().parent / "data"
VERIFIED_CLAIMS_PATH = DATA_DIR / "verified_claims_v1.csv"
REQUIRED_COLUMNS = {
    "claim_id",
    "claim_text",
    "verdict",
    "confidence",
    "category",
    "source",
    "icd10",
    "severity",
}


def load_verified_claims(path: Path | str = VERIFIED_CLAIMS_PATH) -> list[dict]:
    """Load verified/refuted claim ground truth from CSV."""
    csv_path = Path(path)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - fieldnames
        if missing:
            raise ValueError(f"{csv_path} missing columns: {sorted(missing)}")

        rows = []
        for row in reader:
            row["confidence"] = float(row["confidence"])
            row["verdict"] = row["verdict"].strip().lower()
            row["category"] = row["category"].strip().lower()
            row["severity"] = row["severity"].strip().lower()
            rows.append(row)
    return rows


def verdict_to_label(verdict: str) -> int:
    """Map claim verdict to binary label: verified=1, refuted=0."""
    normalized = str(verdict).strip().lower()
    if normalized == "verified":
        return 1
    if normalized == "refuted":
        return 0
    raise ValueError(f"Unsupported verdict: {verdict!r}")


def summarize_verified_claims(records: Iterable[dict]) -> dict:
    """Return compact dataset metadata for audit reports."""
    rows = list(records)
    verdicts = Counter(row["verdict"] for row in rows)
    categories = Counter(row["category"] for row in rows)
    severities = Counter(row["severity"] for row in rows)
    return {
        "path": str(VERIFIED_CLAIMS_PATH),
        "total_records": len(rows),
        "verified_count": verdicts.get("verified", 0),
        "refuted_count": verdicts.get("refuted", 0),
        "category_count": len(categories),
        "categories": dict(sorted(categories.items())),
        "severities": dict(sorted(severities.items())),
    }


__all__ = [
    "VERIFIED_CLAIMS_PATH",
    "load_verified_claims",
    "summarize_verified_claims",
    "verdict_to_label",
]
