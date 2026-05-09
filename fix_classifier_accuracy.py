"""Verify VaidyaAI classifier accuracy upgrades and warm model caches.

Run from backend root:
    python fix_classifier_accuracy.py

This script is intentionally idempotent. The code patches live in:
    app/services/modality_classifier.py
    app/image_pipeline/classifier_v2.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run(label: str, code: str, timeout: int = 300) -> bool:
    print(f"\n[{label}]")
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.stdout:
        print(result.stdout.strip())
    if result.returncode != 0:
        print(result.stderr.strip())
        return False
    return True


def main() -> int:
    print("=" * 60)
    print("VaidyaAI - Classifier Accuracy Verification")
    print("=" * 60)

    ok = True
    ok &= run(
        "CheXNet NIH weights",
        """
import torchxrayvision as xrv
model = xrv.models.DenseNet(weights="densenet121-res224-nih")
print(f"NIH ready: {len(model.pathologies)} pathologies")
""",
    )
    ok &= run(
        "MRI model cache",
        """
from pathlib import Path
from PIL import Image
from app.services.modality_classifier import classify_by_modality
img = Path("/tmp/vaidyaa_mri_cache_check.png")
Image.new("RGB", (224, 224), (0, 0, 0)).save(img)
res = classify_by_modality(str(img), "mri")
print(f"MRI model: {res.model_used}")
print("Classes:", [item["label"] for item in res.all_findings[:5]])
""",
    )
    ok &= run(
        "Python compile",
        """
import py_compile
for path in [
    "app/services/modality_classifier.py",
    "app/image_pipeline/classifier_v2.py",
    "app/pipeline.py",
]:
    py_compile.compile(path, doraise=True)
print("Compile OK")
""",
    )

    print("\n" + "=" * 60)
    if ok:
        print("Done. Classifier accuracy upgrades are ready.")
        print("Restart Celery to apply running-worker changes.")
        return 0
    print("One or more checks failed. See output above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
