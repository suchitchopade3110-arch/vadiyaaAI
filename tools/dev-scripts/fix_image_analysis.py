"""Compatibility shim for Colab-style imports."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.services.fix_image_analysis import *  # noqa: F401,F403,E402
