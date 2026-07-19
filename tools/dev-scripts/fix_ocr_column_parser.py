"""Compatibility shim for Colab-style imports.

The implementation lives in app.services.lab_report_column_parser.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.services.lab_report_column_parser import *  # noqa: F401,F403,E402
