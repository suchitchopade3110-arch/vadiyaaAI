"""Compatibility shim for Colab-style imports.

The implementation lives in app.services.rag_panel_filter.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.services.rag_panel_filter import *  # noqa: F401,F403,E402
