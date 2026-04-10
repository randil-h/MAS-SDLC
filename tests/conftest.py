"""
pytest configuration for the MAS SDLC test harness.

Inserts the package root (mas_sdlc/) onto sys.path so that all test
modules can import agents, tools, and state without needing package
installation or manual sys.path manipulation in every file.
"""

import sys
from pathlib import Path

# Resolve: tests/ → parent → mas_sdlc/
_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))
