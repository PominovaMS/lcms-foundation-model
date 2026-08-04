"""Pytest path setup.

Puts the repo root on ``sys.path`` (for ``source.*``) and ``eval/`` on it too, so
tests can use the same flat imports the eval scripts use (``from data import
assign_splits``) — ``eval/`` is a script directory, not a package, and
``eval/probe_checkpoint.py`` reaches its siblings the same way.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for path in (ROOT, ROOT / "eval"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
