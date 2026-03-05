"""Small generic helpers used by examples."""

from __future__ import annotations

import os


def getenv_int(key: str, default: int) -> int:
    """Read an environment variable as int with a fallback value."""
    return int(os.getenv(key, str(default)))
