from __future__ import annotations

import os


def getenv_int(key: str, default: int) -> int:
    return int(os.getenv(key, str(default)))
