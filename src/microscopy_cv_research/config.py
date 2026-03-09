from __future__ import annotations

from pathlib import Path
import json


def load_config(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
