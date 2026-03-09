from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from microscopy_cv_research.models.encoder_registry import get_encoder_spec


def test_encoder_registry_contains_microscopy_model() -> None:
    spec = get_encoder_spec("micronet")
    assert spec.family == "microscopy-specialized"
