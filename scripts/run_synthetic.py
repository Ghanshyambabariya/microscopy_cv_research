from __future__ import annotations

from pathlib import Path
import json
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from microscopy_cv_research.training.synthetic import generate_synthetic_images


def main() -> None:
    report = generate_synthetic_images(REPO_ROOT / "configs" / "synthetic_generation.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
