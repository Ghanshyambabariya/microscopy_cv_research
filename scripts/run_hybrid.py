from __future__ import annotations

from pathlib import Path
import json
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from microscopy_cv_research.training.hybrid import run_hybrid_experiment


def main() -> None:
    report = run_hybrid_experiment(REPO_ROOT / "configs" / "hybrid_learning.json", REPO_ROOT / "configs" / "synthetic_generation.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
