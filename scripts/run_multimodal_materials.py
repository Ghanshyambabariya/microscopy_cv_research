from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from microscopy_cv_research.training.multimodal_materials import run_multimodal_materials_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multimodal microscopy + process-signal materials ML.")
    parser.add_argument("--config", default="configs/multimodal_materials.json")
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    report = run_multimodal_materials_experiment(config_path)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

