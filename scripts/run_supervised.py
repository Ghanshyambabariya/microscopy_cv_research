from __future__ import annotations

import argparse
from pathlib import Path
import json
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from microscopy_cv_research.training.supervised import plan_supervised_experiment, run_supervised_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/supervised_classification.json")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    config_path = REPO_ROOT / args.config
    report = plan_supervised_experiment(config_path) if args.plan_only else run_supervised_experiment(config_path)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
