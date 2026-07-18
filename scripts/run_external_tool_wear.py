from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from microscopy_cv_research.training.external_tool_wear import run_external_tool_wear_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Download, clean, preprocess, and model an external tool-wear dataset.")
    parser.add_argument("--config", default="configs/external_tool_wear_vicomtech.json")
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    report = run_external_tool_wear_experiment(config_path)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

