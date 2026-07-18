from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from microscopy_cv_research.training.external_commons import run_external_commons_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare optional CoMMonS microscopic material surface benchmark.")
    parser.add_argument("--config", default="configs/external_commons_microscopy.json")
    parser.add_argument("--allow-large-download", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    print(json.dumps(run_external_commons_experiment(config_path, allow_large_download=args.allow_large_download), indent=2))


if __name__ == "__main__":
    main()

