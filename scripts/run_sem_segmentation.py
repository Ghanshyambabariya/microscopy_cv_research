from __future__ import annotations

from pathlib import Path
import json
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from microscopy_cv_research.config import load_config
from microscopy_cv_research.training.segmentation import train_sem_segmentation


def main() -> None:
    config_path = REPO_ROOT / "configs" / "sem_segmentation_nasa_ebc.json"
    config = load_config(config_path)
    config["project_root"] = str(REPO_ROOT)
    results = train_sem_segmentation(config)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
