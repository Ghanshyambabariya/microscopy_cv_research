from __future__ import annotations

from pathlib import Path
import json
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from microscopy_cv_research.config import load_config
from microscopy_cv_research.training.active_sem import ActiveConfig, run_active_learning


def main() -> None:
    cfg_path = REPO_ROOT / "configs" / "active_sem_ebc.json"
    raw = load_config(cfg_path)
    cfg = ActiveConfig(
        project_root=REPO_ROOT,
        benchmark_root=REPO_ROOT / raw["benchmark_root"],
        datasets=raw["datasets"],
        num_classes=raw.get("num_classes", 3),
        image_size=raw.get("image_size", 256),
        batch_size=raw.get("batch_size", 4),
        seed_size=raw.get("seed_size", 6),
        acquisition_size=raw.get("acquisition_size", 4),
        rounds=raw.get("rounds", 2),
        epochs_per_round=raw.get("epochs_per_round", 4),
        learning_rate=raw.get("learning_rate", 1e-3),
        base_channels=raw.get("base_channels", 32),
        results_path=REPO_ROOT / raw.get("results_path", "reports/sem_active_learning_log.json"),
    )
    results = run_active_learning(cfg)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
