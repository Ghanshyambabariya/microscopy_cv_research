from __future__ import annotations

import argparse
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from microscopy_cv_research.config import load_config
from microscopy_cv_research.data.segmentation import SemSegmentationDataset, load_nasa_ebc_samples, load_sem_dataset_from_registry, SegmentationSample
from microscopy_cv_research.models.segmentation import create_segmentation_model
from microscopy_cv_research.training.engine import get_device, save_json
from microscopy_cv_research.training.segmentation import run_segmentation_epoch, compute_class_weights, create_prediction_figure


def split_samples(samples: list[SegmentationSample], seed: int = 42) -> tuple[list[SegmentationSample], list[SegmentationSample], list[SegmentationSample]]:
    rng = random.Random(seed)
    indices = list(range(len(samples)))
    rng.shuffle(indices)
    n = len(indices)
    n_train = max(1, int(0.7 * n))
    n_val = max(1, int(0.15 * n))
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    if not test_idx:
        test_idx = val_idx
    return ([samples[i] for i in train_idx], [samples[i] for i in val_idx], [samples[i] for i in test_idx])


def apply_mask_rules(mask: np.ndarray, mask_map: dict[int, int] | None, threshold: int | None) -> np.ndarray:
    if threshold is not None:
        mask = (mask >= threshold).astype(np.int64)
    if mask_map:
        mapped = np.copy(mask)
        for src, dst in mask_map.items():
            mapped[mask == src] = dst
        mask = mapped
    return mask


def infer_num_classes_full(samples: list[SegmentationSample], mask_map: dict[int, int] | None, threshold: int | None) -> int:
    max_label = 0
    for sample in samples:
        mask = np.array(Image.open(sample.mask_path).convert("L"), dtype=np.int64)
        mask = apply_mask_rules(mask, mask_map, threshold)
        max_label = max(max_label, int(mask.max()))
    return max_label + 1


def limit_samples(samples: list[SegmentationSample], max_count: int | None) -> list[SegmentationSample]:
    if not max_count or max_count <= 0:
        return samples
    return samples[: max_count]


def make_loader(samples: list[SegmentationSample], image_size: int, batch_size: int, shuffle: bool, mask_map: dict[int, int] | None, threshold: int | None, ignore_index: int | None) -> DataLoader:
    ds = SemSegmentationDataset(samples, image_size=image_size, mask_map=mask_map, threshold=threshold, ignore_index=ignore_index)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_one_dataset(cfg: dict, registry: dict, dataset_key: str) -> dict[str, Any]:
    project_root = Path(cfg["project_root"])
    entry = registry[dataset_key]
    dataset_type = entry.get("type", "mask_pairs")
    mask_map = entry.get("mask_map")
    if isinstance(mask_map, dict):
        mask_map = {int(k): int(v) for k, v in mask_map.items()}
    threshold = entry.get("threshold")
    ignore_index = entry.get("ignore_index")

    if dataset_type == "nasa_ebc":
        root = Path(entry["root"])
        train = load_nasa_ebc_samples(root, entry["datasets"], "train")
        val = load_nasa_ebc_samples(root, entry["datasets"], "val")
        test = load_nasa_ebc_samples(root, entry["datasets"], "test")
    elif dataset_type == "pascal":
        train = load_sem_dataset_from_registry(registry, dataset_key, "train")
        val = load_sem_dataset_from_registry(registry, dataset_key, "val")
        test = load_sem_dataset_from_registry(registry, dataset_key, "test")
    elif dataset_type == "emps":
        train = load_sem_dataset_from_registry(registry, dataset_key, "train")
        test = load_sem_dataset_from_registry(registry, dataset_key, "test")
        if train:
            train, val, _ = split_samples(train)
        else:
            val = []
        if not test:
            test = val
    else:
        all_samples = load_sem_dataset_from_registry(registry, dataset_key, "all")
        if not all_samples:
            return {"dataset": dataset_key, "skipped": True, "reason": "no samples found"}
        train, val, test = split_samples(all_samples)

    if not train or not val or not test:
        return {"dataset": dataset_key, "skipped": True, "reason": "insufficient split"}

    num_classes = infer_num_classes_full(train + val + test, mask_map, threshold)

    max_samples = cfg.get("max_samples")
    train = limit_samples(train, max_samples)
    val = limit_samples(val, max_samples)
    test = limit_samples(test, max_samples)
    device = get_device()
    model = create_segmentation_model(cfg["model_name"], num_classes=num_classes, base_channels=cfg["base_channels"], dropout=cfg["dropout"]).to(device)

    train_loader = make_loader(train, cfg["image_size"], cfg["batch_size"], shuffle=True, mask_map=mask_map, threshold=threshold, ignore_index=ignore_index)
    val_loader = make_loader(val, cfg["image_size"], cfg["batch_size"], shuffle=False, mask_map=mask_map, threshold=threshold, ignore_index=ignore_index)
    test_loader = make_loader(test, cfg["image_size"], cfg["batch_size"], shuffle=False, mask_map=mask_map, threshold=threshold, ignore_index=ignore_index)

    class_weights = compute_class_weights(train_loader.dataset, num_classes, ignore_index=ignore_index).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index if ignore_index is not None else -100)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])

    for _ in range(cfg["epochs"]):
        run_segmentation_epoch(model, train_loader, criterion, optimizer, device, num_classes=num_classes, ignore_index=ignore_index)
    _, val_metrics = run_segmentation_epoch(model, val_loader, criterion, None, device, num_classes=num_classes, ignore_index=ignore_index)
    _, test_metrics = run_segmentation_epoch(model, test_loader, criterion, None, device, num_classes=num_classes, ignore_index=ignore_index)
    figure_path = project_root / "reports" / "figures" / f"sem_suite_{dataset_key}.png"
    display_name = entry.get("name", dataset_key)
    figure_metadata = create_prediction_figure(model, test_loader, device, figure_path, title=f"{display_name} SEM segmentation predictions")

    result = {
        "dataset": dataset_key,
        "display_name": display_name,
        "model_name": cfg["model_name"],
        "config_path": cfg.get("config_path"),
        "num_classes": num_classes,
        "train_samples": len(train),
        "val_samples": len(val),
        "test_samples": len(test),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "prediction_figure": str(figure_path),
        "figure_metadata": figure_metadata,
    }
    out_path = project_root / "reports" / f"sem_suite_{dataset_key}.json"
    save_json(result, out_path)
    return result


def write_leaderboard(results: list[dict[str, Any]], output_path: Path) -> None:
    lines = ["# SEM Benchmark Leaderboard", "", "| Dataset | Model | Pixel Acc | Mean IoU (fg) | Mean Dice (fg) | Notes |", "|---|---|---|---|---|---|"]
    for result in results:
        if result.get("skipped"):
            lines.append(f"| {result['dataset']} | - | - | - | - | skipped: {result.get('reason','')} |")
            continue
        test = result["test_metrics"]
        fig = Path(result.get('prediction_figure', ''))
        fig_note = f"; fig {fig.name}" if fig.name else ''
        lines.append(
            f"| {result['display_name']} | {result['model_name']} | {test['pixel_accuracy']:.4f} | {test['mean_iou_fg']:.4f} | {test['mean_dice_fg']:.4f} | train {result['train_samples']}{fig_note} |"
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Run the multi-dataset SEM segmentation benchmark suite.")
    parser.add_argument("--config", default="configs/sem_suite.json", help="Path to a SEM suite config JSON.")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    cfg = load_config(config_path)
    cfg["project_root"] = str(root)
    cfg["config_path"] = str(config_path)
    set_seed(int(cfg.get("seed", 42)))
    registry = load_config(root / "configs" / "sem_dataset_registry.json")
    for entry in registry.values():
        entry_root = Path(entry["root"])
        if not entry_root.is_absolute():
            entry["root"] = str(root / entry_root)

    results: list[dict[str, Any]] = []
    for dataset_key in registry:
        results.append(run_one_dataset(cfg, registry, dataset_key))
    write_leaderboard(results, root / "reports" / "sem_leaderboard.md")


if __name__ == "__main__":
    main()
