from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from microscopy_cv_research.config import load_config
from microscopy_cv_research.data.segmentation import SemSegmentationDataset, load_nasa_ebc_samples, split_labeled_unlabeled, SegmentationSample, load_sem_dataset_from_registry
from microscopy_cv_research.models.segmentation import create_segmentation_model
from microscopy_cv_research.training.engine import get_device, save_json
from microscopy_cv_research.training.segmentation import run_segmentation_epoch, compute_class_weights


@dataclass(slots=True)
class ActiveConfig:
    project_root: Path
    benchmark_root: Path | None
    datasets: list[str] | None
    num_classes: int = 3
    image_size: int = 256
    batch_size: int = 4
    seed_size: int = 6
    acquisition_size: int = 4
    rounds: int = 2
    epochs_per_round: int = 4
    learning_rate: float = 1e-3
    base_channels: int = 32
    model_name: str = "unet_small"
    dropout: float = 0.1
    mc_samples: int = 5
    seed: int = 42
    mask_map: dict[int, int] | None = None
    threshold: int | None = None
    ignore_index: int | None = None
    registry_path: Path | None = None
    dataset_key: str | None = None
    results_path: Path | None = None


def make_dataloader(
    samples: list[SegmentationSample],
    image_size: int,
    batch_size: int,
    shuffle: bool,
    mask_map: dict[int, int] | None = None,
    threshold: int | None = None,
    ignore_index: int | None = None,
) -> DataLoader:
    ds = SemSegmentationDataset(samples, image_size=image_size, mask_map=mask_map, threshold=threshold, ignore_index=ignore_index)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def predictive_entropy(model: torch.nn.Module, images: torch.Tensor, mc_samples: int) -> torch.Tensor:
    if mc_samples <= 1:
        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        return -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean(dim=(1, 2))

    probs_sum = None
    for _ in range(mc_samples):
        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        probs_sum = probs if probs_sum is None else probs_sum + probs
    mean_probs = probs_sum / float(mc_samples)
    return -(mean_probs * torch.log(mean_probs + 1e-8)).sum(dim=1).mean(dim=(1, 2))


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


def load_samples(cfg: ActiveConfig) -> tuple[list[SegmentationSample], list[SegmentationSample], list[SegmentationSample]]:
    if cfg.registry_path and cfg.dataset_key:
        registry = load_config(cfg.registry_path)
        entry = registry[cfg.dataset_key]
        mask_map = entry.get("mask_map")
        cfg.mask_map = {int(k): int(v) for k, v in mask_map.items()} if isinstance(mask_map, dict) else None
        cfg.threshold = entry.get("threshold")
        cfg.ignore_index = entry.get("ignore_index")
        if entry.get("type") == "nasa_ebc":
            root = Path(entry["root"])
            train = load_nasa_ebc_samples(root, entry["datasets"], "train")
            val = load_nasa_ebc_samples(root, entry["datasets"], "val")
            test = load_nasa_ebc_samples(root, entry["datasets"], "test")
            return train, val, test
        all_samples = load_sem_dataset_from_registry(registry, cfg.dataset_key, "all")
        return split_samples(all_samples)

    train = load_nasa_ebc_samples(Path(cfg.benchmark_root), cfg.datasets or [], "train")
    val = load_nasa_ebc_samples(Path(cfg.benchmark_root), cfg.datasets or [], "val")
    test = load_nasa_ebc_samples(Path(cfg.benchmark_root), cfg.datasets or [], "test")
    return train, val, test


def run_active_learning(cfg: ActiveConfig) -> dict[str, Any]:
    rng = random.Random(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    train_samples, val_samples, test_samples = load_samples(cfg)

    labeled, unlabeled = split_labeled_unlabeled(train_samples, cfg.seed_size, rng)

    device = get_device()
    history: list[dict[str, Any]] = []
    model = create_segmentation_model(cfg.model_name, num_classes=cfg.num_classes, base_channels=cfg.base_channels, dropout=cfg.dropout).to(device)

    for round_idx in range(cfg.rounds):
        train_loader = make_dataloader(labeled, cfg.image_size, cfg.batch_size, shuffle=True, mask_map=cfg.mask_map, threshold=cfg.threshold, ignore_index=cfg.ignore_index)
        val_loader = make_dataloader(val_samples, cfg.image_size, cfg.batch_size, shuffle=False, mask_map=cfg.mask_map, threshold=cfg.threshold, ignore_index=cfg.ignore_index)
        test_loader = make_dataloader(test_samples, cfg.image_size, cfg.batch_size, shuffle=False, mask_map=cfg.mask_map, threshold=cfg.threshold, ignore_index=cfg.ignore_index)

        class_weights = compute_class_weights(train_loader.dataset, cfg.num_classes, ignore_index=cfg.ignore_index).to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=cfg.ignore_index if cfg.ignore_index is not None else -100)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

        for _ in range(cfg.epochs_per_round):
            run_segmentation_epoch(model, train_loader, criterion, optimizer, device, num_classes=cfg.num_classes, ignore_index=cfg.ignore_index)
        _, val_metrics = run_segmentation_epoch(model, val_loader, criterion, None, device, num_classes=cfg.num_classes, ignore_index=cfg.ignore_index)
        _, test_metrics = run_segmentation_epoch(model, test_loader, criterion, None, device, num_classes=cfg.num_classes, ignore_index=cfg.ignore_index)

        history.append({
            "round": round_idx + 1,
            "labeled_size": len(labeled),
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
        })

        if not unlabeled:
            break

        unl_loader = make_dataloader(unlabeled, cfg.image_size, cfg.batch_size, shuffle=False, mask_map=cfg.mask_map, threshold=cfg.threshold, ignore_index=cfg.ignore_index)
        scores: list[tuple[float, str]] = []
        model.train()
        with torch.no_grad():
            for batch in unl_loader:
                images = batch["image"].to(device)
                entropy = predictive_entropy(model, images, cfg.mc_samples)
                for ent, path in zip(entropy.cpu().numpy(), batch["image_path"]):
                    scores.append((float(ent), path))
        scores.sort(key=lambda x: x[0], reverse=True)
        acquire = scores[: min(cfg.acquisition_size, len(scores))]
        acquired_paths = set(path for _, path in acquire)
        newly_labeled = [s for s in unlabeled if str(s.image_path) in acquired_paths]
        labeled.extend(newly_labeled)
        unlabeled = [s for s in unlabeled if str(s.image_path) not in acquired_paths]

    results = {
        "benchmark": "SEM active learning",
        "rounds": cfg.rounds,
        "seed_size": cfg.seed_size,
        "acquisition_size": cfg.acquisition_size,
        "mc_samples": cfg.mc_samples,
        "seed": cfg.seed,
        "model_name": cfg.model_name,
        "dataset_key": cfg.dataset_key or "nasa_ebc",
        "mask_map": cfg.mask_map,
        "threshold": cfg.threshold,
        "ignore_index": cfg.ignore_index,
        "history": history,
    }
    out_path = cfg.results_path or (cfg.project_root / "reports/sem_active_learning_log.json")
    save_json(results, out_path)
    return results
