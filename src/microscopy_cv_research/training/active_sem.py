from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from microscopy_cv_research.data.segmentation import SemSegmentationDataset, load_nasa_ebc_samples, split_labeled_unlabeled, SegmentationSample
from microscopy_cv_research.evaluation.metrics import segmentation_metrics
from microscopy_cv_research.models.segmentation import UNetSmall
from microscopy_cv_research.training.engine import get_device, save_checkpoint, save_json
from microscopy_cv_research.training.segmentation import run_segmentation_epoch, compute_class_weights


@dataclass(slots=True)
class ActiveConfig:
    project_root: Path
    benchmark_root: Path
    datasets: list[str]
    num_classes: int = 3
    image_size: int = 256
    batch_size: int = 4
    seed_size: int = 6
    acquisition_size: int = 4
    rounds: int = 2
    epochs_per_round: int = 4
    learning_rate: float = 1e-3
    base_channels: int = 32
    results_path: Path | None = None
    figure_path: Path | None = None


def make_dataloaders(samples: list[SegmentationSample], image_size: int, batch_size: int) -> DataLoader:
    ds = SemSegmentationDataset(samples, image_size=image_size)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)


def uncertainty_scores(model: torch.nn.Module, dataloader: DataLoader, device: torch.device, num_classes: int) -> list[tuple[float, str, SegmentationSample]]:
    model.eval()
    scores: list[tuple[float, str, SegmentationSample]] = []
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            paths = batch["image_path"]
            masks = batch["mask"]  # keep for sample mapping only
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean(dim=(1, 2))
            for ent, path, mask_arr in zip(entropy.cpu().numpy(), paths, masks):
                sample = SegmentationSample(Path(path), Path(path.replace('test','test_annot')) if False else Path(path), "", "")
                scores.append((float(ent), path, sample))
    scores.sort(key=lambda x: x[0], reverse=True)
    return scores


def run_active_learning(cfg: ActiveConfig) -> dict[str, Any]:
    rng = random.Random(42)
    train_samples = load_nasa_ebc_samples(cfg.benchmark_root, cfg.datasets, "train")
    val_samples = load_nasa_ebc_samples(cfg.benchmark_root, cfg.datasets, "val")
    test_samples = load_nasa_ebc_samples(cfg.benchmark_root, cfg.datasets, "test")

    labeled, unlabeled = split_labeled_unlabeled(train_samples, cfg.seed_size, rng)

    device = get_device()
    history: list[dict[str, Any]] = []
    model = UNetSmall(num_classes=cfg.num_classes, base_channels=cfg.base_channels).to(device)

    for round_idx in range(cfg.rounds):
        train_loader = make_dataloaders(labeled, cfg.image_size, cfg.batch_size)
        val_loader = make_dataloaders(val_samples, cfg.image_size, cfg.batch_size)
        test_loader = make_dataloaders(test_samples, cfg.image_size, cfg.batch_size)

        class_weights = compute_class_weights(train_loader.dataset, cfg.num_classes).to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

        for _ in range(cfg.epochs_per_round):
            run_segmentation_epoch(model, train_loader, criterion, optimizer, device)
        _, val_metrics = run_segmentation_epoch(model, val_loader, criterion, None, device)
        _, test_metrics = run_segmentation_epoch(model, test_loader, criterion, None, device)

        history.append({
            "round": round_idx + 1,
            "labeled_size": len(labeled),
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
        })

        if not unlabeled:
            break

        # Select new samples by entropy on unlabeled set
        unl_loader = make_dataloaders(unlabeled, cfg.image_size, cfg.batch_size)
        scores = []
        model.eval()
        with torch.no_grad():
            for batch in unl_loader:
                images = batch["image"].to(device)
                logits = model(images)
                probs = torch.softmax(logits, dim=1)
                entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean(dim=(1, 2))
                for ent, sample in zip(entropy.cpu().numpy(), batch["image_path"]):
                    scores.append((float(ent), sample))
        scores.sort(key=lambda x: x[0], reverse=True)
        acquire = scores[: min(cfg.acquisition_size, len(scores))]
        acquired_paths = set(path for _, path in acquire)
        newly_labeled = [s for s in unlabeled if str(s.image_path) in acquired_paths]
        labeled.extend(newly_labeled)
        unlabeled = [s for s in unlabeled if str(s.image_path) not in acquired_paths]

    results = {
        "benchmark": "NASA EBC SEM active learning",
        "rounds": cfg.rounds,
        "seed_size": cfg.seed_size,
        "acquisition_size": cfg.acquisition_size,
        "history": history,
    }
    out_path = cfg.results_path or (cfg.project_root / "reports/sem_active_learning_log.json")
    save_json(results, out_path)
    return results
