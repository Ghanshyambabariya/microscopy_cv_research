from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader

from microscopy_cv_research.evaluation.metrics import segmentation_metrics
from microscopy_cv_research.training.engine import get_device, save_checkpoint, save_json
from microscopy_cv_research.models.segmentation import create_segmentation_model


def run_segmentation_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    num_classes: int | None = None,
    ignore_index: int | None = None,
) -> tuple[float, dict[str, float]]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []

    for batch in dataloader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device=device, dtype=torch.long)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, masks)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += float(loss.item()) * images.size(0)
        preds.append(logits.argmax(dim=1).detach().cpu().numpy())
        targets.append(masks.detach().cpu().numpy())

    y_pred = np.concatenate(preds, axis=0)
    y_true = np.concatenate(targets, axis=0)
    if num_classes is not None:
        metric_classes = num_classes
    elif ignore_index is not None and np.any(y_true != ignore_index):
        metric_classes = int(y_true[y_true != ignore_index].max()) + 1
    else:
        metric_classes = int(y_true.max()) + 1
    metrics = segmentation_metrics(y_true, y_pred, num_classes=metric_classes, ignore_index=ignore_index)
    return total_loss / max(len(dataloader.dataset), 1), metrics


def create_prediction_figure(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    output_path: str | Path,
    num_examples: int = 3,
    title: str = "SEM segmentation predictions",
    ignore_index: int | None = None,
) -> list[dict[str, Any]]:
    model.eval()
    batch = next(iter(dataloader))
    images = batch["image"].to(device)
    masks = batch["mask"].numpy()
    image_paths = batch["image_path"]

    with torch.no_grad():
        predictions = model(images).argmax(dim=1).cpu().numpy()

    examples = min(num_examples, images.size(0))
    fig, axes = plt.subplots(examples, 5, figsize=(15, 3 * examples))
    if examples == 1:
        axes = np.expand_dims(axes, axis=0)

    metadata: list[dict[str, Any]] = []
    for row in range(examples):
        image = images[row].cpu().permute(1, 2, 0).numpy()
        image = np.clip(image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]), 0.0, 1.0)
        vmax = int(max(masks[row].max(), predictions[row].max()))
        valid = masks[row] != ignore_index if ignore_index is not None else np.ones_like(masks[row], dtype=bool)
        foreground = predictions[row] > 0
        overlay = image.copy()
        overlay[foreground] = 0.55 * overlay[foreground] + 0.45 * np.array([1.0, 0.15, 0.05])
        error_map = np.logical_and(valid, masks[row] != predictions[row])

        axes[row, 0].imshow(image)
        axes[row, 0].set_title("Input")
        axes[row, 1].imshow(masks[row], cmap="viridis", vmin=0, vmax=vmax)
        axes[row, 1].set_title("Ground Truth")
        axes[row, 2].imshow(predictions[row], cmap="viridis", vmin=0, vmax=vmax)
        axes[row, 2].set_title("Prediction")
        axes[row, 3].imshow(overlay)
        axes[row, 3].set_title("Prediction Overlay")
        axes[row, 4].imshow(error_map, cmap="magma", vmin=0, vmax=1)
        axes[row, 4].set_title("Error Map")
        for col in range(5):
            axes[row, col].axis("off")

        metadata.append({
            "image_path": image_paths[row],
            "true_labels": sorted(np.unique(masks[row]).tolist()),
            "pred_labels": sorted(np.unique(predictions[row]).tolist()),
            "error_rate": float(error_map.sum() / max(valid.sum(), 1)),
        })

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return metadata


def compute_class_weights(dataset: Any, num_classes: int, ignore_index: int | None = None) -> torch.Tensor:
    counts = np.zeros(num_classes, dtype=np.float64)
    for idx in range(len(dataset)):
        mask = dataset[idx]["mask"].numpy()
        if ignore_index is not None:
            mask = mask[mask != ignore_index]
        if mask.size == 0:
            continue
        bincount = np.bincount(mask.reshape(-1), minlength=num_classes)
        if bincount.size > counts.size:
            new_counts = np.zeros(bincount.size, dtype=np.float64)
            new_counts[: counts.size] = counts
            counts = new_counts
        counts[: bincount.size] += bincount
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (counts * counts.size)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def train_sem_segmentation(config: dict) -> dict[str, Any]:
    from microscopy_cv_research.data.segmentation import SemSegmentationDataset, load_nasa_ebc_samples

    project_root = Path(config.get("project_root", Path(__file__).resolve().parents[3]))
    benchmark_root = project_root / config["benchmark_root"]
    datasets = config["datasets"]
    image_size = int(config.get("image_size", 256))
    batch_size = int(config.get("batch_size", 4))
    epochs = int(config.get("epochs", 10))
    learning_rate = float(config.get("learning_rate", 1e-3))
    num_classes = int(config.get("num_classes", 3))
    dropout = float(config.get("dropout", 0.1))
    model_name = config.get("model_name", "unet_small")

    train_ds = SemSegmentationDataset(load_nasa_ebc_samples(benchmark_root, datasets, "train"), image_size=image_size)
    val_ds = SemSegmentationDataset(load_nasa_ebc_samples(benchmark_root, datasets, "val"), image_size=image_size)
    test_ds = SemSegmentationDataset(load_nasa_ebc_samples(benchmark_root, datasets, "test"), image_size=image_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    device = get_device()
    model = create_segmentation_model(model_name, num_classes=num_classes, base_channels=int(config.get("base_channels", 32)), dropout=dropout).to(device)
    class_weights = compute_class_weights(train_loader.dataset, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history: list[dict[str, Any]] = []
    best_val_iou = -1.0
    checkpoint_path = project_root / config.get("checkpoint_path", "models/checkpoints/sem_ebc_unet.pt")

    for epoch in range(1, epochs + 1):
        train_loss, train_metrics = run_segmentation_epoch(model, train_loader, criterion, optimizer, device, num_classes=num_classes)
        val_loss, val_metrics = run_segmentation_epoch(model, val_loader, criterion, None, device, num_classes=num_classes)
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }
        history.append(epoch_record)
        if val_metrics["mean_iou_fg"] > best_val_iou:
            best_val_iou = val_metrics["mean_iou_fg"]
            save_checkpoint({"model_state": model.state_dict(), "config": config, "epoch": epoch}, checkpoint_path)

    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model_state"])
    test_loss, test_metrics = run_segmentation_epoch(model, test_loader, criterion, None, device, num_classes=num_classes)

    figure_path = project_root / config.get("prediction_figure_path", "reports/figures/sem_ebc_predictions.png")
    example_metadata = create_prediction_figure(model, test_loader, device, figure_path)

    results = {
        "benchmark": "NASA EBC SEM segmentation",
        "datasets": datasets,
        "num_classes": num_classes,
        "device": str(device),
        "model_name": model_name,
        "image_size": image_size,
        "epochs": epochs,
        "batch_size": batch_size,
        "dropout": dropout,
        "class_weights": class_weights.detach().cpu().tolist(),
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "test_samples": len(test_ds),
        "best_val_mean_iou_fg": best_val_iou,
        "test_loss": test_loss,
        "test_metrics": test_metrics,
        "prediction_figure_path": str(figure_path),
        "example_metadata": example_metadata,
        "history": history,
    }

    results_path = project_root / config.get("results_path", "reports/sem_ebc_segmentation_metrics.json")
    save_json(results, results_path)
    return results
