from __future__ import annotations

from pathlib import Path
import json

import torch
from torch import nn
from torch.utils.data import DataLoader


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(state: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def save_json(data: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def run_supervised_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer | None, device: torch.device) -> tuple[float, list, list]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    predictions = []
    targets = []

    for inputs, target in dataloader:
        inputs = inputs.to(device)
        if target.dtype.is_floating_point:
            target = target.to(device=device, dtype=torch.float32)
        else:
            target = target.to(device=device, dtype=torch.long)
        with torch.set_grad_enabled(training):
            output = model(inputs)
            loss = criterion(output, target)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += float(loss.item()) * inputs.size(0)
        if target.dtype.is_floating_point:
            predictions.extend(output.detach().cpu().tolist())
            targets.extend(target.detach().cpu().tolist())
        else:
            predictions.extend(output.argmax(dim=1).detach().cpu().tolist())
            targets.extend(target.detach().cpu().tolist())
    return total_loss / max(len(dataloader.dataset), 1), predictions, targets


def run_hybrid_epoch(model: nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer | None, device: torch.device, classification_weight: float, regression_weight: float, consistency_weight: float) -> tuple[float, list[int], list[int], list[float], list[float]]:
    training = optimizer is not None
    model.train(training)
    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()
    total_loss = 0.0
    class_preds = []
    class_targets = []
    reg_preds = []
    reg_targets = []

    for batch in dataloader:
        images = batch["image"].to(device)
        class_target = batch["classification_target"].to(device=device, dtype=torch.long)
        reg_target = batch["regression_target"].to(device=device, dtype=torch.float32)
        with torch.set_grad_enabled(training):
            outputs = model(images)
            classification_loss = ce_loss(outputs["classification_logits"], class_target)
            regression_loss = mse_loss(outputs["regression_output"], reg_target)
            consistency_loss = outputs["embeddings"].pow(2).mean()
            loss = classification_weight * classification_loss + regression_weight * regression_loss + consistency_weight * consistency_loss
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += float(loss.item()) * images.size(0)
        class_preds.extend(outputs["classification_logits"].argmax(dim=1).detach().cpu().tolist())
        class_targets.extend(class_target.detach().cpu().tolist())
        reg_preds.extend(outputs["regression_output"].detach().cpu().tolist())
        reg_targets.extend(reg_target.detach().cpu().tolist())
    return total_loss / max(len(dataloader.dataset), 1), class_preds, class_targets, reg_preds, reg_targets
