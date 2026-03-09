from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score


def classification_metrics(y_true, y_pred) -> dict[str, float]:
    return {"accuracy": float(accuracy_score(y_true, y_pred)), "macro_f1": float(f1_score(y_true, y_pred, average="macro"))}


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "r2": float(r2_score(y_true, y_pred)),
    }


def segmentation_metrics(y_true: np.ndarray, y_pred: np.ndarray, *, num_classes: int) -> dict[str, float]:
    ious: list[float] = []
    dices: list[float] = []
    per_class: dict[str, dict[str, float]] = {}

    for class_index in range(num_classes):
        true_mask = y_true == class_index
        pred_mask = y_pred == class_index
        intersection = float(np.logical_and(true_mask, pred_mask).sum())
        union = float(np.logical_or(true_mask, pred_mask).sum())
        denom = float(true_mask.sum() + pred_mask.sum())
        iou = intersection / union if union > 0 else 1.0
        dice = (2.0 * intersection) / denom if denom > 0 else 1.0
        per_class[str(class_index)] = {"iou": iou, "dice": dice}
        if class_index != 0:
            ious.append(iou)
            dices.append(dice)

    pixel_accuracy = float((y_true == y_pred).mean())
    return {
        "pixel_accuracy": pixel_accuracy,
        "mean_iou_fg": float(np.mean(ious)) if ious else 0.0,
        "mean_dice_fg": float(np.mean(dices)) if dices else 0.0,
        "per_class": per_class,
    }
