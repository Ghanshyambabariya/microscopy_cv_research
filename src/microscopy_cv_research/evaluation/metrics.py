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


def segmentation_metrics(y_true: np.ndarray, y_pred: np.ndarray, *, num_classes: int, ignore_index: int | None = None) -> dict[str, float]:
    if ignore_index is not None:
        valid = y_true != ignore_index
        y_true = y_true[valid]
        y_pred = y_pred[valid]
        if y_true.size == 0:
            return {
                "pixel_accuracy": 0.0,
                "mean_iou_fg": 0.0,
                "mean_dice_fg": 0.0,
                "per_class": {},
            }

    ious: list[float] = []
    dices: list[float] = []
    precisions: list[float] = []
    recalls: list[float] = []
    per_class: dict[str, dict[str, float]] = {}

    for class_index in range(num_classes):
        true_mask = y_true == class_index
        pred_mask = y_pred == class_index
        intersection = float(np.logical_and(true_mask, pred_mask).sum())
        union = float(np.logical_or(true_mask, pred_mask).sum())
        denom = float(true_mask.sum() + pred_mask.sum())
        predicted_count = float(pred_mask.sum())
        true_count = float(true_mask.sum())
        iou = intersection / union if union > 0 else 1.0
        dice = (2.0 * intersection) / denom if denom > 0 else 1.0
        precision = intersection / predicted_count if predicted_count > 0 else 0.0
        recall = intersection / true_count if true_count > 0 else 0.0
        per_class[str(class_index)] = {"iou": iou, "dice": dice, "precision": precision, "recall": recall}
        if class_index != 0:
            ious.append(iou)
            dices.append(dice)
            precisions.append(precision)
            recalls.append(recall)

    pixel_accuracy = float((y_true == y_pred).mean())
    return {
        "pixel_accuracy": pixel_accuracy,
        "mean_iou_fg": float(np.mean(ious)) if ious else 0.0,
        "mean_dice_fg": float(np.mean(dices)) if dices else 0.0,
        "mean_precision_fg": float(np.mean(precisions)) if precisions else 0.0,
        "mean_recall_fg": float(np.mean(recalls)) if recalls else 0.0,
        "per_class": per_class,
    }
