from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "ink": "#132436",
    "muted": "#5B6B7A",
    "grid": "#E3E8EB",
    "paper": "#FBFCFC",
    "panel": "#FBFCFC",
    "teal": "#1D7A82",
    "blue": "#355C9B",
    "gold": "#C9952D",
    "green": "#4C956C",
    "red": "#C94C4C",
    "purple": "#7B5EA7",
}


def apply_lab_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": COLORS["paper"],
            "axes.facecolor": COLORS["panel"],
            "axes.edgecolor": "#B9C7BF",
            "axes.labelcolor": COLORS["ink"],
            "axes.titlecolor": COLORS["ink"],
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.8,
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelsize": 11,
            "xtick.color": COLORS["muted"],
            "ytick.color": COLORS["muted"],
            "legend.frameon": True,
            "legend.facecolor": COLORS["panel"],
            "legend.edgecolor": "#D6DED8",
            "savefig.facecolor": COLORS["paper"],
        }
    )


def save_figure(fig: plt.Figure, output_path: str | Path, dpi: int = 200) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def annotate_metrics(ax: plt.Axes, metrics: Mapping[str, float | int | str], loc: str = "upper left") -> None:
    lines = [f"{key}: {value}" for key, value in metrics.items()]
    x = 0.03 if "left" in loc else 0.97
    y = 0.97 if "upper" in loc else 0.03
    ha = "left" if "left" in loc else "right"
    va = "top" if "upper" in loc else "bottom"
    ax.text(
        x,
        y,
        "\n".join(lines),
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=9,
        color=COLORS["ink"],
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#FFFFFFE6", "edgecolor": "#D2DDD5"},
    )


def plot_regression_panel(
    ax: plt.Axes,
    y_true: Sequence[float],
    y_pred: Sequence[float],
    metrics: Mapping[str, float],
    xlabel: str,
    ylabel: str,
    title: str,
) -> None:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    residuals = y_pred_arr - y_true_arr
    rmse = float(metrics.get("rmse", np.sqrt(np.mean(residuals**2))))
    scatter = ax.scatter(
        y_true_arr,
        y_pred_arr,
        c=np.abs(residuals),
        cmap="magma_r",
        s=34,
        alpha=0.82,
        edgecolor="#FFFFFF",
        linewidth=0.35,
    )
    lo = float(min(y_true_arr.min(), y_pred_arr.min()))
    hi = float(max(y_true_arr.max(), y_pred_arr.max()))
    ax.plot([lo, hi], [lo, hi], color=COLORS["ink"], linewidth=1.2, label="ideal")
    ax.fill_between([lo, hi], [lo - rmse, hi - rmse], [lo + rmse, hi + rmse], color=COLORS["teal"], alpha=0.13, label="+/- 1 RMSE")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="lower right")
    annotate_metrics(
        ax,
        {
            "R2": f"{metrics.get('r2', float('nan')):.3f}",
            "MAE": f"{metrics.get('mae', float('nan')):.2f}",
            "RMSE": f"{rmse:.2f}",
            "n": len(y_true_arr),
        },
    )
    plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label="absolute residual")


def plot_feature_importance_panel(
    ax: plt.Axes,
    top_features: Iterable[Sequence[object]],
    title: str = "Feature importance",
    max_features: int = 8,
    label_map: Mapping[str, str] | None = None,
) -> None:
    rows = [(str(name), float(value)) for name, value in list(top_features)[:max_features]]
    if not rows:
        ax.text(0.5, 0.5, "No feature importance available", ha="center", va="center")
        ax.axis("off")
        return
    raw_names, values = zip(*rows)
    names = tuple((label_map or {}).get(name, name) for name in raw_names)
    y_pos = np.arange(len(names))
    top = max(values)
    colors = [COLORS["gold"] if value == top else "#9FD2D5" for value in values]
    bars = ax.barh(y_pos, values, color=colors, alpha=0.95, height=0.62)
    ax.set_yticks(y_pos, labels=names)
    ax.invert_yaxis()
    ax.set_xlabel("importance")
    ax.set_title(title)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    for bar, value in zip(bars, values):
        ax.text(bar.get_width() + top * 0.015, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", fontsize=9, color=COLORS["muted"])
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)


def plot_confusion_panel(
    ax: plt.Axes,
    confusion: Sequence[Sequence[int | float]],
    labels: Sequence[str],
    title: str,
    normalize: bool = False,
) -> None:
    matrix = np.asarray(confusion, dtype=float)
    raw = matrix.copy()
    if normalize:
        row_sums = matrix.sum(axis=1, keepdims=True)
        matrix = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums != 0)
    im = ax.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=1 if normalize else None)
    ax.set_title(title)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_xticks(np.arange(len(labels)), labels=labels, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(labels)), labels=labels)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            text = f"{matrix[row, col]:.2f}" if normalize else f"{int(raw[row, col])}"
            threshold = 0.55 if normalize else max(float(matrix.max()) * 0.55, 1.0)
            text_color = "#FFFFFF" if matrix[row, col] >= threshold else COLORS["ink"]
            ax.text(col, row, text, ha="center", va="center", color=text_color, fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def style_axis(ax: plt.Axes, title: str | None = None) -> None:
    if title:
        ax.set_title(title)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
