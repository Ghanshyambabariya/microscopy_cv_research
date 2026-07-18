from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from microscopy_cv_research.config import load_config
from microscopy_cv_research.evaluation.metrics import regression_metrics
from microscopy_cv_research.training.engine import save_json


def download_concrete(config: dict[str, Any]) -> Path:
    raw_path = Path(config["raw_path"])
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if not raw_path.exists() or raw_path.stat().st_size == 0:
        urlretrieve(config["source_url"], raw_path)
    return raw_path


def clean_concrete_table(raw_path: str | Path, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    table = pd.read_csv(raw_path)
    table.columns = [column.strip().lower().replace(" ", "_") for column in table.columns]
    target = config["target_column"]
    if target not in table.columns:
        raise ValueError(f"Target column {target!r} was not found. Columns: {list(table.columns)}")
    for column in table.columns:
        table[column] = pd.to_numeric(table[column], errors="coerce")
    table = table.dropna().drop_duplicates().reset_index(drop=True)
    audit = {
        "raw_rows": int(pd.read_csv(raw_path).shape[0]),
        "clean_rows": int(len(table)),
        "columns": table.columns.tolist(),
        "feature_count": int(len(table.columns) - 1),
        "target_min": float(table[target].min()),
        "target_max": float(table[target].max()),
        "missing_values_after_cleaning": int(table.isna().sum().sum()),
    }
    return table, audit


def make_figure(y_true, y_pred, config: dict[str, Any]) -> None:
    figure_path = Path(config["figure_path"])
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(y_true, y_pred, alpha=0.7)
    lo = min(min(y_true), min(y_pred))
    hi = max(max(y_true), max(y_pred))
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1)
    ax.set_xlabel("Measured compressive strength")
    ax.set_ylabel("Predicted compressive strength")
    ax.set_title("Concrete strength regression")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_markdown_report(report: dict[str, Any], config: dict[str, Any]) -> None:
    lines = [
        "# External Concrete Strength Benchmark",
        "",
        f"- source: [{config['source_name']}]({config['source_repo']})",
        f"- cleaned rows: `{report['data_audit']['clean_rows']}`",
        f"- features used: `{report['data_audit']['feature_count']}`",
        f"- train/test split: `{report['splits']['train']}` / `{report['splits']['test']}`",
        "",
        "## Results",
        "",
        f"- compressive-strength MAE: `{report['regression_metrics']['mae']:.4f}`",
        f"- compressive-strength RMSE: `{report['regression_metrics']['rmse']:.4f}`",
        f"- compressive-strength R2: `{report['regression_metrics']['r2']:.4f}`",
        "",
        "![External concrete strength benchmark](figures/external_concrete_strength.png)",
        "",
        "## Why This Matters",
        "",
        "This benchmark strengthens the materials-informatics side of the platform with a compact real property-prediction dataset: ingredient/process variables to mechanical strength.",
    ]
    Path(config["markdown_report_path"]).write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_external_concrete_experiment(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    raw_path = download_concrete(config)
    table, audit = clean_concrete_table(raw_path, config)
    clean_path = Path(config["clean_path"])
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(clean_path, index=False)

    target = config["target_column"]
    feature_cols = [column for column in table.columns if column != target]
    train_df, test_df = train_test_split(table, test_size=float(config.get("test_size", 0.25)), random_state=int(config.get("seed", 42)))
    model = make_pipeline(StandardScaler(), RandomForestRegressor(n_estimators=int(config.get("n_estimators", 300)), random_state=int(config.get("seed", 42))))
    model.fit(train_df[feature_cols], train_df[target])
    pred = model.predict(test_df[feature_cols])
    report = {
        "experiment_name": config["experiment_name"],
        "source_name": config["source_name"],
        "source_repo": config["source_repo"],
        "source_url": config["source_url"],
        "raw_path": str(raw_path),
        "clean_path": str(clean_path),
        "data_audit": audit,
        "splits": {"train": int(len(train_df)), "test": int(len(test_df))},
        "regression_metrics": regression_metrics(test_df[target], pred),
        "top_features": sorted(zip(feature_cols, model.named_steps["randomforestregressor"].feature_importances_), key=lambda x: x[1], reverse=True),
    }
    save_json(report, config["report_path"])
    make_figure(test_df[target].to_numpy(), pred, config)
    write_markdown_report(report, config)
    return report

