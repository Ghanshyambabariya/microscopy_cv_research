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
from microscopy_cv_research.viz.style import apply_lab_style, plot_feature_importance_panel, plot_regression_panel, save_figure


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


def make_figure(y_true, y_pred, report: dict[str, Any], config: dict[str, Any]) -> None:
    figure_path = Path(config["figure_path"])
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    apply_lab_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot_regression_panel(
        axes[0],
        y_true,
        y_pred,
        report["regression_metrics"],
        xlabel="Measured compressive strength (MPa)",
        ylabel="Predicted compressive strength (MPa)",
        title="Concrete compressive strength - held-out fit",
    )
    label_map = {
        "cement": "Cement",
        "slag": "Blast-furnace slag",
        "ash": "Fly ash",
        "water": "Water",
        "superplastic": "Superplasticizer",
        "coarseagg": "Coarse aggregate",
        "fineagg": "Fine aggregate",
        "age": "Curing age",
    }
    plot_feature_importance_panel(axes[1], report["top_features"], title="What drives the prediction", max_features=8, label_map=label_map)
    fig.text(
        0.01,
        0.01,
        "Data: UCI Concrete Compressive Strength | Model: RandomForestRegressor(n_estimators=300) | Held-out split uses seed 42",
        fontsize=8,
        color="#5B6B7A",
    )
    fig.tight_layout()
    save_figure(fig, figure_path)


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
        "This benchmark tests a compact real property-prediction dataset: ingredient/process variables to mechanical strength.",
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
    make_figure(test_df[target].to_numpy(), pred, report, config)
    write_markdown_report(report, config)
    return report
