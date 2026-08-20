from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from microscopy_cv_research.config import load_config
from microscopy_cv_research.evaluation.metrics import regression_metrics
from microscopy_cv_research.training.engine import save_json
from microscopy_cv_research.viz.style import apply_lab_style, plot_feature_importance_panel, plot_regression_panel, save_figure


FEATURE_LABELS = {
    "cement": "Cement",
    "slag": "Blast-furnace slag",
    "ash": "Fly ash",
    "water": "Water",
    "superplastic": "Superplasticizer",
    "coarseagg": "Coarse aggregate",
    "fineagg": "Fine aggregate",
    "age": "Curing age",
}


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


def build_models(seed: int, n_estimators: int) -> dict[str, Any]:
    return {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "random_forest": RandomForestRegressor(n_estimators=n_estimators, random_state=seed),
        "xgboost": XGBRegressor(
            n_estimators=n_estimators,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=seed,
            n_jobs=1,
        ),
    }


def evaluate_models(
    models: dict[str, Any],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    target: str,
    seed: int,
) -> tuple[dict[str, Any], str, np.ndarray]:
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    results: dict[str, Any] = {}
    predictions: dict[str, np.ndarray] = {}
    for name, model in models.items():
        scores = cross_validate(
            model,
            train_df[feature_cols],
            train_df[target],
            cv=cv,
            scoring={
                "mae": "neg_mean_absolute_error",
                "rmse": "neg_root_mean_squared_error",
                "r2": "r2",
            },
            n_jobs=1,
        )
        model.fit(train_df[feature_cols], train_df[target])
        pred = np.asarray(model.predict(test_df[feature_cols]), dtype=float)
        predictions[name] = pred
        results[name] = {
            "holdout": regression_metrics(test_df[target], pred),
            "cv": {
                "mae_mean": float(-scores["test_mae"].mean()),
                "mae_std": float(scores["test_mae"].std()),
                "rmse_mean": float(-scores["test_rmse"].mean()),
                "rmse_std": float(scores["test_rmse"].std()),
                "r2_mean": float(scores["test_r2"].mean()),
                "r2_std": float(scores["test_r2"].std()),
            },
        }
    best_name = max(results, key=lambda key: results[key]["holdout"]["r2"])
    return results, best_name, predictions[best_name]


def xgb_feature_importance(model: XGBRegressor, feature_cols: list[str]) -> list[tuple[str, float]]:
    rows = [(name, float(value)) for name, value in zip(feature_cols, model.feature_importances_)]
    return sorted(rows, key=lambda row: row[1], reverse=True)


def write_shap_outputs(model: XGBRegressor, test_df: pd.DataFrame, feature_cols: list[str], config: dict[str, Any]) -> dict[str, Any]:
    shap_dir = Path(config.get("shap_dir", "reports/figures/shap_concrete"))
    shap_dir.mkdir(parents=True, exist_ok=True)
    x_test = test_df[feature_cols]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_test)
    shap_array = np.asarray(shap_values, dtype=float)
    mean_abs = np.abs(shap_array).mean(axis=0)
    shap_importance = sorted(zip(feature_cols, mean_abs), key=lambda row: row[1], reverse=True)

    apply_lab_style()
    fig, ax = plt.subplots(figsize=(7, 4.8))
    plot_feature_importance_panel(ax, shap_importance, title="SHAP global importance", max_features=8, label_map=FEATURE_LABELS)
    fig.tight_layout()
    global_path = shap_dir / "concrete_shap_global.png"
    save_figure(fig, global_path)

    plt.figure(figsize=(8, 5.5))
    shap.summary_plot(shap_array, x_test.rename(columns=FEATURE_LABELS), show=False, max_display=8)
    beeswarm_path = shap_dir / "concrete_shap_beeswarm.png"
    plt.tight_layout()
    plt.savefig(beeswarm_path, dpi=200, bbox_inches="tight")
    plt.close()

    first = shap_array[0]
    order = np.argsort(np.abs(first))[::-1][:8]
    local_rows = [(FEATURE_LABELS.get(feature_cols[i], feature_cols[i]), float(first[i])) for i in order]
    apply_lab_style()
    fig, ax = plt.subplots(figsize=(7, 4.8))
    names = [row[0] for row in local_rows][::-1]
    values = np.array([row[1] for row in local_rows][::-1])
    colors = ["#C98A1E" if value >= 0 else "#1D7A82" for value in values]
    ax.barh(np.arange(len(names)), values, color=colors, alpha=0.9)
    ax.axvline(0, color="#132436", linewidth=1)
    ax.set_yticks(np.arange(len(names)), labels=names)
    ax.set_xlabel("SHAP value")
    ax.set_title("Individual prediction explanation")
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    local_path = shap_dir / "concrete_shap_individual.png"
    save_figure(fig, local_path)

    return {
        "global_importance_path": str(global_path),
        "beeswarm_path": str(beeswarm_path),
        "individual_path": str(local_path),
        "top_shap_features": [(name, float(value)) for name, value in shap_importance[:8]],
    }


def make_figure(y_true, y_pred, report: dict[str, Any], config: dict[str, Any]) -> None:
    figure_path = Path(config["figure_path"])
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    apply_lab_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    best = report["best_model"]
    plot_regression_panel(
        axes[0],
        y_true,
        y_pred,
        report["model_results"][best]["holdout"],
        xlabel="Measured compressive strength (MPa)",
        ylabel="Predicted compressive strength (MPa)",
        title=f"Concrete strength - {best.replace('_', ' ')}",
    )
    plot_feature_importance_panel(axes[1], report["top_features"], title="What drives the prediction", max_features=8, label_map=FEATURE_LABELS)
    fig.text(
        0.01,
        0.01,
        "Data: UCI Concrete Compressive Strength | Models: Ridge, Random Forest, XGBoost | Held-out split uses seed 42",
        fontsize=8,
        color="#5B6B7A",
    )
    fig.tight_layout()
    save_figure(fig, figure_path)


def markdown_model_table(model_results: dict[str, Any]) -> list[str]:
    lines = [
        "| Model | Hold-out MAE | Hold-out RMSE | Hold-out R2 | CV R2 mean +/- std |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, result in model_results.items():
        holdout = result["holdout"]
        cv = result["cv"]
        lines.append(
            f"| {name.replace('_', ' ')} | {holdout['mae']:.4f} | {holdout['rmse']:.4f} | {holdout['r2']:.4f} | {cv['r2_mean']:.4f} +/- {cv['r2_std']:.4f} |"
        )
    return lines


def write_markdown_report(report: dict[str, Any], config: dict[str, Any]) -> None:
    best = report["best_model"]
    lines = [
        "# External Concrete Strength Benchmark",
        "",
        f"- source: [{config['source_name']}]({config['source_repo']})",
        f"- cleaned rows: `{report['data_audit']['clean_rows']}`",
        f"- features used: `{report['data_audit']['feature_count']}`",
        f"- train/test split: `{report['splits']['train']}` / `{report['splits']['test']}`",
        f"- best held-out model: `{best}`",
        "",
        "## Model Benchmark",
        "",
        *markdown_model_table(report["model_results"]),
        "",
        "## SHAP Explainability",
        "",
        "- global feature ranking: `reports/figures/shap_concrete/concrete_shap_global.png`",
        "- beeswarm plot: `reports/figures/shap_concrete/concrete_shap_beeswarm.png`",
        "- individual prediction explanation: `reports/figures/shap_concrete/concrete_shap_individual.png`",
        "",
        "![External concrete strength benchmark](figures/external_concrete_strength.png)",
        "",
        "## Why This Matters",
        "",
        "This benchmark compares linear and tree-based regressors for materials-property prediction and uses SHAP to inspect dominant composition/process variables.",
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
    seed = int(config.get("seed", 42))
    n_estimators = int(config.get("n_estimators", 300))
    train_df, test_df = train_test_split(table, test_size=float(config.get("test_size", 0.25)), random_state=seed)
    models = build_models(seed, n_estimators)
    model_results, best_name, best_pred = evaluate_models(models, train_df, test_df, feature_cols, target, seed)

    xgb_model = models["xgboost"]
    shap_outputs = write_shap_outputs(xgb_model, test_df, feature_cols, config)
    report = {
        "experiment_name": config["experiment_name"],
        "source_name": config["source_name"],
        "source_repo": config["source_repo"],
        "source_url": config["source_url"],
        "raw_path": str(raw_path),
        "clean_path": str(clean_path),
        "data_audit": audit,
        "splits": {"train": int(len(train_df)), "test": int(len(test_df))},
        "model_results": model_results,
        "best_model": best_name,
        "regression_metrics": model_results[best_name]["holdout"],
        "top_features": xgb_feature_importance(xgb_model, feature_cols),
        "shap": shap_outputs,
    }
    save_json(report, config["report_path"])
    make_figure(test_df[target].to_numpy(), best_pred, report, config)
    write_markdown_report(report, config)
    return report
