from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from microscopy_cv_research.config import load_config
from microscopy_cv_research.evaluation.metrics import classification_metrics, regression_metrics
from microscopy_cv_research.signals.features import channel_features
from microscopy_cv_research.training.engine import save_json


SENSOR_COLUMNS = ("force_z", "vibration_x", "vibration_y")


def download_uniwear(config: dict[str, Any]) -> Path:
    raw_path = Path(config["raw_path"])
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if not raw_path.exists() or raw_path.stat().st_size == 0:
        urlretrieve(config["source_url"], raw_path)
    return raw_path


def clean_uniwear_table(raw_path: str | Path, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = pd.read_csv(raw_path)
    raw = raw.drop(columns=[column for column in raw.columns if column.startswith("Unnamed")], errors="ignore")
    required = {"timestamp", *SENSOR_COLUMNS, config["target_column"], config["group_column"], "dataset_tag"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    table = raw.copy()
    for column in ["timestamp", *SENSOR_COLUMNS, config["target_column"]]:
        table[column] = pd.to_numeric(table[column], errors="coerce")
    table = table.dropna(subset=["timestamp", *SENSOR_COLUMNS, config["target_column"], config["group_column"]]).reset_index(drop=True)
    quantiles = table[config["target_column"]].quantile([0.33, 0.66]).tolist()
    table["wear_stage"] = pd.cut(table[config["target_column"]], bins=[-np.inf, quantiles[0], quantiles[1], np.inf], labels=["low_wear", "medium_wear", "high_wear"]).astype(str)
    audit = {
        "raw_rows": int(len(raw)),
        "clean_rows": int(len(table)),
        "raw_columns": int(raw.shape[1]),
        "groups": int(table[config["group_column"]].nunique()),
        "dataset_tags": sorted(table["dataset_tag"].astype(str).unique().tolist()),
        "target_min": float(table[config["target_column"]].min()),
        "target_max": float(table[config["target_column"]].max()),
        "wear_stage_counts": table["wear_stage"].value_counts().to_dict(),
    }
    return table, audit


def build_window_features(table: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    window_size = int(config.get("window_size", 64))
    stride = int(config.get("stride", max(1, window_size // 2)))
    sampling_rate_hz = int(config.get("sampling_rate_hz", 2))
    rows: list[dict[str, Any]] = []
    for experiment_tag, group in table.groupby(config["group_column"]):
        group = group.sort_values("timestamp").reset_index(drop=True)
        if len(group) < window_size:
            continue
        for start in range(0, len(group) - window_size + 1, stride):
            chunk = group.iloc[start : start + window_size]
            features: dict[str, Any] = {
                "experiment_tag": str(experiment_tag),
                "dataset_tag": str(chunk["dataset_tag"].mode().iat[0]),
                "window_start": float(chunk["timestamp"].iloc[0]),
                "window_end": float(chunk["timestamp"].iloc[-1]),
                "tool_wear": float(chunk[config["target_column"]].mean()),
                "wear_stage": str(chunk["wear_stage"].mode().iat[0]),
            }
            for column in SENSOR_COLUMNS:
                features.update(channel_features(chunk[column].to_numpy(dtype=np.float64), sampling_rate_hz, column))
            rows.append(features)
    if not rows:
        raise ValueError("No windows were created. Reduce window_size or check the dataset grouping.")
    return pd.DataFrame(rows)


def split_by_group(table: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=float(config.get("test_size", 0.3)), random_state=int(config.get("seed", 42)))
    train_idx, test_idx = next(splitter.split(table, groups=table[config["group_column"]]))
    return table.iloc[train_idx].reset_index(drop=True), table.iloc[test_idx].reset_index(drop=True)


def make_figure(feature_table: pd.DataFrame, report: dict[str, Any], config: dict[str, Any]) -> None:
    figure_path = Path(config["figure_path"])
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for dataset_tag, group in feature_table.groupby("dataset_tag"):
        axes[0].scatter(group["force_z_rms"], group["vibration_x_rms"], s=10, alpha=0.6, label=dataset_tag)
    axes[0].set_title("Uniwear window feature space")
    axes[0].set_xlabel("force_z RMS")
    axes[0].set_ylabel("vibration_x RMS")
    axes[0].legend()

    stage_counts = pd.Series(report["data_audit"]["wear_stage_counts"])
    axes[1].bar(stage_counts.index, stage_counts.values)
    axes[1].set_title("Wear-stage distribution")
    axes[1].set_ylabel("rows")
    axes[1].tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_markdown_report(report: dict[str, Any], config: dict[str, Any]) -> None:
    lines = [
        "# External Uniwear Tool-Wear Benchmark",
        "",
        f"- source: [{config['source_name']}]({config['source_repo']})",
        f"- cleaned rows: `{report['data_audit']['clean_rows']}`",
        f"- experiment groups: `{report['data_audit']['groups']}`",
        f"- window features: `{report['window_feature_rows']}`",
        f"- group split: train `{report['splits']['train']}`, test `{report['splits']['test']}`",
        "",
        "## Results",
        "",
        f"- tool-wear regression MAE: `{report['tool_wear_regression']['mae']:.4f}`",
        f"- tool-wear regression RMSE: `{report['tool_wear_regression']['rmse']:.4f}`",
        f"- tool-wear regression R2: `{report['tool_wear_regression']['r2']:.4f}`",
        f"- wear-stage accuracy: `{report['wear_stage_classification']['accuracy']:.4f}`",
        f"- wear-stage macro F1: `{report['wear_stage_classification']['macro_f1']:.4f}`",
        "",
        "![External Uniwear benchmark](figures/external_uniwear_tool_wear.png)",
        "",
        "## Why This Matters",
        "",
        "This benchmark adds a second real online materials-process dataset with vibration and force signals. It tests whether the platform can ingest a different schema, window the time series, extract features, split by experiment, train models, and return quantitative results.",
    ]
    Path(config["markdown_report_path"]).write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_external_uniwear_experiment(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    raw_path = download_uniwear(config)
    clean_table, audit = clean_uniwear_table(raw_path, config)
    clean_path = Path(config["clean_path"])
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    clean_table.to_csv(clean_path, index=False)

    feature_table = build_window_features(clean_table, config)
    feature_path = Path(config["feature_table_path"])
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    feature_table.to_csv(feature_path, index=False)

    train_df, test_df = split_by_group(feature_table, config)
    feature_cols = [c for c in feature_table.columns if c not in {"experiment_tag", "dataset_tag", "wear_stage", "tool_wear", "window_start", "window_end"}]
    stage_encoder = LabelEncoder()
    y_train_stage = stage_encoder.fit_transform(train_df["wear_stage"])
    y_test_stage = stage_encoder.transform(test_df["wear_stage"])
    regressor = make_pipeline(StandardScaler(), RandomForestRegressor(n_estimators=int(config.get("n_estimators", 250)), random_state=int(config.get("seed", 42))))
    classifier = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=int(config.get("n_estimators", 250)), random_state=int(config.get("seed", 42)), class_weight="balanced"))
    regressor.fit(train_df[feature_cols], train_df["tool_wear"])
    classifier.fit(train_df[feature_cols], y_train_stage)
    wear_pred = regressor.predict(test_df[feature_cols])
    stage_pred = classifier.predict(test_df[feature_cols])

    report = {
        "experiment_name": config["experiment_name"],
        "source_name": config["source_name"],
        "source_repo": config["source_repo"],
        "source_url": config["source_url"],
        "raw_path": str(raw_path),
        "clean_path": str(clean_path),
        "feature_table_path": str(feature_path),
        "data_audit": audit,
        "window_feature_rows": int(len(feature_table)),
        "splits": {"train": int(len(train_df)), "test": int(len(test_df)), "train_groups": sorted(train_df[config["group_column"]].unique().tolist()), "test_groups": sorted(test_df[config["group_column"]].unique().tolist())},
        "tool_wear_regression": regression_metrics(test_df["tool_wear"], wear_pred),
        "wear_stage_classification": classification_metrics(y_test_stage, stage_pred),
        "wear_stage_label_mapping": {str(i): label for i, label in enumerate(stage_encoder.classes_)},
        "top_features": sorted(zip(feature_cols, regressor.named_steps["randomforestregressor"].feature_importances_), key=lambda x: x[1], reverse=True)[:15],
    }
    save_json(report, config["report_path"])
    make_figure(feature_table, report, config)
    write_markdown_report(report, config)
    return report
