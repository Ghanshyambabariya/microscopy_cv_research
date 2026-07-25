from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from microscopy_cv_research.config import load_config
from microscopy_cv_research.evaluation.metrics import classification_metrics, regression_metrics
from microscopy_cv_research.training.engine import save_json
from microscopy_cv_research.viz.style import COLORS, apply_lab_style, annotate_metrics, plot_confusion_panel, save_figure
import matplotlib.pyplot as plt


def download_source(config: dict[str, Any]) -> Path:
    raw_path = Path(config["raw_path"])
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if not raw_path.exists() or raw_path.stat().st_size == 0:
        urlretrieve(config["source_url"], raw_path)
    return raw_path


def clean_tool_wear_table(raw_path: str | Path, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = pd.read_csv(raw_path)
    raw.columns = [column.strip() for column in raw.columns]
    required = {config["target_column"], config["group_column"]}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    table = raw.copy()
    for column in table.columns:
        table[column] = pd.to_numeric(table[column], errors="coerce")
    table = table.dropna(subset=[config["target_column"], config["group_column"]])
    table[config["group_column"]] = table[config["group_column"]].astype("int64").astype(str)
    table = table.drop_duplicates().reset_index(drop=True)

    numeric_features = [c for c in table.columns if c not in {config["target_column"], config["group_column"]}]
    keep_features = [c for c in numeric_features if table[c].notna().mean() >= 0.95 and table[c].nunique(dropna=True) > 1]
    cleaned = table[[*keep_features, config["group_column"], config["target_column"]]].copy()

    quantiles = cleaned[config["target_column"]].quantile([0.33, 0.66]).tolist()
    cleaned["wear_stage"] = pd.cut(
        cleaned[config["target_column"]],
        bins=[-np.inf, quantiles[0], quantiles[1], np.inf],
        labels=["low_wear", "medium_wear", "high_wear"],
    ).astype(str)

    audit = {
        "raw_rows": int(len(raw)),
        "clean_rows": int(len(cleaned)),
        "raw_columns": int(raw.shape[1]),
        "model_features": int(len(keep_features)),
        "groups": int(cleaned[config["group_column"]].nunique()),
        "target_min": float(cleaned[config["target_column"]].min()),
        "target_max": float(cleaned[config["target_column"]].max()),
        "wear_stage_counts": cleaned["wear_stage"].value_counts().to_dict(),
        "dropped_feature_count": int(len(numeric_features) - len(keep_features)),
    }
    return cleaned, audit


def split_by_group(table: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=float(config.get("test_size", 0.3)), random_state=int(config.get("seed", 42)))
    train_idx, test_idx = next(splitter.split(table, groups=table[config["group_column"]]))
    return table.iloc[train_idx].reset_index(drop=True), table.iloc[test_idx].reset_index(drop=True)


def make_figure(table: pd.DataFrame, report: dict[str, Any], config: dict[str, Any]) -> None:
    figure_path = Path(config["figure_path"])
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    apply_lab_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    for tool, group in table.groupby(config["group_column"]):
        axes[0].plot(group.index, group[config["target_column"]], marker=".", linewidth=0.85, alpha=0.82, label=f"tool {tool}")
    axes[0].set_title("Flank wear by segment")
    axes[0].set_xlabel("cleaned row index")
    axes[0].set_ylabel("Vb flank wear")
    axes[0].legend(ncol=2, fontsize=7)
    annotate_metrics(
        axes[0],
        {
            "R2": f"{report['flank_wear_regression']['r2']:.3f}",
            "MAE": f"{report['flank_wear_regression']['mae']:.1f}",
            "held-out tools": len(report["splits"]["test_groups"]),
        },
        loc="upper right",
    )

    labels = report["wear_stage_label_mapping"]
    confusion = np.asarray(report["wear_stage_confusion_matrix"])
    display_labels = [labels[str(i)] for i in range(len(labels))]
    plot_confusion_panel(axes[1], confusion, display_labels, "Wear-stage counts", normalize=False)
    plot_confusion_panel(axes[2], confusion, display_labels, "Wear-stage normalized", normalize=True)
    annotate_metrics(
        axes[2],
        {
            "accuracy": f"{report['wear_stage_classification']['accuracy']:.3f}",
            "macro F1": f"{report['wear_stage_classification']['macro_f1']:.3f}",
        },
        loc="lower right",
    )
    fig.tight_layout()
    save_figure(fig, figure_path)


def write_markdown_report(report: dict[str, Any], config: dict[str, Any]) -> None:
    lines = [
        "# External Tool Wear Benchmark",
        "",
        f"- source: [{config['source_name']}]({config['source_repo']})",
        f"- rows after cleaning: `{report['data_audit']['clean_rows']}`",
        f"- tools/groups: `{report['data_audit']['groups']}`",
        f"- features used: `{report['data_audit']['model_features']}`",
        f"- group split: train `{report['splits']['train']}`, test `{report['splits']['test']}`",
        "",
        "## Results",
        "",
        f"- flank-wear regression MAE: `{report['flank_wear_regression']['mae']:.4f}`",
        f"- flank-wear regression RMSE: `{report['flank_wear_regression']['rmse']:.4f}`",
        f"- flank-wear regression R2: `{report['flank_wear_regression']['r2']:.4f}`",
        f"- wear-stage accuracy: `{report['wear_stage_classification']['accuracy']:.4f}`",
        f"- wear-stage macro F1: `{report['wear_stage_classification']['macro_f1']:.4f}`",
        "",
        "![External tool wear benchmark](figures/external_tool_wear_vicomtech.png)",
        "",
        "## Why This Matters",
        "",
        "This is a real GitHub-hosted machine-tool dataset. It gives the project a direct path from online data extraction to cleaning, preprocessing, grouped train/test splitting, model fitting, and quantitative evaluation.",
    ]
    Path(config["markdown_report_path"]).write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_external_tool_wear_experiment(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    raw_path = download_source(config)
    clean_table, audit = clean_tool_wear_table(raw_path, config)
    clean_path = Path(config["clean_path"])
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    clean_table.to_csv(clean_path, index=False)

    train_df, test_df = split_by_group(clean_table, config)
    feature_cols = [c for c in clean_table.columns if c not in {config["target_column"], config["group_column"], "wear_stage"}]
    regressor = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), RandomForestRegressor(n_estimators=int(config.get("n_estimators", 300)), random_state=int(config.get("seed", 42))))
    classifier = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), RandomForestClassifier(n_estimators=int(config.get("n_estimators", 300)), random_state=int(config.get("seed", 42)), class_weight="balanced"))

    stage_encoder = LabelEncoder()
    y_train_stage = stage_encoder.fit_transform(train_df["wear_stage"])
    y_test_stage = stage_encoder.transform(test_df["wear_stage"])
    regressor.fit(train_df[feature_cols], train_df[config["target_column"]])
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
        "data_audit": audit,
        "splits": {"train": int(len(train_df)), "test": int(len(test_df)), "train_groups": sorted(train_df[config["group_column"]].unique().tolist()), "test_groups": sorted(test_df[config["group_column"]].unique().tolist())},
        "flank_wear_regression": regression_metrics(test_df[config["target_column"]], wear_pred),
        "wear_stage_classification": classification_metrics(y_test_stage, stage_pred),
        "wear_stage_label_mapping": {str(i): label for i, label in enumerate(stage_encoder.classes_)},
        "wear_stage_confusion_matrix": pd.crosstab(pd.Series(y_test_stage, name="true"), pd.Series(stage_pred, name="pred"), dropna=False).reindex(index=range(len(stage_encoder.classes_)), columns=range(len(stage_encoder.classes_)), fill_value=0).values.tolist(),
        "top_features": sorted(
            zip(feature_cols, regressor.named_steps["randomforestregressor"].feature_importances_),
            key=lambda x: x[1],
            reverse=True,
        )[:15],
    }
    save_json(report, config["report_path"])
    make_figure(clean_table, report, config)
    write_markdown_report(report, config)
    return report
