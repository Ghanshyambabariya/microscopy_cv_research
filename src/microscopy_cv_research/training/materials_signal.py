from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from microscopy_cv_research.config import load_config
from microscopy_cv_research.evaluation.metrics import classification_metrics, regression_metrics
from microscopy_cv_research.signals.features import extract_signal_features
from microscopy_cv_research.signals.simulation import simulate_grinding_signals
from microscopy_cv_research.training.engine import save_json
from microscopy_cv_research.viz.style import apply_lab_style, annotate_metrics, plot_confusion_panel, save_figure


def load_specimen_table(labels_csv: str | Path) -> pd.DataFrame:
    labels = pd.read_csv(labels_csv)
    return (
        labels.groupby("specimen_id")
        .agg(
            target_class=("target_class", lambda x: x.mode().iat[0]),
            property_value=("property_value", "mean"),
            magnification=("magnification", "mean"),
            batch_id=("batch_id", lambda x: x.mode().iat[0]),
        )
        .reset_index()
    )


def build_signal_feature_table(config: dict[str, Any]) -> pd.DataFrame:
    specimens = load_specimen_table(config["microscopy_labels_csv"])
    rows: list[dict[str, Any]] = []
    sampling_rate_hz = int(config.get("sampling_rate_hz", 20_000))
    duration_seconds = float(config.get("analysis_window_seconds", config.get("duration_seconds", 2.0)))
    seed = int(config.get("seed", 42))

    for idx, row in specimens.iterrows():
        run = simulate_grinding_signals(
            specimen_id=row["specimen_id"],
            target_class=row["target_class"],
            property_value=float(row["property_value"]),
            sampling_rate_hz=sampling_rate_hz,
            duration_seconds=duration_seconds,
            seed=seed + idx,
        )
        features = extract_signal_features(run.signals, sampling_rate_hz)
        features.update(
            {
                "run_id": run.run_id,
                "specimen_id": run.specimen_id,
                "sampling_rate_hz": run.sampling_rate_hz,
                "analysis_window_seconds": run.duration_seconds,
                "nominal_process_seconds": float(config.get("duration_seconds", run.duration_seconds)),
                "target_class": run.target_class,
                "property_value": run.property_value,
                "process_quality": run.process_quality,
            }
        )
        rows.append(features)
    return pd.DataFrame(rows)


def split_by_specimen(table: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    train_idx, test_idx = next(splitter.split(table, groups=table["specimen_id"]))
    return table.iloc[train_idx].reset_index(drop=True), table.iloc[test_idx].reset_index(drop=True)


def make_signal_summary_figure(table: pd.DataFrame, report: dict[str, Any], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    apply_lab_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for quality, group in table.groupby("process_quality"):
        axes[0].scatter(group["F_resultant_rms"], group["F_resultant_bandpower_high"], label=quality, alpha=0.75, edgecolor="#FFFFFF", linewidth=0.3)
    axes[0].set_xlabel("Resultant force RMS")
    axes[0].set_ylabel("High-frequency bandpower")
    axes[0].set_title("Grinding signal feature space")
    axes[0].legend()
    annotate_metrics(
        axes[0],
        {
            "accuracy": f"{report['quality_metrics']['accuracy']:.3f}",
            "R2": f"{report['property_metrics']['r2']:.3f}",
            "features": report["num_features"],
        },
    )

    labels = report["quality_label_mapping"]
    confusion = pd.DataFrame(report["quality_confusion_matrix"]).values
    display_labels = [labels[str(i)] for i in range(len(labels))]
    plot_confusion_panel(axes[1], confusion, display_labels, "Process quality counts", normalize=False)
    plot_confusion_panel(axes[2], confusion, display_labels, "Process quality normalized", normalize=True)
    fig.tight_layout()
    save_figure(fig, output_path)


def run_materials_signal_experiment(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    seed = int(config.get("seed", 42))
    feature_table = build_signal_feature_table(config)
    feature_path = Path(config.get("feature_table_path", "data/processed/materials_signal_features.csv"))
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    feature_table.to_csv(feature_path, index=False)

    train_df, test_df = split_by_specimen(feature_table, seed)
    feature_cols = [c for c in feature_table.columns if c not in {"run_id", "specimen_id", "target_class", "property_value", "process_quality", "batch_id"}]
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df["process_quality"])
    y_test = label_encoder.transform(test_df["process_quality"])

    quality_model = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(n_estimators=int(config.get("n_estimators", 200)), random_state=seed, class_weight="balanced"),
    )
    property_model = make_pipeline(
        StandardScaler(),
        RandomForestRegressor(n_estimators=int(config.get("n_estimators", 200)), random_state=seed),
    )
    quality_model.fit(train_df[feature_cols], y_train)
    property_model.fit(train_df[feature_cols], train_df["property_value"])
    quality_pred = quality_model.predict(test_df[feature_cols])
    property_pred = property_model.predict(test_df[feature_cols])

    report = {
        "experiment_name": config.get("experiment_name", "materials_signal_intelligence"),
        "task": "high-frequency grinding signal classification and property regression",
        "sampling_rate_hz": int(config.get("sampling_rate_hz", 20_000)),
        "nominal_process_seconds": float(config.get("duration_seconds", 20.0)),
        "analysis_window_seconds": float(config.get("analysis_window_seconds", config.get("duration_seconds", 2.0))),
        "feature_table_path": str(feature_path),
        "splits": {"train": len(train_df), "test": len(test_df)},
        "num_features": len(feature_cols),
        "quality_label_mapping": {str(i): label for i, label in enumerate(label_encoder.classes_)},
        "quality_metrics": classification_metrics(y_test, quality_pred),
        "quality_confusion_matrix": pd.crosstab(pd.Series(y_test, name="true"), pd.Series(quality_pred, name="pred"), dropna=False).reindex(index=range(len(label_encoder.classes_)), columns=range(len(label_encoder.classes_)), fill_value=0).values.tolist(),
        "property_metrics": regression_metrics(test_df["property_value"], property_pred),
        "top_signal_features": sorted(
            zip(feature_cols, quality_model.named_steps["randomforestclassifier"].feature_importances_),
            key=lambda x: x[1],
            reverse=True,
        )[:12],
    }
    make_signal_summary_figure(feature_table, report, config.get("figure_path", "reports/figures/materials_signal_summary.png"))
    report["figure_path"] = config.get("figure_path", "reports/figures/materials_signal_summary.png")
    save_json(report, config.get("report_path", "reports/materials_signal_metrics.json"))
    return report
