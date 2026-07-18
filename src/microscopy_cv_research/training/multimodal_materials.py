from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from microscopy_cv_research.config import load_config
from microscopy_cv_research.evaluation.metrics import classification_metrics, regression_metrics
from microscopy_cv_research.training.engine import save_json
from microscopy_cv_research.training.materials_signal import build_signal_feature_table, load_specimen_table


def build_multimodal_table(config: dict[str, Any]) -> pd.DataFrame:
    signal_path = Path(config["signal_feature_table_path"])
    if signal_path.exists():
        signal_features = pd.read_csv(signal_path)
    else:
        signal_features = build_signal_feature_table(config)
        signal_path.parent.mkdir(parents=True, exist_ok=True)
        signal_features.to_csv(signal_path, index=False)

    specimens = load_specimen_table(config["microscopy_labels_csv"])
    microscopy_summary = specimens.rename(
        columns={
            "target_class": "microscopy_target_class",
            "property_value": "microscopy_property_value",
            "magnification": "microscopy_magnification",
        }
    )
    table = signal_features.merge(microscopy_summary, on="specimen_id", how="inner")
    table["property_delta_abs"] = (table["property_value"] - table["microscopy_property_value"]).abs()
    return table


def run_multimodal_materials_experiment(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    seed = int(config.get("seed", 42))
    table = build_multimodal_table(config)
    output_table = Path(config.get("multimodal_table_path", "data/processed/materials_multimodal_table.csv"))
    output_table.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_table, index=False)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    train_idx, test_idx = next(splitter.split(table, groups=table["specimen_id"]))
    train_df = table.iloc[train_idx].reset_index(drop=True)
    test_df = table.iloc[test_idx].reset_index(drop=True)

    ignore = {"run_id", "specimen_id", "target_class", "property_value", "process_quality", "microscopy_target_class"}
    feature_cols = [c for c in table.columns if c not in ignore]
    numeric_cols = table[feature_cols].select_dtypes(include="number").columns.tolist()
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_cols),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    quality_encoder = LabelEncoder()
    y_train = quality_encoder.fit_transform(train_df["process_quality"])
    y_test = quality_encoder.transform(test_df["process_quality"])
    quality_model = make_pipeline(preprocessor, RandomForestClassifier(n_estimators=int(config.get("n_estimators", 250)), random_state=seed, class_weight="balanced"))
    property_model = make_pipeline(preprocessor, RandomForestRegressor(n_estimators=int(config.get("n_estimators", 250)), random_state=seed))

    quality_model.fit(train_df[feature_cols], y_train)
    property_model.fit(train_df[feature_cols], train_df["property_value"])
    quality_pred = quality_model.predict(test_df[feature_cols])
    property_pred = property_model.predict(test_df[feature_cols])

    report = {
        "experiment_name": config.get("experiment_name", "multimodal_materials_intelligence"),
        "task": "fusion of microscopy specimen descriptors and grinding signal features",
        "multimodal_table_path": str(output_table),
        "splits": {"train": len(train_df), "test": len(test_df)},
        "num_signal_features": len([c for c in feature_cols if c.startswith(("Fx_", "Fy_", "Fz_", "Mz_", "F_resultant_"))]),
        "num_microscopy_features": len([c for c in feature_cols if c.startswith("microscopy_")]),
        "quality_label_mapping": {str(i): label for i, label in enumerate(quality_encoder.classes_)},
        "quality_metrics": classification_metrics(y_test, quality_pred),
        "property_metrics": regression_metrics(test_df["property_value"], property_pred),
        "notes": [
            "This fusion baseline is designed as a portfolio-ready bridge between microscopy CV and process-signal ML.",
            "Replace simulated grinding signals with measured force, acoustic, vibration, or spindle-current CSV files when available.",
        ],
    }
    save_json(report, config.get("report_path", "reports/multimodal_materials_metrics.json"))
    return report
