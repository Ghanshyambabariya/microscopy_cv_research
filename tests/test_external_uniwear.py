from __future__ import annotations

import pandas as pd

from microscopy_cv_research.training.external_uniwear import build_window_features, clean_uniwear_table


def test_clean_uniwear_and_build_windows(tmp_path) -> None:
    raw_path = tmp_path / "uniwear.csv"
    pd.DataFrame(
        {
            "timestamp": list(range(8)),
            "force_z": [1, 2, 3, 4, 5, 6, 7, 8],
            "vibration_x": [2, 2, 3, 3, 4, 4, 5, 5],
            "vibration_y": [1, 1, 1, 2, 2, 2, 3, 3],
            "tool_wear": [0.1, 0.1, 0.2, 0.2, 0.7, 0.7, 0.8, 0.8],
            "experiment_tag": ["W1"] * 8,
            "dataset_tag": ["test"] * 8,
        }
    ).to_csv(raw_path, index=False)
    config = {"target_column": "tool_wear", "group_column": "experiment_tag", "window_size": 4, "stride": 2}
    cleaned, audit = clean_uniwear_table(raw_path, config)
    features = build_window_features(cleaned, config)
    assert audit["clean_rows"] == 8
    assert len(features) == 3
    assert "force_z_rms" in features.columns

