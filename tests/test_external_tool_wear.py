from __future__ import annotations

import pandas as pd

from microscopy_cv_research.training.external_tool_wear import clean_tool_wear_table


def test_clean_tool_wear_table_derives_stage(tmp_path) -> None:
    raw_path = tmp_path / "tool.csv"
    pd.DataFrame(
        {
            "F_c_RMS": [1.0, 2.0, 3.0, 4.0],
            "AE_RMS": [0.1, 0.2, 0.3, 0.4],
            "tool": [1, 1, 2, 2],
            "Vb": [0.0, 0.2, 0.5, 0.8],
        }
    ).to_csv(raw_path, index=False)
    config = {"target_column": "Vb", "group_column": "tool"}
    cleaned, audit = clean_tool_wear_table(raw_path, config)
    assert "wear_stage" in cleaned.columns
    assert audit["clean_rows"] == 4
    assert audit["model_features"] == 2

