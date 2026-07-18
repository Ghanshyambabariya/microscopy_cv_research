from __future__ import annotations

import pandas as pd

from microscopy_cv_research.training.external_concrete import clean_concrete_table


def test_clean_concrete_table(tmp_path) -> None:
    raw_path = tmp_path / "concrete.csv"
    pd.DataFrame(
        {
            "cement": [100, 200],
            "water": [150, 120],
            "age": [28, 56],
            "strength": [25.0, 45.0],
        }
    ).to_csv(raw_path, index=False)
    cleaned, audit = clean_concrete_table(raw_path, {"target_column": "strength"})
    assert len(cleaned) == 2
    assert audit["feature_count"] == 3

