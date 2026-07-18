from __future__ import annotations

from pathlib import Path

import pandas as pd

from microscopy_cv_research.signals.features import CHANNELS


def load_signal_csv(path: str | Path, channel_columns: tuple[str, ...] = CHANNELS) -> dict[str, pd.Series]:
    table = pd.read_csv(path)
    missing = [column for column in channel_columns if column not in table.columns]
    if missing:
        raise ValueError(f"Missing required signal columns in {path}: {missing}")
    return {column: table[column].astype("float32") for column in channel_columns}

