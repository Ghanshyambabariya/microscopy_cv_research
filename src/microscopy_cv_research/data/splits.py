from __future__ import annotations

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def make_group_train_val_test_split(table: pd.DataFrame, *, group_column: str, val_size: float = 0.2, test_size: float = 0.2, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    outer = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_val_idx, test_idx = next(outer.split(table, groups=table[group_column]))
    train_val = table.iloc[train_val_idx].reset_index(drop=True)
    test_df = table.iloc[test_idx].reset_index(drop=True)

    adjusted_val = val_size / (1.0 - test_size)
    inner = GroupShuffleSplit(n_splits=1, test_size=adjusted_val, random_state=random_state)
    train_idx, val_idx = next(inner.split(train_val, groups=train_val[group_column]))
    return train_val.iloc[train_idx].reset_index(drop=True), train_val.iloc[val_idx].reset_index(drop=True), test_df
