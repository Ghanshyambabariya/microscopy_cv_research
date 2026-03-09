from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class MicroscopyImageDataset(Dataset):
    def __init__(self, table: pd.DataFrame, image_root: str | Path, target_column: str | None = None, transform=None) -> None:
        self.table = table.reset_index(drop=True)
        self.image_root = Path(image_root)
        self.target_column = target_column
        self.transform = transform

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, index: int):
        row = self.table.iloc[index]
        image = Image.open(self.image_root / row["image_path"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        if self.target_column is None:
            return image, row.to_dict()
        return image, row[self.target_column]


class MicroscopyHybridDataset(Dataset):
    def __init__(self, table: pd.DataFrame, image_root: str | Path, classification_target: str, regression_target: str, transform=None) -> None:
        self.table = table.reset_index(drop=True)
        self.image_root = Path(image_root)
        self.classification_target = classification_target
        self.regression_target = regression_target
        self.transform = transform

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, index: int):
        row = self.table.iloc[index]
        image = Image.open(self.image_root / row["image_path"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return {
            "image": image,
            "classification_target": row[self.classification_target],
            "regression_target": float(row[self.regression_target]),
        }
