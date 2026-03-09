from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as F


@dataclass(slots=True)
class SegmentationSample:
    image_path: Path
    mask_path: Path
    dataset_name: str
    split: str


class SemSegmentationDataset(Dataset):
    def __init__(self, samples: list[SegmentationSample], image_size: int = 256) -> None:
        self.samples = samples
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.samples[index]
        image = Image.open(sample.image_path).convert("RGB")
        mask = Image.open(sample.mask_path).convert("L")

        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)

        image_tensor = F.to_tensor(image)
        image_tensor = F.normalize(image_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        mask_array = np.array(mask, dtype=np.int64)
        mask_tensor = torch.from_numpy(mask_array)

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "image_path": str(sample.image_path),
            "mask_path": str(sample.mask_path),
            "dataset_name": sample.dataset_name,
            "split": sample.split,
        }


def resolve_annotation_path(dataset_dir: Path, split: str, image_path: Path) -> Path:
    annot_dir = dataset_dir / f"{split}_annot"
    direct = annot_dir / image_path.name
    if direct.exists():
        return direct
    candidate = annot_dir / f"{image_path.stem}_mask{image_path.suffix}"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Could not find mask for {image_path}")


def load_nasa_ebc_samples(root: Path, datasets: list[str], split: str) -> list[SegmentationSample]:
    samples: list[SegmentationSample] = []
    for dataset_name in datasets:
        dataset_dir = root / dataset_name
        for image_path in sorted((dataset_dir / split).glob("*.tif")):
            samples.append(
                SegmentationSample(
                    image_path=image_path,
                    mask_path=resolve_annotation_path(dataset_dir, split, image_path),
                    dataset_name=dataset_name,
                    split=split,
                )
            )
    return samples
