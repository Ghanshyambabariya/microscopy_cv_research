from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import List

from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F


@dataclass(slots=True)
class SegmentationSample:
    image_path: Path
    mask_path: Path
    dataset_name: str
    split: str


class SemSegmentationDataset(Dataset):
    def __init__(self, samples: List[SegmentationSample], image_size: int = 256, mask_map: dict[int, int] | None = None, threshold: int | None = None, ignore_index: int | None = None) -> None:
        self.samples = samples
        self.image_size = image_size
        self.mask_map = mask_map
        self.threshold = threshold
        self.ignore_index = ignore_index

    def __len__(self) -> int:
        return len(self.samples)

    def _process_mask(self, mask: np.ndarray) -> np.ndarray:
        if self.threshold is not None:
            mask = (mask >= self.threshold).astype(np.int64)
        if self.mask_map:
            mapped = np.copy(mask)
            for src, dst in self.mask_map.items():
                mapped[mask == src] = dst
            mask = mapped
        return mask

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = Image.open(sample.image_path).convert("RGB")
        mask = Image.open(sample.mask_path).convert("L")

        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)

        image_tensor = F.to_tensor(image)
        image_tensor = F.normalize(image_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        mask_array = np.array(mask, dtype=np.int64)
        mask_array = self._process_mask(mask_array)
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


def _load_csv_list(csv_path: Path) -> list[str]:
    try:
        import pandas as pd
        table = pd.read_csv(csv_path, header=None)
        if table.shape[1] == 1:
            return table.iloc[:, 0].astype(str).tolist()
        for column in ["filename", "image", "image_path", "file"]:
            if column in table.columns:
                return table[column].astype(str).tolist()
    except Exception:
        pass
    lines = [line.strip() for line in csv_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if lines:
        return lines
    raise ValueError(f"Could not parse split file: {csv_path}")


def load_emps_samples(root: Path, split: str) -> list[SegmentationSample]:
    image_dir = root / "images"
    mask_dir = root / "segmaps"
    if split == "train":
        candidates = _load_csv_list(root / "train.csv") if (root / "train.csv").exists() else []
    elif split == "test":
        candidates = _load_csv_list(root / "test.csv") if (root / "test.csv").exists() else []
    else:
        candidates = []
    if not candidates:
        candidates = [path.name for path in sorted(image_dir.glob("*")) if path.is_file()]
    samples: list[SegmentationSample] = []
    for name in candidates:
        image_path = image_dir / f"{name}.png" if not (image_dir / name).exists() else image_dir / name
        mask_path = mask_dir / f"{name}.png" if not (mask_dir / name).exists() else mask_dir / name
        if image_path.exists() and mask_path.exists():
            samples.append(SegmentationSample(image_path=image_path, mask_path=mask_path, dataset_name="EMPS", split=split))
    return samples


def load_mask_pair_samples(root: Path, image_dir: str, mask_dir: str, split: str, dataset_name: str, pattern: str = "*", mask_suffix: str = "", mask_ext: str | None = None) -> list[SegmentationSample]:
    images_root = root / image_dir
    masks_root = root / mask_dir
    samples: list[SegmentationSample] = []
    for image_path in sorted(images_root.glob(pattern)):
        suffix = mask_ext if mask_ext else image_path.suffix
        mask_name = f"{image_path.stem}{mask_suffix}{suffix}"
        mask_path = masks_root / mask_name
        if mask_path.exists():
            samples.append(SegmentationSample(image_path=image_path, mask_path=mask_path, dataset_name=dataset_name, split=split))
    return samples


def load_pascal_samples(root: Path, split: str, dataset_name: str) -> list[SegmentationSample]:
    images_root = root / "JPEGImages"
    masks_root = root / "SegmentationClassRaw"
    split_file = root / "ImageSets" / f"{split}.txt"
    if not split_file.exists():
        raise FileNotFoundError(f"Missing split file: {split_file}")
    ids = [line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    samples: list[SegmentationSample] = []
    for sample_id in ids:
        image_path = images_root / f"{sample_id}.jpg"
        if not image_path.exists():
            image_path = images_root / f"{sample_id}.png"
        mask_path = masks_root / f"{sample_id}.png"
        if image_path.exists() and mask_path.exists():
            samples.append(SegmentationSample(image_path=image_path, mask_path=mask_path, dataset_name=dataset_name, split=split))
    return samples


def load_sem_dataset_from_registry(registry: dict, dataset_key: str, split: str) -> list[SegmentationSample]:
    entry = registry[dataset_key]
    root = Path(entry["root"])
    dataset_type = entry.get("type", "mask_pairs")
    if dataset_type == "nasa_ebc":
        return load_nasa_ebc_samples(root, entry["datasets"], split)
    if dataset_type == "emps":
        return load_emps_samples(root, split)
    if dataset_type == "pascal":
        return load_pascal_samples(root, split, entry.get("name", dataset_key))
    return load_mask_pair_samples(
        root,
        entry["image_dir"],
        entry["mask_dir"],
        split,
        entry.get("name", dataset_key),
        entry.get("pattern", "*"),
        entry.get("mask_suffix", ""),
        entry.get("mask_ext"),
    )


def split_labeled_unlabeled(samples: list[SegmentationSample], seed_size: int, rng: random.Random) -> tuple[list[SegmentationSample], list[SegmentationSample]]:
    indices = list(range(len(samples)))
    rng.shuffle(indices)
    labeled_idx = set(indices[:seed_size])
    labeled = [samples[i] for i in labeled_idx]
    unlabeled = [samples[i] for i in indices[seed_size:]]
    return labeled, unlabeled
