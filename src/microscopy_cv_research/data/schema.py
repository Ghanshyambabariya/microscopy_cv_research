from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class SampleRecord:
    image_path: Path
    split_group: str
    metadata: dict[str, str | float | int]
    targets: dict[str, str | float | int]


REQUIRED_COLUMNS = ["image_path", "specimen_id"]
OPTIONAL_COLUMNS = ["target_class", "property_value", "magnification", "stain", "batch_id"]
