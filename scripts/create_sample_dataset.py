from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _draw_pattern(draw: ImageDraw.ImageDraw, image_size: int, class_name: str, rng: np.random.Generator) -> None:
    if class_name == "grain":
        for _ in range(25):
            x0 = int(rng.integers(0, image_size - 18)); y0 = int(rng.integers(0, image_size - 18))
            x1 = x0 + int(rng.integers(8, 22)); y1 = y0 + int(rng.integers(8, 22))
            shade = int(rng.integers(120, 230))
            draw.ellipse((x0, y0, x1, y1), outline=(shade, shade, shade), width=2)
    elif class_name == "pore":
        for _ in range(16):
            x = int(rng.integers(10, image_size - 10)); y = int(rng.integers(10, image_size - 10)); r = int(rng.integers(3, 10))
            draw.ellipse((x - r, y - r, x + r, y + r), fill=(15, 15, 15), outline=(80, 80, 80))
    else:
        for _ in range(14):
            x0 = int(rng.integers(0, image_size)); y0 = int(rng.integers(0, image_size)); x1 = int(rng.integers(0, image_size)); y1 = int(rng.integers(0, image_size))
            shade = int(rng.integers(140, 240))
            draw.line((x0, y0, x1, y1), fill=(shade, shade, shade), width=int(rng.integers(1, 3)))


def main() -> None:
    rng = np.random.default_rng(42)
    image_dir = REPO_ROOT / "data" / "raw" / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    classes = ["grain", "pore", "crack"]
    for group_id in range(12):
        class_name = classes[group_id % len(classes)]
        base_property = {"grain": 0.25, "pore": 0.62, "crack": 0.88}[class_name]
        magnification = int(rng.choice([200, 500, 1000]))
        for image_id in range(6):
            image = Image.new("RGB", (128, 128), (int(rng.integers(95, 140)),) * 3)
            draw = ImageDraw.Draw(image)
            _draw_pattern(draw, 128, class_name, rng)
            image = image.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.1, 0.9))))
            filename = f"specimen_{group_id:02d}_{image_id:02d}.png"
            image.save(image_dir / filename)
            rows.append({"image_path": filename, "target_class": class_name, "property_value": round(base_property + float(rng.normal(0, 0.035)), 4), "specimen_id": f"specimen_{group_id:02d}", "split_group": f"specimen_{group_id:02d}", "magnification": magnification, "stain": "none", "batch_id": f"batch_{group_id % 3}", "acquisition_date": "2026-03-09"})
    table = pd.DataFrame(rows)
    labels_path = REPO_ROOT / "data" / "processed" / "labels.csv"
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(labels_path, index=False)
    print(f"Wrote {len(table)} labeled images to {labels_path}")


if __name__ == "__main__":
    main()
