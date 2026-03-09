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

IMAGE_SIZE = 128
SPECIMENS_PER_CLASS = 60
IMAGES_PER_SPECIMEN = 8
CLASSES = ["grain", "pore", "crack"]
BASE_PROPERTY = {"grain": 0.22, "pore": 0.58, "crack": 0.9}


def _background_texture(image: Image.Image, rng: np.random.Generator) -> None:
    draw = ImageDraw.Draw(image)
    for _ in range(160):
        x = int(rng.integers(0, IMAGE_SIZE))
        y = int(rng.integers(0, IMAGE_SIZE))
        shade = int(rng.integers(92, 138))
        draw.point((x, y), fill=(shade, shade, shade))


def _draw_grain(draw: ImageDraw.ImageDraw, rng: np.random.Generator) -> None:
    for _ in range(26):
        x0 = int(rng.integers(0, IMAGE_SIZE - 22))
        y0 = int(rng.integers(0, IMAGE_SIZE - 22))
        w = int(rng.integers(10, 24))
        h = int(rng.integers(10, 24))
        shade = int(rng.integers(170, 235))
        draw.ellipse((x0, y0, x0 + w, y0 + h), outline=(shade, shade, shade), fill=(shade - 18, shade - 18, shade - 18), width=2)
    for _ in range(18):
        x0 = int(rng.integers(0, IMAGE_SIZE - 18))
        y0 = int(rng.integers(0, IMAGE_SIZE - 18))
        x1 = x0 + int(rng.integers(6, 18))
        y1 = y0 + int(rng.integers(6, 18))
        draw.rectangle((x0, y0, x1, y1), outline=(150, 150, 150), width=1)


def _draw_pore(draw: ImageDraw.ImageDraw, rng: np.random.Generator) -> None:
    for _ in range(22):
        x = int(rng.integers(10, IMAGE_SIZE - 10))
        y = int(rng.integers(10, IMAGE_SIZE - 10))
        r = int(rng.integers(4, 12))
        draw.ellipse((x - r, y - r, x + r, y + r), fill=(12, 12, 12), outline=(88, 88, 88), width=2)
    for _ in range(10):
        x = int(rng.integers(10, IMAGE_SIZE - 10))
        y = int(rng.integers(10, IMAGE_SIZE - 10))
        r = int(rng.integers(2, 5))
        draw.ellipse((x - r, y - r, x + r, y + r), fill=(35, 35, 35))


def _draw_crack(draw: ImageDraw.ImageDraw, rng: np.random.Generator) -> None:
    for _ in range(12):
        x0 = int(rng.integers(0, IMAGE_SIZE))
        y0 = int(rng.integers(0, IMAGE_SIZE))
        x1 = int(np.clip(x0 + rng.integers(-50, 50), 0, IMAGE_SIZE - 1))
        y1 = int(np.clip(y0 + rng.integers(-50, 50), 0, IMAGE_SIZE - 1))
        draw.line((x0, y0, x1, y1), fill=(235, 235, 235), width=int(rng.integers(2, 4)))
    for _ in range(20):
        x0 = int(rng.integers(0, IMAGE_SIZE))
        y0 = int(rng.integers(0, IMAGE_SIZE))
        x1 = int(np.clip(x0 + rng.integers(-16, 16), 0, IMAGE_SIZE - 1))
        y1 = int(np.clip(y0 + rng.integers(-16, 16), 0, IMAGE_SIZE - 1))
        draw.line((x0, y0, x1, y1), fill=(170, 170, 170), width=1)


def _render_image(class_name: str, rng: np.random.Generator) -> Image.Image:
    base = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (int(rng.integers(96, 132)),) * 3)
    _background_texture(base, rng)
    draw = ImageDraw.Draw(base)
    if class_name == "grain":
        _draw_grain(draw, rng)
    elif class_name == "pore":
        _draw_pore(draw, rng)
    else:
        _draw_crack(draw, rng)
    image = base.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.15, 0.7))))
    return image


def main() -> None:
    rng = np.random.default_rng(42)
    image_dir = REPO_ROOT / "data" / "raw" / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    for old_file in image_dir.glob('*.png'):
        old_file.unlink()

    rows = []
    specimen_counter = 0
    for class_name in CLASSES:
        for _ in range(SPECIMENS_PER_CLASS):
            specimen_id = f"specimen_{specimen_counter:03d}"
            specimen_counter += 1
            magnification = int(rng.choice([200, 500, 1000]))
            batch_id = f"batch_{specimen_counter % 6}"
            specimen_offset = float(rng.normal(0, 0.02))
            for image_id in range(IMAGES_PER_SPECIMEN):
                image = _render_image(class_name, rng)
                filename = f"{specimen_id}_{image_id:02d}.png"
                image.save(image_dir / filename)
                rows.append(
                    {
                        "image_path": filename,
                        "target_class": class_name,
                        "property_value": round(BASE_PROPERTY[class_name] + specimen_offset + float(rng.normal(0, 0.018)), 4),
                        "specimen_id": specimen_id,
                        "split_group": specimen_id,
                        "magnification": magnification,
                        "stain": "none",
                        "batch_id": batch_id,
                        "acquisition_date": "2026-03-09",
                    }
                )

    table = pd.DataFrame(rows)
    labels_path = REPO_ROOT / "data" / "processed" / "labels.csv"
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(labels_path, index=False)
    print(f"Wrote {len(table)} labeled images across {SPECIMENS_PER_CLASS * len(CLASSES)} specimens to {labels_path}")


if __name__ == "__main__":
    main()
