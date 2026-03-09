from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter

from microscopy_cv_research.config import load_config
from microscopy_cv_research.evaluation.reporting import save_report
from microscopy_cv_research.models.encoder_registry import get_encoder_spec


def _draw_micro_pattern(draw: ImageDraw.ImageDraw, image_size: int, class_name: str, rng: np.random.Generator) -> None:
    if class_name == "grain":
        for _ in range(18):
            x0 = int(rng.integers(0, image_size - 20)); y0 = int(rng.integers(0, image_size - 20))
            x1 = x0 + int(rng.integers(8, 26)); y1 = y0 + int(rng.integers(8, 26))
            shade = int(rng.integers(110, 220))
            draw.ellipse((x0, y0, x1, y1), outline=(shade, shade, shade), width=2)
    elif class_name == "pore":
        for _ in range(14):
            x = int(rng.integers(10, image_size - 10)); y = int(rng.integers(10, image_size - 10)); r = int(rng.integers(3, 9))
            draw.ellipse((x - r, y - r, x + r, y + r), fill=(20, 20, 20), outline=(80, 80, 80))
    else:
        for _ in range(12):
            x0 = int(rng.integers(0, image_size)); y0 = int(rng.integers(0, image_size)); x1 = int(rng.integers(0, image_size)); y1 = int(rng.integers(0, image_size))
            shade = int(rng.integers(130, 230))
            draw.line((x0, y0, x1, y1), fill=(shade, shade, shade), width=int(rng.integers(1, 3)))


def plan_synthetic_study(config_path: str | Path) -> dict:
    config = load_config(config_path)
    return {
        "experiment_name": config["experiment_name"],
        "generator_family": config["generator_family"],
        "conditioning_mode": config["conditioning_mode"],
        "evaluation_protocol": ["feature-distance comparison to real images", "class-balance improvement", "downstream supervised gain", "failure-case audit"],
    }


def generate_synthetic_images(config_path: str | Path) -> dict:
    config = load_config(config_path)
    rng = np.random.default_rng(config["seed"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    classes = ["grain", "pore", "crack"]
    rows = []
    for class_index, class_name in enumerate(classes):
        for sample_index in range(config["samples_per_class"]):
            base = Image.new("RGB", (config["image_size"], config["image_size"]), (int(rng.integers(90, 140)),) * 3)
            draw = ImageDraw.Draw(base)
            _draw_micro_pattern(draw, config["image_size"], class_name, rng)
            image = base.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.2, 1.0))))
            filename = f"synthetic_{class_name}_{sample_index:03d}.png"
            image.save(output_dir / filename)
            rows.append({"image_path": filename, "target_class": class_name, "property_value": round(0.2 + class_index * 0.35 + float(rng.normal(0, 0.03)), 4), "specimen_id": f"synthetic_{class_name}_{sample_index // 3}", "magnification": int(rng.choice([200, 500, 1000])), "batch_id": f"syn_batch_{class_index}", "source": "synthetic"})
    table = pd.DataFrame(rows)
    table_path = output_dir.parent / "synthetic_labels.csv"
    table.to_csv(table_path, index=False)
    report = {
        "experiment_name": config["experiment_name"],
        "generator_family": config["generator_family"],
        "conditioning_mode": config["conditioning_mode"],
        "encoder_reference": asdict(get_encoder_spec("micronet")),
        "num_generated_images": len(table),
        "class_distribution": table["target_class"].value_counts().sort_index().to_dict(),
        "table_path": str(table_path),
        "image_dir": str(output_dir),
        "evaluation_protocol": ["feature-distance comparison to real images", "class-balance improvement", "downstream supervised gain", "expert visual audit"],
    }
    save_report(report, config["report_path"])
    return report
