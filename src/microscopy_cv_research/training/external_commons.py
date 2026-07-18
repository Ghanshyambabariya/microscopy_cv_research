from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.request import urlretrieve
from zipfile import ZipFile

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from microscopy_cv_research.config import load_config
from microscopy_cv_research.training.engine import save_json


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def maybe_download_commons(config: dict[str, Any], allow_large_download: bool = False) -> Path | None:
    archive_path = Path(config["raw_archive_path"])
    if archive_path.exists() and archive_path.stat().st_size > 0:
        return archive_path
    if not allow_large_download:
        return None
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(config["sampled_zip_url"], archive_path)
    return archive_path


def maybe_extract_commons(config: dict[str, Any], archive_path: Path | None) -> Path | None:
    extract_dir = Path(config["extract_dir"])
    if extract_dir.exists() and any(extract_dir.rglob("*")):
        return extract_dir
    if archive_path is None:
        return None
    extract_dir.mkdir(parents=True, exist_ok=True)
    with ZipFile(archive_path) as zf:
        zf.extractall(extract_dir)
    return extract_dir


def infer_label_from_path(path: Path, root: Path) -> str:
    relative = path.relative_to(root)
    if len(relative.parts) > 1:
        return relative.parts[0]
    stem_parts = path.stem.replace("-", "_").split("_")
    return stem_parts[0] if stem_parts else "unknown"


def build_manifest(extract_dir: Path, config: dict[str, Any]) -> pd.DataFrame:
    image_paths = [path for path in extract_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS]
    image_paths = sorted(image_paths)[: int(config.get("max_images", 500))]
    rows = [{"image_path": str(path), "label": infer_label_from_path(path, extract_dir)} for path in image_paths]
    if not rows:
        raise ValueError(f"No images found in {extract_dir}")
    manifest = pd.DataFrame(rows)
    manifest_path = Path(config["manifest_path"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)
    return manifest


def image_descriptor(path: str | Path, image_size: int) -> list[float]:
    image = Image.open(path).convert("L").resize((image_size, image_size))
    pixels = pd.Series(list(image.getdata()), dtype="float64") / 255.0
    return [
        float(pixels.mean()),
        float(pixels.std()),
        float(pixels.quantile(0.1)),
        float(pixels.quantile(0.5)),
        float(pixels.quantile(0.9)),
    ]


def train_commons_baseline(manifest: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    labels = manifest["label"].astype(str)
    if labels.nunique() < 2:
        raise ValueError("Need at least two inferred classes to train CoMMonS baseline.")
    descriptors = manifest["image_path"].apply(lambda p: image_descriptor(p, int(config.get("image_size", 96))))
    X = pd.DataFrame(descriptors.tolist(), columns=["mean", "std", "p10", "p50", "p90"])
    y = labels
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=float(config.get("test_size", 0.25)), random_state=int(config.get("seed", 42)), stratify=y if y.value_counts().min() > 1 else None)
    model = RandomForestClassifier(n_estimators=200, random_state=int(config.get("seed", 42)), class_weight="balanced")
    model.fit(train_x, train_y)
    pred = model.predict(test_x)
    return {
        "splits": {"train": int(len(train_x)), "test": int(len(test_x))},
        "classification_metrics": {"accuracy": float(accuracy_score(test_y, pred)), "macro_f1": float(f1_score(test_y, pred, average="macro"))},
        "feature_names": X.columns.tolist(),
    }


def write_commons_report(report: dict[str, Any], config: dict[str, Any]) -> None:
    lines = [
        "# External CoMMonS Microscopy Target",
        "",
        f"- source: [{config['source_name']}]({config['source_repo']})",
        f"- sampled archive size: about `{config['expected_archive_size_gb']} GB`",
        f"- status: `{report['status']}`",
        "",
        "## Current Handling",
        "",
        report["message"],
        "",
        "## Runnable Command",
        "",
        "`python scripts/run_external_commons.py --config configs/external_commons_microscopy.json --allow-large-download`",
    ]
    if "classification_metrics" in report:
        lines.extend([
            "",
            "## Results",
            "",
            f"- accuracy: `{report['classification_metrics']['accuracy']:.4f}`",
            f"- macro F1: `{report['classification_metrics']['macro_f1']:.4f}`",
        ])
    Path(config["markdown_report_path"]).write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_external_commons_experiment(config_path: str | Path, allow_large_download: bool = False) -> dict[str, Any]:
    config = load_config(config_path)
    archive = maybe_download_commons(config, allow_large_download=allow_large_download)
    extract_dir = maybe_extract_commons(config, archive)
    if extract_dir is None:
        report = {
            "experiment_name": config["experiment_name"],
            "source_name": config["source_name"],
            "source_repo": config["source_repo"],
            "status": "large_dataset_not_downloaded",
            "message": "CoMMonS is configured as an optional large microscopy-material benchmark. The sampled archive is about 1.1 GB, so the default run documents the target without downloading or committing the archive.",
        }
    else:
        manifest = build_manifest(extract_dir, config)
        report = {
            "experiment_name": config["experiment_name"],
            "source_name": config["source_name"],
            "source_repo": config["source_repo"],
            "status": "trained" if manifest["label"].nunique() >= 2 else "manifest_only",
            "message": f"Manifest built with {len(manifest)} images and {manifest['label'].nunique()} inferred labels.",
            "manifest_path": config["manifest_path"],
            "num_images": int(len(manifest)),
            "num_labels": int(manifest["label"].nunique()),
        }
        if manifest["label"].nunique() >= 2:
            report.update(train_commons_baseline(manifest, config))
    save_json(report, config["report_path"])
    write_commons_report(report, config)
    return report

