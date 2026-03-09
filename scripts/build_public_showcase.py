from __future__ import annotations

import json
import urllib.request
import zipfile
from io import BytesIO
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageOps, UnidentifiedImageError

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from microscopy_cv_research.data.datasets import MicroscopyImageDataset

ALLOWED_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def load_sources() -> list[dict]:
    return json.loads((REPO_ROOT / "configs" / "public_showcase_sources.json").read_text(encoding="utf-8"))


def download_archive(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        urllib.request.urlretrieve(url, destination)


def pick_members(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    members = [info for info in archive.infolist() if (not info.is_dir()) and info.filename.lower().endswith(ALLOWED_SUFFIXES)]
    members.sort(key=lambda item: item.filename)
    return members


def normalize_image(image: Image.Image) -> Image.Image:
    if getattr(image, "is_animated", False):
        image.seek(0)
    return ImageOps.autocontrast(image.convert("L")).convert("RGB")


def save_preview_from_zip(archive: zipfile.ZipFile, member: zipfile.ZipInfo, output_path: Path) -> dict:
    with archive.open(member) as handle:
        raw = handle.read()
    with Image.open(BytesIO(raw)) as image:
        normalized = normalize_image(image)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        normalized.save(output_path)
        return {
            "archive_member": member.filename,
            "preview_path": str(output_path),
            "width": normalized.width,
            "height": normalized.height,
            "mode": normalized.mode,
        }


def save_montage(dataset_name: str, preview_paths: list[Path], output_path: Path) -> None:
    columns = min(2, len(preview_paths))
    rows = max((len(preview_paths) + columns - 1) // columns, 1)
    fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 4 * rows))
    flat_axes = list(getattr(axes, "flat", [axes]))
    for axis, path in zip(flat_axes, preview_paths, strict=False):
        with Image.open(path) as image:
            axis.imshow(image)
        axis.set_title(path.stem, fontsize=9)
        axis.axis("off")
    for axis in flat_axes[len(preview_paths):]:
        axis.axis("off")
    fig.suptitle(f"{dataset_name} public microscopy samples")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def validate_dataset_loader(preview_dir: Path, preview_files: list[Path]) -> dict:
    manifest = pd.DataFrame({"image_path": [path.name for path in preview_files], "target_class": ["showcase"] * len(preview_files)})
    dataset = MicroscopyImageDataset(manifest, preview_dir, target_column="target_class")
    image, target = dataset[0]
    return {"num_samples": len(dataset), "first_image_size": list(image.size), "first_target": target}


def build_report(records: list[dict], output_path: Path) -> None:
    lines = ["# Public Microscopy Showcase", "", "This report was generated from official public microscopy-image sources.", ""]
    for record in records:
        lines.extend([
            f"## {record['title']}",
            "",
            f"- Project page: {record['project_page']}",
            f"- Download URL: {record['download_url']}",
            f"- License note: {record['license_note']}",
            f"- Archive size (MB): {record['archive_size_mb']}",
            f"- Extracted previews: {record['num_previews']}",
            f"- Loader smoke test: {record['loader_validation']}",
            f"- Montage: `{record['montage_path']}`",
            "",
        ])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    downloads_dir = REPO_ROOT / "data" / "external" / "showcase" / "downloads"
    previews_root = REPO_ROOT / "data" / "external" / "showcase" / "previews"
    figures_root = REPO_ROOT / "reports" / "figures" / "showcase"
    manifest_path = REPO_ROOT / "reports" / "public_showcase_manifest.json"
    report_path = REPO_ROOT / "reports" / "public_showcase.md"

    records = []
    for source in load_sources():
        archive_path = downloads_dir / f"{source['name']}.zip"
        download_archive(source["download_url"], archive_path)
        with zipfile.ZipFile(archive_path) as archive:
            preview_dir = previews_root / source["name"]
            preview_dir.mkdir(parents=True, exist_ok=True)
            preview_files = []
            preview_records = []
            preview_index = 1
            for member in pick_members(archive):
                if len(preview_files) >= source["sample_limit"]:
                    break
                preview_path = preview_dir / f"{source['name']}_{preview_index:02d}.png"
                try:
                    preview_records.append(save_preview_from_zip(archive, member, preview_path))
                    preview_files.append(preview_path)
                    preview_index += 1
                except (UnidentifiedImageError, OSError, ValueError):
                    continue

        if not preview_files:
            records.append({
                **source,
                "archive_path": str(archive_path),
                "archive_size_mb": round(archive_path.stat().st_size / (1024 * 1024), 2),
                "num_previews": 0,
                "preview_records": [],
                "loader_validation": "no readable preview images found",
                "montage_path": "not generated",
            })
            continue

        montage_path = figures_root / f"{source['name']}_montage.png"
        save_montage(source["name"], preview_files, montage_path)
        records.append({
            **source,
            "archive_path": str(archive_path),
            "archive_size_mb": round(archive_path.stat().st_size / (1024 * 1024), 2),
            "num_previews": len(preview_files),
            "preview_records": preview_records,
            "loader_validation": validate_dataset_loader(preview_dir, preview_files),
            "montage_path": str(montage_path),
        })

    manifest_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    build_report(records, report_path)
    print(json.dumps({"manifest_path": str(manifest_path), "report_path": str(report_path), "datasets": len(records)}, indent=2))


if __name__ == "__main__":
    main()
