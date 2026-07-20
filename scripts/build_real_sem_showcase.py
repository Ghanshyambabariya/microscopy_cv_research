from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


DATASETS = ["EBC1", "EBC2", "EBC3", "Super1", "Super2", "Super3", "Super4"]
SPLITS = ["train", "val", "test"]


def annotation_path(dataset_dir: Path, split: str, image_path: Path) -> Path:
    annot_dir = dataset_dir / f"{split}_annot"
    direct = annot_dir / image_path.name
    if direct.exists():
        return direct
    stem = image_path.stem
    suffix = image_path.suffix
    mask_name = f"{stem}_mask{suffix}"
    candidate = annot_dir / mask_name
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"No annotation found for {image_path}")


def collect_summary(root: Path) -> list[dict]:
    records: list[dict] = []
    for dataset_name in DATASETS:
        dataset_dir = root / dataset_name
        split_counts = {}
        for split in SPLITS:
            images = sorted((dataset_dir / split).glob("*.tif"))
            masks = [annotation_path(dataset_dir, split, image) for image in images]
            split_counts[split] = {
                "images": len(images),
                "masks": len(masks),
            }
        records.append({
            "dataset": dataset_name,
            "modality": "SEM",
            "task": "semantic segmentation",
            "split_counts": split_counts,
        })
    return records


def save_portfolio_figure(root: Path, output_path: Path) -> None:
    sample_specs = [
        ("EBC1", "test"),
        ("Super1", "test"),
    ]

    fig, axes = plt.subplots(len(sample_specs), 3, figsize=(10, 6.5))
    if len(sample_specs) == 1:
        axes = [axes]

    for row_axes, (dataset_name, split) in zip(axes, sample_specs, strict=False):
        dataset_dir = root / dataset_name
        image_path = sorted((dataset_dir / split).glob("*.tif"))[0]
        mask_path = annotation_path(dataset_dir, split, image_path)

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        row_axes[0].imshow(image)
        row_axes[0].set_title(f"{dataset_name} input", fontsize=10)
        row_axes[0].axis("off")

        row_axes[1].imshow(mask, cmap="viridis")
        row_axes[1].set_title(f"{dataset_name} mask", fontsize=10)
        row_axes[1].axis("off")

        row_axes[2].imshow(image)
        row_axes[2].imshow(mask, cmap="magma", alpha=0.35)
        row_axes[2].set_title(f"{dataset_name} overlay", fontsize=10)
        row_axes[2].axis("off")

    fig.suptitle("NASA MicroNet SEM benchmark samples", fontsize=14)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_markdown(summary: list[dict], figure_path: Path, output_path: Path) -> None:
    lines = [
        "# Real SEM Benchmark Showcase",
        "",
        "This report is built from NASA's `pretrained-microscopy-models` benchmark segmentation data bundled under the MIT license in that repository.",
        "",
        "## What Is Integrated",
        "",
        "- Modality: SEM-style materials microscopy",
        "- Task: semantic segmentation",
        "- Source: NASA MicroNet benchmark segmentation data",
        "- Evidence in this repo: real benchmark sample inputs, masks, overlays, and split counts",
        "",
        "## Split Summary",
        "",
        "| Dataset | Train pairs | Val pairs | Test pairs |",
        "|---|---|---|---|",
    ]

    for record in summary:
        split_counts = record["split_counts"]
        lines.append(
            f"| {record['dataset']} | {split_counts['train']['images']} | {split_counts['val']['images']} | {split_counts['test']['images']} |"
        )

    lines.extend(
        [
            "",
            "## Result Figure",
            "",
            f"![NASA SEM benchmark]({figure_path.as_posix()})",
            "",
            "## Interpretation",
            "",
            "- This is the first real microscopy benchmark integrated directly into the project structure.",
            "- It is segmentation-focused, so it complements the repo's existing classification and regression starter tracks.",
            "- TEM and EBSD are still benchmark targets rather than fully integrated evaluated tasks.",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    root = project_root / "data" / "external" / "nasa_microscopy_models" / "benchmark_segmentation_data"
    summary = collect_summary(root)

    reports_dir = project_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    figure_path = project_root / "reports" / "figures" / "real_sem_benchmark.png"
    save_portfolio_figure(root, figure_path)

    summary_json = reports_dir / "real_sem_benchmark_summary.json"
    summary_md = reports_dir / "real_sem_benchmark_showcase.md"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown(summary, Path("reports/figures/real_sem_benchmark.png"), summary_md)

    print(json.dumps({
        "summary_json": str(summary_json),
        "summary_md": str(summary_md),
        "figure": str(figure_path),
        "datasets": len(summary),
    }, indent=2))


if __name__ == "__main__":
    main()
