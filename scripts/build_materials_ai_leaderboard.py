from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_json(path: str | Path) -> dict:
    return json.loads((REPO_ROOT / path).read_text(encoding="utf-8"))


def maybe_metric(path: str, metric_path: list[str], default: str = "-") -> str:
    try:
        data = load_json(path)
        value = data
        for key in metric_path:
            value = value[key]
        return f"{float(value):.4f}"
    except Exception:
        return default


def write_leaderboard() -> None:
    rows = [
        ["NASA/SEM suite", "microscopy", "segmentation", "dataset split", "UNetSmall", "mean IoU fg", maybe_metric("reports/sem_suite_nasa_ebc.json", ["test_metrics", "mean_iou_fg"]), "reports/sem_leaderboard.md"],
        ["Vicomtech tool wear", "process sensors", "flank-wear regression", "held-out tool IDs", "RandomForest", "R2", maybe_metric("reports/external_tool_wear_vicomtech_metrics.json", ["flank_wear_regression", "r2"]), "reports/external_tool_wear_vicomtech_report.md"],
        ["Vicomtech tool wear", "process sensors", "wear-stage classification", "held-out tool IDs", "RandomForest", "macro F1", maybe_metric("reports/external_tool_wear_vicomtech_metrics.json", ["wear_stage_classification", "macro_f1"]), "reports/external_tool_wear_vicomtech_report.md"],
        ["Katulu Uniwear", "force/vibration", "tool-wear regression", "held-out experiment tags", "RandomForest", "R2", maybe_metric("reports/external_uniwear_tool_wear_metrics.json", ["tool_wear_regression", "r2"]), "reports/external_uniwear_tool_wear_report.md"],
        ["Katulu Uniwear", "force/vibration", "wear-stage classification", "held-out experiment tags", "RandomForest", "macro F1", maybe_metric("reports/external_uniwear_tool_wear_metrics.json", ["wear_stage_classification", "macro_f1"]), "reports/external_uniwear_tool_wear_report.md"],
        ["Concrete strength", "materials tabular", "compressive-strength regression", "random split", "RandomForest", "R2", maybe_metric("reports/external_concrete_strength_metrics.json", ["regression_metrics", "r2"]), "reports/external_concrete_strength_report.md"],
        ["CoMMonS", "microscopy material surface", "optional image classification", "large-data target", "RandomForest descriptors", "status", "target", "reports/external_commons_microscopy_report.md"],
    ]
    lines = [
        "# Materials AI Benchmark Leaderboard",
        "",
        "| Dataset | Modality | Task | Split | Model | Metric | Result | Report |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    (REPO_ROOT / "reports" / "materials_ai_leaderboard.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_dataset_cards() -> None:
    registry = load_json("configs/online_dataset_registry.json")
    lines = [
        "# Dataset Cards",
        "",
        "These cards summarize the online datasets currently connected to the separate materials-ML projects.",
        "",
    ]
    for section in ["compact_runnable", "large_targets"]:
        title = "Runnable Datasets" if section == "compact_runnable" else "Large Optional Targets"
        lines.extend([f"## {title}", ""])
        for item in registry.get(section, []):
            lines.extend([
                f"### {item['name']}",
                "",
                f"- type: `{item['type']}`",
                f"- source: {item['source_repo']}",
                f"- status: `{item['status']}`",
            ])
            if "runner" in item:
                lines.append(f"- runner: `{item['runner']}`")
            if "report" in item:
                lines.append(f"- report: `{item['report']}`")
            if "reason" in item:
                lines.append(f"- reason: {item['reason']}")
            lines.append("")
    (REPO_ROOT / "docs" / "datasets.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    write_leaderboard()
    write_dataset_cards()
    print(REPO_ROOT / "reports" / "materials_ai_leaderboard.md")
    print(REPO_ROOT / "docs" / "datasets.md")


if __name__ == "__main__":
    main()
