from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "configs" / "real_benchmark_targets.json"
    report_dir = project_root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    config = json.loads(config_path.read_text(encoding="utf-8"))
    benchmark_tracks = config["benchmark_tracks"]
    encoder_refs = config["encoder_references"]

    markdown_lines = [
        "# Real Microscopy Benchmark Map",
        "",
        "This report separates implemented microscopy benchmark evidence from documented public benchmark references.",
        "",
        "## Encoder References",
        "",
        "| Encoder or Corpus | Role | Why it matters | Source |",
        "|---|---|---|---|",
    ]

    for ref in encoder_refs:
        markdown_lines.append(
            f"| {ref['name']} | {ref['kind']} | {ref['why_it_matters']} | {ref['source_url']} |"
        )

    markdown_lines.extend(
        [
            "",
            "## Real Benchmark Targets",
            "",
            "| Modality | Task | Dataset | Status | Methodological purpose | Source |",
            "|---|---|---|---|---|---|",
        ]
    )

    for item in benchmark_tracks:
        markdown_lines.append(
            f"| {item['modality']} | {item['task']} | {item['dataset']} | {item['current_status']} | {item['method_purpose']} | {item['source_url']} |"
        )

    markdown_lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Current integrated evidence is SEM segmentation with generated masks, metric tables, and qualitative figures.",
            "- TEM and EBSD entries are documented as public benchmark references, not reported results.",
            "- The table separates implemented status from reference status for each modality.",
            "",
        ]
    )

    markdown_path = report_dir / "real_microscopy_benchmark_map.md"
    json_path = report_dir / "real_microscopy_benchmark_map.json"
    markdown_path.write_text("\n".join(markdown_lines), encoding="utf-8")
    json_path.write_text(json.dumps(config, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
