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
        "# Real Microscopy Benchmark Notes",
        "",
        "This report separates the current synthetic starter benchmark from the real microscopy targets the project should support next.",
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
            "| Modality | Task | Dataset | Status | Project goal | Source |",
            "|---|---|---|---|---|---|",
        ]
    )

    for item in benchmark_tracks:
        markdown_lines.append(
            f"| {item['modality']} | {item['task']} | {item['dataset']} | {item['current_status']} | {item['portfolio_goal']} | {item['source_url']} |"
        )

    markdown_lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The current repo is a valid framework demonstration, but it is not yet comparable to NASA-style microscopy transfer-learning benchmarks.",
            "- The next scientific step is to ingest at least one real SEM task, one TEM task, and one EBSD task and report task-appropriate metrics.",
            "- Real evidence should include actual test images, predicted outputs, metric tables, and failure-case visualizations.",
            "",
        ]
    )

    markdown_path = report_dir / "real_benchmark_portfolio.md"
    json_path = report_dir / "real_benchmark_portfolio.json"
    markdown_path.write_text("\n".join(markdown_lines), encoding="utf-8")
    json_path.write_text(json.dumps(config, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
