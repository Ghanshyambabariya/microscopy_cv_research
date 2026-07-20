from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def metric(value: float) -> str:
    return f"{value:.4f}"


def main() -> None:
    signal = load_json(REPO_ROOT / "reports" / "materials_signal_metrics.json")
    multimodal = load_json(REPO_ROOT / "reports" / "multimodal_materials_metrics.json")
    vicomtech = load_json(REPO_ROOT / "reports" / "external_tool_wear_vicomtech_metrics.json")
    uniwear = load_json(REPO_ROOT / "reports" / "external_uniwear_tool_wear_metrics.json")
    lines = [
        "# Materials AI Platform Report",
        "",
        "This report connects microscopy computer vision, high-frequency materials process signals, and multimodal ML for structure-process-property learning.",
        "",
        "## Signal Intelligence",
        "",
        f"- task: {signal['task']}",
        f"- sampling rate: `{signal['sampling_rate_hz']} Hz`",
        f"- nominal process duration: `{signal['nominal_process_seconds']} s`",
        f"- fast analysis window: `{signal['analysis_window_seconds']} s`",
        f"- feature table: `{signal['feature_table_path']}`",
        f"- process-quality accuracy: `{metric(signal['quality_metrics']['accuracy'])}`",
        f"- process-quality macro F1: `{metric(signal['quality_metrics']['macro_f1'])}`",
        f"- property regression MAE: `{metric(signal['property_metrics']['mae'])}`",
        f"- property regression R2: `{metric(signal['property_metrics']['r2'])}`",
        "",
        "![Materials signal summary](figures/materials_signal_summary.png)",
        "",
        "## Multimodal Fusion",
        "",
        f"- task: {multimodal['task']}",
        f"- multimodal table: `{multimodal['multimodal_table_path']}`",
        f"- train/test split: `{multimodal['splits']['train']}` / `{multimodal['splits']['test']}` specimens",
        f"- signal features: `{multimodal['num_signal_features']}`",
        f"- microscopy features: `{multimodal['num_microscopy_features']}`",
        f"- process-quality accuracy: `{metric(multimodal['quality_metrics']['accuracy'])}`",
        f"- process-quality macro F1: `{metric(multimodal['quality_metrics']['macro_f1'])}`",
        f"- property regression MAE: `{metric(multimodal['property_metrics']['mae'])}`",
        f"- property regression R2: `{metric(multimodal['property_metrics']['r2'])}`",
        "",
        "## Real Online Benchmarks",
        "",
        "| Dataset | Source | Task | Held-out split | Main result |",
        "|---|---|---|---|---|",
        f"| Vicomtech tool wear | `{vicomtech['source_repo']}` | flank-wear regression + wear-stage classification | tool IDs | R2 `{metric(vicomtech['flank_wear_regression']['r2'])}`, macro F1 `{metric(vicomtech['wear_stage_classification']['macro_f1'])}` |",
        f"| Katulu Uniwear | `{uniwear['source_repo']}` | force/vibration window wear prediction | experiment tags | R2 `{metric(uniwear['tool_wear_regression']['r2'])}`, macro F1 `{metric(uniwear['wear_stage_classification']['macro_f1'])}` |",
        "",
        "![External tool wear benchmark](figures/external_tool_wear_vicomtech.png)",
        "",
        "![External Uniwear benchmark](figures/external_uniwear_tool_wear.png)",
        "",
        "## Large Microscopy Target",
        "",
        "CoMMonS is a strong next microscopy-material dataset target: it contains 6,912 microscopic fabric-surface images across 24 samples and expert-rated fabric properties. The sampled archive is about 1.1 GB, so it is documented as a large-data target rather than committed directly into this lightweight repository.",
        "",
        "## Interpretation",
        "",
        "The current signal data is simulated but physics-inspired: force level, chatter-band energy, torque, bursts, and impulse behavior are linked to material class and property values. This gives a working ML scaffold that can be replaced with real grinding, milling, acoustic-emission, vibration, force, torque, spindle-current, or temperature CSV files.",
        "",
        "The project value is the full structure: microscopy CV, high-frequency signal features, supervised ML, regression, multimodal fusion, and generated reports.",
    ]
    out_path = REPO_ROOT / "reports" / "materials_ai_platform_report.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()
