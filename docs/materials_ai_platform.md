# Materials AI Platform Direction

MicroForge AI is evolving from a microscopy-only computer-vision repository into a broader materials-AI portfolio system.

## Core Research Question

Can microscopy images and high-frequency process signals be fused to predict material quality, microstructure class, and structure-property behavior?

## Modules

| Module | Data | Task | Job/research signal |
|---|---|---|---|
| Microscopy CV | SEM/TEM/EBSD-style images | segmentation, classification, regression | computer vision for characterization |
| Process Signal ML | `Fx`, `Fy`, `Fz`, `Mz` force/torque signals | quality classification, property regression | manufacturing analytics and time-series ML |
| Multimodal Fusion | microscopy specimen descriptors + signal features | structure-process-property prediction | materials informatics |
| Active Learning | uncertain microscopy masks | annotation prioritization | low-data scientific ML |

## Current Implementation

The signal module simulates grinding-style high-frequency data with a nominal 20 kHz sampling rate and 20 second process target. The default config analyzes a 1 second window for fast local runs while preserving the full-process metadata.

The external data module downloads the Vicomtech machine-tool-wear dataset directly from GitHub, validates the schema, cleans numeric sensor features, derives wear-stage labels from flank wear, splits by tool ID, and trains both regression and classification baselines.

The Uniwear module downloads a second GitHub-hosted multi-material tool-wear dataset, windows force/vibration time-series rows by experiment, extracts signal features, excludes timestamp leakage from model features, and evaluates wear prediction on held-out experiment tags.

The concrete-strength module adds a compact materials-informatics benchmark: mix composition and curing age are used to predict compressive strength.

The CoMMonS module adds the optional microscopic material-surface image path. Because the sampled archive is about 1.1 GB, the default command documents the target and the large-download command prepares the manifest/training flow only when explicitly requested.

Generated artifacts:

- `data/processed/materials_signal_features.csv`
- `data/processed/materials_multimodal_table.csv`
- `data/processed/vicomtech_tool_wear_clean.csv`
- `data/processed/uniwear_tool_wear_clean.csv`
- `data/processed/uniwear_window_features.csv`
- `data/processed/concrete_strength_clean.csv`
- `reports/materials_signal_metrics.json`
- `reports/multimodal_materials_metrics.json`
- `reports/external_tool_wear_vicomtech_metrics.json`
- `reports/external_tool_wear_vicomtech_report.md`
- `reports/external_uniwear_tool_wear_metrics.json`
- `reports/external_uniwear_tool_wear_report.md`
- `reports/external_concrete_strength_metrics.json`
- `reports/external_concrete_strength_report.md`
- `reports/external_commons_microscopy_report.md`
- `reports/materials_ai_leaderboard.md`
- `docs/datasets.md`
- `reports/figures/materials_signal_summary.png`
- `reports/figures/external_tool_wear_vicomtech.png`
- `reports/figures/external_uniwear_tool_wear.png`
- `reports/figures/external_concrete_strength.png`

## How To Run

```powershell
python scripts/run_materials_signal.py --config configs/materials_signal.json
python scripts/run_multimodal_materials.py --config configs/multimodal_materials.json
python scripts/run_external_tool_wear.py --config configs/external_tool_wear_vicomtech.json
python scripts/run_external_uniwear.py --config configs/external_uniwear_tool_wear.json
python scripts/run_external_concrete.py --config configs/external_concrete_strength.json
python scripts/run_external_commons.py --config configs/external_commons_microscopy.json
python scripts/build_materials_ai_leaderboard.py
python scripts/run_all_benchmarks.py
```

## Next Research Upgrade

Replace simulated signals with measured grinding, milling, acoustic-emission, vibration, spindle-current, or temperature CSV files. The feature extraction layer already expects channels named `Fx`, `Fy`, `Fz`, and `Mz`, so real data can be inserted without changing the ML report structure.
