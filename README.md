<p align="center">
  <img src="assets/matsci_ai_portfolio.svg" alt="MicroForge materials ML banner" width="100%">
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB">
  <img alt="ML" src="https://img.shields.io/badge/ML-CV%20%7C%20Signals%20%7C%20Materials-2E8B57">
  <img alt="Status" src="https://img.shields.io/badge/status-active%20research%20workspace-1F6FEB">
</p>

# MicroForge AI: Materials ML Research Workspace

This repository is my working materials ML project. It connects microscopy image analysis, manufacturing/process signals, tool-wear prediction, and materials-property regression in one reproducible Python codebase.

The project is split into smaller folders so each part can be checked independently. Shared code is in `src/`, experiment runners are in `scripts/`, configuration files are in `configs/`, and generated outputs are in `reports/`.

![MicroForge AI system architecture](assets/system_architecture.svg)

## Project Areas

![Project evidence map](assets/portfolio_evidence_map.svg)

| Project | Method Focus | Current Output |
|---|---|---|
| [01 Microscopy CV](projects/01_microscopy_cv) | SEM-style segmentation, active learning, and prediction panels | segmentation metrics and qualitative outputs |
| [02 Process Signal ML](projects/02_process_signal_ml) | 20 kHz force/moment feature extraction and ML modelling | `Fx`, `Fy`, `Fz`, `Mz` feature table and model results |
| [03 Tool-Wear Benchmarks](projects/03_tool_wear_benchmarks) | grouped-validation wear regression and classification | Vicomtech and Uniwear benchmark reports |
| [04 Materials Property ML](projects/04_materials_property_ml) | tabular preprocessing, property regression, and feature importance | concrete compressive-strength regression report |
| [05 Multimodal Platform](projects/05_multimodal_platform) | shared benchmark automation and report generation | combined leaderboard, reports, and benchmark overview |

## Current Results

| Benchmark | Task | Validation | Result |
|---|---|---|---|
| Vicomtech tool wear | flank-wear regression | held-out tool IDs | R2 `0.8680` |
| Vicomtech tool wear | wear-stage classification | held-out tool IDs | macro F1 `0.6472` |
| Katulu Uniwear | wear-stage classification | held-out experiment tags | macro F1 `0.5205` |
| Concrete strength | compressive-strength regression | random train/test split | R2 `0.8990` |
| NASA EBC SEM suite | segmentation smoke test | held-out images | foreground IoU `0.1174` |

Full leaderboard: [reports/materials_ai_leaderboard.md](reports/materials_ai_leaderboard.md)

## Repository Map

```text
projects/                  Short project-specific summaries and copied result figures
src/                       Reusable Python package
scripts/                   Data loading, preprocessing, training, evaluation, and reporting
configs/                   Dataset and benchmark configuration files
reports/                   Generated metrics, figures, leaderboards, and summaries
docs/                      Dataset notes, methodology, project summary, and research notes
tests/                     Smoke tests for data, models, reports, links, and visuals
```

## Quick Start

```powershell
git clone https://github.com/Ghanshyambabariya/microscopy_cv_research.git
cd microscopy_cv_research
python -m pip install -e . pytest timm
python scripts/run_all_benchmarks.py
python -m pytest -q
```

## Useful Notes

- [Project summary](docs/project_summary.md)
- [Research notes](docs/research_notes.md)
- [System design](docs/advanced_system_design.md)
- [Dataset cards](docs/datasets.md)
- [Methodology](docs/methodology.md)

## Current Status

This is an active research workspace. Some parts use real online datasets, and some parts use clearly marked development scaffolds. The microscopy results are baseline measurements, not final performance claims.
