# MatSci-AI Portfolio

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB)
![ML](https://img.shields.io/badge/ML-CV%20%7C%20Signals%20%7C%20Materials-2E8B57)
![Status](https://img.shields.io/badge/status-CV--ready%20portfolio-1F6FEB)

**A clean materials-AI portfolio connecting microscopy computer vision, process-signal analysis, tool-wear prediction, materials-property ML, and multimodal benchmark automation.**

This repository is organized as five focused projects. Each folder has its own README, results, figures, run commands, and next-step ideas.

## Project Folders

| Project | Focus | Best Evidence |
|---|---|---|
| [01 Microscopy CV](projects/01_microscopy_cv) | SEM segmentation, active learning, synthetic microscopy data | NASA EBC SEM figures and leaderboard |
| [02 Process Signal ML](projects/02_process_signal_ml) | high-frequency `Fx`, `Fy`, `Fz`, `Mz` feature engineering | signal summary figure and process-quality model |
| [03 Tool-Wear Benchmarks](projects/03_tool_wear_benchmarks) | real online machining/tool-wear datasets | Vicomtech R2 `0.8680`, Uniwear macro F1 `0.5205` |
| [04 Materials Property ML](projects/04_materials_property_ml) | tabular materials-informatics regression | concrete strength R2 `0.8990` |
| [05 Multimodal Platform](projects/05_multimodal_platform) | unified reports, dataset cards, benchmark runner | materials-AI leaderboard and recruiter summary |

## Why This Looks Good For A CV

- **Real datasets:** Vicomtech tool wear, Katulu Uniwear, concrete strength, NASA SEM benchmark, optional CoMMonS target.
- **Multiple ML directions:** computer vision, signal analysis, materials informatics, active learning, and multimodal fusion.
- **End-to-end pipeline:** download, clean, preprocess, split, train, evaluate, visualize, and report.
- **Research judgment:** group-based splits for tool/experiment leakage control and clear documentation of large-data limitations.
- **Clean structure:** each project is separated, but all share one reproducible codebase.

## Headline Results

| Benchmark | Task | Result |
|---|---|---|
| Vicomtech tool wear | flank-wear regression on held-out tool IDs | R2 `0.8680` |
| Vicomtech tool wear | wear-stage classification on held-out tool IDs | macro F1 `0.6472` |
| Katulu Uniwear | wear-stage classification on held-out experiments | macro F1 `0.5205` |
| Concrete strength | compressive-strength regression | R2 `0.8990` |
| NASA EBC SEM suite | quick semantic segmentation smoke test | foreground IoU `0.1174` |

Full leaderboard: [reports/materials_ai_leaderboard.md](reports/materials_ai_leaderboard.md)

## Quick Start

```powershell
git clone https://github.com/Ghanshyambabariya/microscopy_cv_research.git
cd microscopy_cv_research
python -m pip install -e . pytest timm
python scripts/run_all_benchmarks.py
python -m pytest -q
```

## Key Links

- [Recruiter summary](docs/recruiter_summary.md)
- [Dataset cards](docs/datasets.md)
- [Materials AI leaderboard](reports/materials_ai_leaderboard.md)
- [Materials AI platform report](reports/materials_ai_platform_report.md)
- [Online dataset registry](configs/online_dataset_registry.json)

## Project ID

`MATSCI-AI-PORTFOLIO`

**Positioning:** materials science + machine learning portfolio for microscopy, manufacturing signals, and structure-property prediction.

