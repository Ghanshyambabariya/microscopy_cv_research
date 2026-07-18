# 05. Multimodal Materials AI Platform

[Back to project index](../README.md) | [Back to portfolio](../../README.md)

Integration layer that connects microscopy, process signals, tool-wear benchmarks, and materials-property prediction into one reproducible portfolio.

> Role fit: ML engineering, scientific software, research automation, materials-AI portfolio review.

## At A Glance

| Item | Details |
|---|---|
| Inputs | microscopy images, manufacturing signals, tool-wear tables, property datasets |
| Outputs | leaderboard, dataset cards, benchmark reports, recruiter summary |
| Automation | shared configs, scripts, generated reports, copied project snapshots |
| Main evidence | [materials_ai_leaderboard.md](results/materials_ai_leaderboard.md) |
| Main command | `python scripts/run_all_benchmarks.py` |

## Result Snapshot

![Benchmark overview](results/benchmark_overview.png)

![System architecture](../../assets/system_architecture.svg)

Reports: [leaderboard](results/materials_ai_leaderboard.md) | [platform report](results/materials_ai_platform_report.md)

## What To Inspect

- `scripts/run_all_benchmarks.py` for the unified benchmark runner.
- `scripts/build_materials_ai_leaderboard.py` for collecting metrics into one table.
- `docs/recruiter_summary.md` for a short, CV-ready explanation.
- `docs/datasets.md` for dataset cards and limitations.

## Research Upgrade Path

Build a lightweight dashboard or GitHub Pages site, add model cards, and track dataset-by-dataset performance changes after active-learning retraining.
