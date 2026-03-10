# Showcase

This page is a quick portfolio-style view of the project.

## Framework

- Supervised encoder training for microscopy classification and regression
- Synthetic image generation for augmentation and stress testing
- Hybrid multitask learning for structure-property prediction
- Public microscopy dataset ingestion showcase
- Real benchmark portfolio for SEM, with active learning loop
- Planned targets for TEM and EBSD

## Benchmark Snapshot

The starter dataset contains 1,440 tracked microscopy-style images across 180 specimen groups, and the repo now includes real SEM segmentation experiments (baseline + active learning) from NASA MicroNet benchmark data.

| Track | Metric | Value |
|---|---|---|
| Supervised classification | Accuracy | 1.0000 |
| Supervised classification | Macro F1 | 1.0000 |
| Supervised regression | MAE | 0.0414 |
| Supervised regression | RMSE | 0.0507 |
| Supervised regression | R2 | 0.9678 |
| Hybrid classification | Accuracy | 1.0000 |
| Hybrid classification | Macro F1 | 1.0000 |
| Hybrid regression | MAE | 0.0328 |
| Hybrid regression | RMSE | 0.0395 |
| Hybrid regression | R2 | 0.9804 |
| Synthetic generation | Images created | 120 |
| NASA EBC SEM baseline | Mean IoU fg | 0.4334 |
| NASA EBC SEM active round 1 | Mean IoU fg | 0.1107 |
| NASA EBC SEM active round 2 | Mean IoU fg | 0.0126 |

## Visual Summary

![Benchmark overview](reports/figures/benchmark_overview.png)

![NASA EBC predictions](reports/figures/sem_ebc_predictions.png)

## Real SEM Experiments

- Baseline: UNetSmall on EBC1/2/3 with class-weighted CE
- Active learning: entropy acquisition, seed 6 ? +4, 2 rounds
- Metrics and leaderboard in [reports/sem_leaderboard.md](reports/sem_leaderboard.md)
- Active log in `reports/sem_active_learning_log.json`

## What Is Still Missing

- TEM benchmark testing with real task outputs
- EBSD benchmark testing with real pattern data and metrics
- Additional SEM datasets (MudrockNet, EMPS, Automatic SEM) integrated into the leaderboard

See [reports/real_benchmark_portfolio.md](reports/real_benchmark_portfolio.md) for the full benchmark target map.
