# Showcase

This page is a quick portfolio-style view of the project.

## Framework

- Supervised encoder training for microscopy classification and regression
- Synthetic image generation for augmentation and stress testing
- Hybrid multitask learning for structure-property prediction
- Public microscopy dataset ingestion showcase
- Real benchmark target registry for SEM, TEM, and EBSD work

## Benchmark Snapshot

The starter dataset now contains 1,440 tracked microscopy-style images across 180 specimen groups.

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

## Visual Summary

![Benchmark overview](reports/figures/benchmark_overview.png)

![Training curves](reports/figures/training_curves.png)

![Sample gallery](reports/figures/sample_gallery.png)

## Real Benchmark Direction

This portfolio is now split into two levels:

1. Current verified framework results
- the tracked metrics on the 1,440-image starter dataset

2. Real microscopy benchmark targets
- SEM segmentation and defect tasks
- TEM or EMPIAR-derived transfer tasks
- EBSD phase or orientation tasks

See [reports/real_benchmark_portfolio.md](reports/real_benchmark_portfolio.md) for the full benchmark map.

## Notes

These are still starter-data results, but they now come from a much larger and more separable 1,440-image benchmark rather than the earlier tiny toy set. The repo is behaving like a solid framework demonstration, but it is not yet a completed real-microscopy benchmark portfolio.
