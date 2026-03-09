# Showcase

This page is a quick portfolio-style view of the project.

## Framework

- Supervised encoder training for microscopy classification and regression
- Synthetic image generation for augmentation and stress testing
- Hybrid multitask learning for structure-property prediction
- Public microscopy dataset ingestion showcase

## Benchmark Snapshot

| Track | Metric | Value |
|---|---|---|
| Supervised classification | Accuracy | 0.3333 |
| Supervised classification | Macro F1 | 0.2500 |
| Supervised regression | MAE | 0.2607 |
| Supervised regression | RMSE | 0.3092 |
| Supervised regression | R2 | -1.9232 |
| Hybrid classification | Accuracy | 0.0000 |
| Hybrid classification | Macro F1 | 0.0000 |
| Hybrid regression | MAE | 0.5312 |
| Hybrid regression | RMSE | 0.5622 |
| Hybrid regression | R2 | -8.6643 |
| Synthetic generation | Images created | 54 |

## Visual Summary

![Benchmark overview](reports/figures/benchmark_overview.png)

![Training curves](reports/figures/training_curves.png)

![Sample gallery](reports/figures/sample_gallery.png)

## Public Dataset Showcase

The repo also includes a public-data showcase builder for BBBC microscopy sources.

Generated locally:
- `reports/public_showcase.md`
- `reports/public_showcase_manifest.json`
- `reports/figures/showcase/*.png`

## Notes

These results are from a small synthetic starter dataset designed to demonstrate the framework. Real scientific performance will depend on your real microscopy data, labels, and task logic.
