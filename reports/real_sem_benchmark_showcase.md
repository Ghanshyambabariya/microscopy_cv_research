# Real SEM Benchmark Showcase

This report is built from NASA's `pretrained-microscopy-models` benchmark segmentation data bundled under the MIT license in that repository.

## What Is Integrated

- Modality: SEM-style materials microscopy
- Task: semantic segmentation
- Source: NASA MicroNet benchmark segmentation data
- Evidence in this repo: real benchmark sample inputs, masks, overlays, split counts, trained baseline metrics, and prediction figures

## Split Summary

| Dataset | Train pairs | Val pairs | Test pairs |
|---|---|---|---|
| EBC1 | 18 | 3 | 3 |
| EBC2 | 4 | 3 | 3 |
| EBC3 | 15 | 3 | 6 |
| Super1 | 10 | 4 | 4 |
| Super2 | 4 | 4 | 4 |
| Super3 | 1 | 4 | 4 |
| Super4 | 4 | 4 | 5 |

## Result Figure

![NASA SEM benchmark](reports/figures/real_sem_benchmark.png)

## Trained EBC Baseline

A real segmentation baseline is now trained on the usable EBC benchmark family.

- benchmark family: `EBC1`, `EBC2`, `EBC3`
- test pixel accuracy: `0.9480`
- test mean IoU foreground: `0.4334`
- test mean Dice foreground: `0.5293`

Detailed experiment report: [reports/sem_ebc_segmentation_report.md](reports/sem_ebc_segmentation_report.md)

![NASA EBC predictions](reports/figures/sem_ebc_predictions.png)

## Interpretation

- This is the first real microscopy benchmark integrated directly into the project structure.
- It is now more than a showcase: the repo contains a trained segmentation baseline with real benchmark metrics.
- TEM and EBSD are still benchmark targets rather than fully integrated evaluated tasks.
