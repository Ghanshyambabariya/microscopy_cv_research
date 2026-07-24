# Microscopy CV

[Back to project index](../README.md) | [Back to main README](../../README.md)

This project contains the microscopy image-analysis part of the workspace. It includes semantic segmentation, active-learning logic, and result panels for SEM-style material images.

## Why This Project Matters

Microscopy datasets are often small, expensive to label, and sensitive to imaging conditions. A useful workflow should handle segmentation, show predictions visually, and keep the path open for low-label training.

## Methods

- loaded SEM-style image/mask data from dataset registries
- resized and normalized images for model training
- trained a lightweight UNet segmentation baseline
- evaluated pixel accuracy, IoU, and F1-style segmentation metrics
- added an entropy-based active-learning loop
- saved qualitative prediction panels

## Best Result

| Dataset | Task | Model | Result |
|---|---|---|---|
| NASA EBC SEM suite | semantic segmentation smoke test | UNetSmall | foreground IoU `0.1174` |
| NASA EBC SEM baseline | semantic segmentation | UNetSmall | pixel accuracy `0.9480` from previous trained run |

## Relevance

Microscopy image analysis, segmentation, active learning, small-data modelling, and visual model inspection.

## Results

![SEM predictions](results/sem_ebc_predictions.png)

![Active learning loop](../../assets/active_learning_loop.svg)

Leaderboard: [results/sem_leaderboard.md](results/sem_leaderboard.md)

## Main Files

- `scripts/run_sem_suite.py`
- `scripts/run_active_sem.py`
- `configs/sem_suite.json`
- `configs/active_sem_ebc.json`
- `src/microscopy_cv_research/models/segmentation.py`

## Run

```powershell
python scripts/run_sem_suite.py --config configs/sem_suite.json
python scripts/run_active_sem.py --config configs/active_sem_ebc.json
```
