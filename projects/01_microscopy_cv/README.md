# 01. Microscopy CV

[Back to project index](../README.md) | [Back to main README](../../README.md)

Independent project for SEM-style microscopy segmentation, active learning, and visual inspection of predicted masks.

## Why Used

Microscopy datasets are often small, expensive to label, and sensitive to imaging conditions. The workflow combines segmentation, visual prediction panels, and active learning for low-label image analysis.

## How Used

- loaded SEM-style image/mask data from dataset registries
- resized and normalized images for model training
- trained a lightweight UNet segmentation baseline
- evaluated pixel accuracy, IoU, and F1-style segmentation metrics
- added an entropy-based active-learning loop
- saved qualitative prediction panels

## Final Results

| Dataset | Task | Model | Result |
|---|---|---|---|
| NASA EBC SEM suite | semantic segmentation smoke test | UNetSmall | foreground IoU `0.1174` |
| NASA EBC SEM baseline | semantic segmentation | UNetSmall | pixel accuracy `0.9480` from previous trained run |

## Research Alignment

The implementation follows current microscopy-CV practice by reporting pixel-level metrics, saving qualitative masks, and using uncertainty sampling for low-label segmentation.

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
