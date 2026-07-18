# Project 01: Microscopy CV Research

**Focus:** computer vision for microscopic material images.

This project contains the microscopy-facing part of the portfolio: SEM segmentation, synthetic microscopy data, supervised image classification/regression, active learning, and benchmark reporting.

## Why This Project Is Unique

- Uses real NASA EBC SEM segmentation data in addition to synthetic/starter microscopy data.
- Includes active learning for low-label microscopy workflows.
- Produces qualitative segmentation figures with prediction overlays and error maps.
- Designed for research roles involving microstructure analysis, defect segmentation, and scientific computer vision.

## Main Evidence

| Task | Dataset | Model | Result |
|---|---|---|---|
| SEM segmentation | NASA EBC quick suite | UNetSmall | foreground IoU `0.1174` smoke-test |
| SEM baseline report | NASA EBC | UNetSmall | pixel accuracy `0.9480` previous trained run |
| Active learning | NASA EBC | entropy acquisition | tracked rounds in `sem_active_learning_log.json` |

## Results

![SEM predictions](results/sem_ebc_predictions.png)

See: [results/sem_leaderboard.md](results/sem_leaderboard.md)

## Run

```powershell
python scripts/run_sem_suite.py --config configs/sem_suite.json
python scripts/run_active_sem.py --config configs/active_sem_ebc.json
```

## Next Upgrade

Run the longer `configs/sem_suite_benchmark.json` config on GPU and add a stronger pretrained encoder comparison.

