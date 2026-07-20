# 01. Microscopy CV Research

[Back to project index](../README.md) | [Back to main README](../../README.md)

Computer vision workflow for microscopic material images: SEM segmentation, synthetic microscopy generation, active learning, and transfer-learning-ready benchmark structure.

Context: microscopy image analysis, semantic segmentation, and low-label learning.

## At A Glance

| Item | Details |
|---|---|
| Data | NASA EBC SEM benchmark snapshots plus synthetic/starter microscopy data |
| Tasks | semantic segmentation, uncertainty ranking, synthetic data expansion |
| Model | lightweight UNet baseline; designed for MicroNet/Swin encoder upgrade |
| Current result | foreground IoU `0.1174` smoke test; previous SEM pixel accuracy `0.9480` |
| Main command | `python scripts/run_sem_suite.py --config configs/sem_suite.json` |

## Result Snapshot

![SEM predictions](results/sem_ebc_predictions.png)

![Active learning loop](../../assets/active_learning_loop.svg)

Leaderboard: [results/sem_leaderboard.md](results/sem_leaderboard.md)

## What To Inspect

- `scripts/run_sem_suite.py` for dataset loading, preprocessing, training, and evaluation.
- `scripts/run_active_sem.py` for entropy-based active learning rounds.
- `configs/sem_suite.json` and `configs/active_sem_ebc.json` for reproducible benchmark settings.
- `reports/real_sem_benchmark_showcase.md` for the broader SEM benchmark narrative.

## Research Upgrade Path

Replace the lightweight baseline encoder with a microscopy-pretrained encoder such as MicroNet or Swin, run the longer GPU config, and report cross-dataset SEM/TEM/EBSD generalization.
