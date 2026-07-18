# 03. Real Tool-Wear Benchmarks

[Back to project index](../README.md) | [Back to portfolio](../../README.md)

Real online machining benchmark project for tool-wear regression and wear-stage classification.

> Role fit: predictive maintenance, machining AI, manufacturing data science, leakage-aware ML.

## At A Glance

| Item | Details |
|---|---|
| Data | Vicomtech tool-wear data and Katulu Uniwear force/vibration data |
| Tasks | flank-wear regression and wear-stage classification |
| Validation | held-out tool IDs and held-out experiment tags |
| Best result | Vicomtech flank-wear R2 `0.8680`; Uniwear macro F1 `0.5205` |
| Main commands | `python scripts/run_external_tool_wear.py --config configs/external_tool_wear_vicomtech.json` and `python scripts/run_external_uniwear.py --config configs/external_uniwear_tool_wear.json` |

## Result Snapshots

![Vicomtech tool wear](results/external_tool_wear_vicomtech.png)

![Uniwear force vibration](results/external_uniwear_tool_wear.png)

Reports: [Vicomtech](results/external_tool_wear_vicomtech_report.md) | [Uniwear](results/external_uniwear_tool_wear_report.md)

## What To Inspect

- `scripts/run_external_tool_wear.py` for real tabular machining-data ingestion and grouped validation.
- `scripts/run_external_uniwear.py` for force/vibration dataset cleaning and experiment-level splits.
- `configs/online_dataset_registry.json` for the online dataset registry used by the benchmark system.

## Research Upgrade Path

Add temporal deep learning on raw windows, compare frequency-domain features against learned embeddings, and report uncertainty-aware wear-stage predictions for active inspection.
