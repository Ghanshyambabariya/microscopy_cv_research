# 03. Tool-Wear Prediction Benchmark

[Back to project index](../README.md) | [Back to main README](../../README.md)

Independent project for tool-wear regression and wear-stage classification on public machining datasets.

## Why Used

Tool wear is a practical condition-monitoring problem. The benchmark estimates wear or wear stage from measured process data and evaluates generalization on tools or experiments not seen during training.

## How Used

- downloaded and cleaned real public machining datasets
- checked numeric columns and target labels
- created regression and wear-stage classification targets
- used grouped train/test splits by tool ID or experiment tag
- trained Random Forest baselines
- reported regression and classification metrics
- saved plots and Markdown reports

## Final Results

| Dataset | Task | Split | Result |
|---|---|---|---|
| Vicomtech tool wear | flank-wear regression | held-out tool IDs | R2 `0.8680` |
| Vicomtech tool wear | wear-stage classification | held-out tool IDs | macro F1 `0.6472` |
| Katulu Uniwear | wear-stage classification | held-out experiment tags | macro F1 `0.5205` |

## Research Alignment

The implementation uses grouped validation to reduce leakage and reports both regression and classification metrics for practical tool-wear analysis.

## Results

![Vicomtech tool wear](results/external_tool_wear_vicomtech.png)

![Uniwear force vibration](results/external_uniwear_tool_wear.png)

Reports: [Vicomtech](results/external_tool_wear_vicomtech_report.md) | [Uniwear](results/external_uniwear_tool_wear_report.md)

## Main Files

- `scripts/run_external_tool_wear.py`
- `scripts/run_external_uniwear.py`
- `configs/external_tool_wear_vicomtech.json`
- `configs/external_uniwear_tool_wear.json`
- `configs/online_dataset_registry.json`

## Run

```powershell
python scripts/run_external_tool_wear.py --config configs/external_tool_wear_vicomtech.json
python scripts/run_external_uniwear.py --config configs/external_uniwear_tool_wear.json
```
