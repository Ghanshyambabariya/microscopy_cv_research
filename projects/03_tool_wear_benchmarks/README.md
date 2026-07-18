# Project 03: Real Tool-Wear Benchmarks

**Focus:** real online manufacturing datasets for tool-wear prediction.

This project downloads real GitHub-hosted machining datasets, cleans them, creates wear labels or time windows, splits by tool/experiment to reduce leakage, trains models, and reports performance.

## Why This Project Is Unique

- Uses real online datasets, not only synthetic data.
- Validates schema and cleans numeric features before training.
- Uses grouped validation: held-out tool IDs or held-out experiment tags.
- Includes both tabular sensor features and time-window force/vibration features.

## Benchmarks

| Dataset | Task | Split | Result |
|---|---|---|---|
| Vicomtech tool wear | flank-wear regression | held-out tool IDs | R2 `0.8680` |
| Vicomtech tool wear | wear-stage classification | held-out tool IDs | macro F1 `0.6472` |
| Katulu Uniwear | tool-wear regression | held-out experiment tags | R2 `0.2397` |
| Katulu Uniwear | wear-stage classification | held-out experiment tags | macro F1 `0.5205` |

## Results

![Vicomtech tool wear](results/external_tool_wear_vicomtech.png)

![Uniwear force vibration](results/external_uniwear_tool_wear.png)

Reports:
- [Vicomtech report](results/external_tool_wear_vicomtech_report.md)
- [Uniwear report](results/external_uniwear_tool_wear_report.md)

## Run

```powershell
python scripts/run_external_tool_wear.py --config configs/external_tool_wear_vicomtech.json
python scripts/run_external_uniwear.py --config configs/external_uniwear_tool_wear.json
```

## Next Upgrade

Add temporal models such as 1D CNN, LSTM, or Transformer on raw time-series windows for the Uniwear dataset.

