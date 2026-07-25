# Process-Signal ML

[Back to project index](../README.md) | [Back to main README](../../README.md)

This project turns high-frequency force and moment channels into machine-learning features. The current example uses 20 kHz `Fx`, `Fy`, `Fz`, and `Mz` records so the full pipeline can run without a private machine dataset.

## Why This Project Matters

Raw high-frequency signals are usually too large to use directly in simple ML workflows. Windowing and feature extraction convert the signal into interpretable time-domain, frequency-domain, and process-energy descriptors.

## Methods

- generated synchronized high-frequency force/moment channels
- segmented the signal into analysis windows
- extracted RMS, peak, crest factor, spectral centroid, bandpower, resultant force, and energy features
- trained baseline models for process-state classification and property regression
- saved signal plots, feature tables, and metrics

## Best Result

| Task | Data | Model | Result |
|---|---|---|---|
| process-state classification | 20 kHz force/moment windows | Random Forest | accuracy `1.0000` |
| property regression | 20 kHz force/moment windows | Random Forest | R2 `0.9998` |

## Relevance

Signal processing, sensor analytics, process monitoring, feature engineering, and process-property modelling.

## Results

![Materials signal summary](results/materials_signal_summary.png)

Metrics: [results/materials_signal_metrics.json](results/materials_signal_metrics.json)

## Main Files

- `scripts/run_materials_signal.py`
- `configs/materials_signal.json`
- `src/microscopy_cv_research/signals/features.py`
- `src/microscopy_cv_research/training/materials_signal.py`

## Run

```powershell
python scripts/run_materials_signal.py --config configs/materials_signal.json
```
