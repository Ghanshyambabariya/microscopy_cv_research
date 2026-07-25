# 02. Process-Signal ML

[Back to project index](../README.md) | [Back to main README](../../README.md)

Independent project for converting high-frequency force and moment channels into modelling features.

## Why Used

Raw high-frequency signals are usually too large to use directly in simple ML workflows. Windowing and feature extraction convert the signal into interpretable time-domain, frequency-domain, and process-energy descriptors.

## How Used

- generated synchronized high-frequency force/moment channels
- segmented the signal into analysis windows
- extracted RMS, peak, crest factor, spectral centroid, bandpower, resultant force, and energy features
- trained baseline models for process-state classification and property regression
- saved signal plots, feature tables, and metrics

## Final Results

| Task | Data | Model | Result |
|---|---|---|---|
| process-state classification | 20 kHz force/moment windows | Random Forest | accuracy `1.0000` |
| property regression | 20 kHz force/moment windows | Random Forest | R2 `0.9998` |

## Research Alignment

The implementation follows signal-analysis practice by windowing the raw channels before modelling and by separating time-domain, spectral, resultant-force, and energy features.

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
