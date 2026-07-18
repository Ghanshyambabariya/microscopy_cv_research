# Project 02: Materials Process Signal ML

**Focus:** high-frequency process-signal analysis for materials manufacturing.

This project simulates grinding-style force/torque channels and extracts physics-aware features for process-quality classification and property regression.

## Why This Project Is Unique

- Models the exact signal style needed for force, vibration, acoustic, torque, or spindle-current datasets.
- Uses `Fx`, `Fy`, `Fz`, and `Mz` channels with a nominal `20 kHz` sampling rate.
- Extracts RMS, peak, crest factor, spectral centroid, bandpower, resultant force, and energy proxy features.
- Bridges materials processing with ML-ready feature engineering.

## Main Evidence

| Task | Data | Model | Result |
|---|---|---|---|
| Process quality | simulated grinding signals | RandomForest | accuracy `1.0000` |
| Property regression | simulated grinding signals | RandomForest | R2 `0.9998` |

## Results

![Materials signal summary](results/materials_signal_summary.png)

See: [results/materials_signal_metrics.json](results/materials_signal_metrics.json)

## Run

```powershell
python scripts/run_materials_signal.py --config configs/materials_signal.json
```

## Next Upgrade

Replace simulated signals with measured machine force, acoustic emission, vibration, spindle current, or temperature CSV files.

