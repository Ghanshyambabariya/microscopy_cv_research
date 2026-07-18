# Materials AI Platform Report

This report connects microscopy computer vision, high-frequency materials process signals, and multimodal ML for structure-process-property learning.

## Signal Intelligence

- task: high-frequency grinding signal classification and property regression
- sampling rate: `20000 Hz`
- nominal process duration: `20.0 s`
- fast analysis window: `1.0 s`
- feature table: `data\processed\materials_signal_features.csv`
- process-quality accuracy: `1.0000`
- process-quality macro F1: `1.0000`
- property regression MAE: `0.0021`
- property regression R2: `0.9998`

![Materials signal summary](figures/materials_signal_summary.png)

## Multimodal Fusion

- task: fusion of microscopy specimen descriptors and grinding signal features
- multimodal table: `data\processed\materials_multimodal_table.csv`
- train/test split: `135` / `45` specimens
- signal features: `50`
- microscopy features: `2`
- process-quality accuracy: `1.0000`
- process-quality macro F1: `1.0000`
- property regression MAE: `0.0020`
- property regression R2: `0.9998`

## Real Online Benchmarks

| Dataset | Source | Task | Held-out split | Main result |
|---|---|---|---|---|
| Vicomtech tool wear | `https://github.com/Vicomtech/dataset-machine-tool-wear` | flank-wear regression + wear-stage classification | tool IDs | R2 `0.8680`, macro F1 `0.6472` |
| Katulu Uniwear | `https://github.com/katulu-io/uniwear-dataset` | force/vibration window wear prediction | experiment tags | R2 `0.2397`, macro F1 `0.5205` |

![External tool wear benchmark](figures/external_tool_wear_vicomtech.png)

![External Uniwear benchmark](figures/external_uniwear_tool_wear.png)

## Large Microscopy Target

CoMMonS is a strong next microscopy-material dataset target: it contains 6,912 microscopic fabric-surface images across 24 samples and expert-rated fabric properties. The sampled archive is about 1.1 GB, so it is documented as a large-data target rather than committed directly into this lightweight GitHub portfolio.

## Interpretation

The current signal data is simulated but physics-inspired: force level, chatter-band energy, torque, bursts, and impulse behavior are linked to material class and property values. This gives a working ML scaffold that can be replaced with real grinding, milling, acoustic-emission, vibration, force, torque, spindle-current, or temperature CSV files.

The portfolio value is the full structure: microscopy CV, high-frequency signal features, supervised ML, regression, multimodal fusion, and generated reports.
