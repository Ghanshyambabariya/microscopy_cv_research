# Dataset Cards

These cards summarize the online datasets currently connected to the materials-AI platform.

## Dataset Card Fields

Dataset cards record source URL, license or usage note, sample count, label definition, split strategy, preprocessing steps, known limitations, and benchmark command.

## Runnable Datasets

### Vicomtech dataset-machine-tool-wear

- type: `tabular_sensor_features`
- source: https://github.com/Vicomtech/dataset-machine-tool-wear
- status: `implemented`
- runner: `scripts/run_external_tool_wear.py`
- report: `reports/external_tool_wear_vicomtech_report.md`
- note: strongest current real-data manufacturing benchmark because validation is grouped by held-out tool IDs

### Katulu Uniwear Dataset

- type: `force_vibration_time_series`
- source: https://github.com/katulu-io/uniwear-dataset
- status: `implemented`
- runner: `scripts/run_external_uniwear.py`
- report: `reports/external_uniwear_tool_wear_report.md`
- note: force/vibration wear-stage modelling benchmark with window-level signal features

### Concrete Compressive Strength

- type: `materials_property_tabular`
- source: https://github.com/stedy/Machine-Learning-with-R-datasets
- status: `implemented`
- runner: `scripts/run_external_concrete.py`
- report: `reports/external_concrete_strength_report.md`
- note: compact materials-property regression benchmark for materials-informatics work

## Large Optional Targets

### CoMMonS microscopic material surface dataset

- type: `microscopy_material_surface_images`
- source: https://github.com/olivesgatech/CoMMonS
- status: `documented_large_target`
- runner: `scripts/run_external_commons.py`
- report: `reports/external_commons_microscopy_report.md`
- reason: Sampled archive is about 1.1 GB; use for a dedicated image benchmark run rather than committing directly into this lightweight repo.
