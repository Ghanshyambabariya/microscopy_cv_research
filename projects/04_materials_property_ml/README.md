# 04. Materials Property ML

[Back to project index](../README.md) | [Back to main README](../../README.md)

Materials-informatics workflow for predicting properties from composition, processing, and descriptor-style tabular data.

Context: materials informatics, property prediction, and structure-property relationships.

## At A Glance

| Item | Details |
|---|---|
| Data | concrete mix composition, curing age, and compressive strength |
| Task | compressive-strength regression |
| Model | RandomForest baseline with feature importance reporting |
| Current result | R2 `0.8990` |
| Main command | `python scripts/run_external_concrete.py --config configs/external_concrete_strength.json` |

## Result Snapshot

![Concrete strength](results/external_concrete_strength.png)

Report: [results/external_concrete_strength_report.md](results/external_concrete_strength_report.md)

## What To Inspect

- `scripts/run_external_concrete.py` for download, cleaning, preprocessing, training, and reporting.
- `reports/external_concrete_strength_report.md` for metrics and top feature drivers.
- `docs/structure_property_roadmap.md` for planned alloy, battery, polymer, and heat-treatment extensions.

## Research Upgrade Path

Add richer open materials datasets and compare RandomForest/XGBoost baselines against graph, composition-transformer, or descriptor-fusion models.
