# Materials AI Benchmark Leaderboard

| Dataset | Modality | Task | Split | Model | Metric | Result | Report |
|---|---|---|---|---|---|---|---|
| NASA EBC SEM | microscopy | segmentation | held-out images | UNetSmall | mean IoU fg | 0.4388 | reports/sem_ebc_architecture_comparison.md |
| NASA/SEM suite | microscopy | segmentation | dataset split | UNetSmall | mean IoU fg | 0.1174 | reports/sem_leaderboard.md |
| Vicomtech tool wear | process sensors | flank-wear regression | held-out tool IDs | RandomForest | R2 | 0.8680 | reports/external_tool_wear_vicomtech_report.md |
| Vicomtech tool wear | process sensors | wear-stage classification | held-out tool IDs | RandomForest | macro F1 | 0.6472 | reports/external_tool_wear_vicomtech_report.md |
| Katulu Uniwear | force/vibration | tool-wear regression | held-out experiment tags | RandomForest | R2 | 0.2397 | reports/external_uniwear_tool_wear_report.md |
| Katulu Uniwear | force/vibration | wear-stage classification | held-out experiment tags | RandomForest | macro F1 | 0.5205 | reports/external_uniwear_tool_wear_report.md |
| Concrete strength | materials tabular | compressive-strength regression | random split | XGBoost | R2 | 0.9173 | reports/external_concrete_strength_report.md |
| CoMMonS | microscopy material surface | optional image classification | large-data target | RandomForest descriptors | status | target | reports/external_commons_microscopy_report.md |
