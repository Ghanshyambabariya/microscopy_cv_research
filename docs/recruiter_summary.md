# Recruiter Summary: MatSci-AI Benchmark Platform

## One-Line Pitch

Materials-AI portfolio project combining microscopy computer vision, high-frequency process-signal analysis, materials-property prediction, and multimodal machine learning.

## Why It Is Relevant

This project demonstrates practical ML work across the full data lifecycle: online dataset discovery, download, cleaning, preprocessing, train/test splitting, model training, metrics, visualizations, and reproducible reports.

## Best Evidence

| Area | Evidence |
|---|---|
| Process monitoring | Tool-wear prediction on real Vicomtech and Uniwear datasets |
| Materials informatics | Concrete compressive-strength property regression |
| Microscopy AI | SEM segmentation and active-learning workflow |
| ML engineering | Config-driven benchmark scripts, tests, reports, dataset cards |
| Scientific judgment | Grouped splits for tool/experiment leakage control and documented dataset limitations |

## Strongest Results

| Benchmark | Result |
|---|---|
| Vicomtech flank-wear regression | R2 `0.8680` on held-out tool IDs |
| Vicomtech wear-stage classification | macro F1 `0.6472` on held-out tool IDs |
| Concrete compressive-strength regression | R2 `0.8990` |
| Katulu Uniwear wear-stage classification | macro F1 `0.5205` on held-out experiment tags |

## Technologies

Python, PyTorch, torchvision, scikit-learn, pandas, NumPy, matplotlib, image processing, signal features, Random Forest models, config-driven benchmark automation.

## Suggested CV Bullet

Built a materials-AI benchmark platform integrating microscopy CV, high-frequency process-signal ML, and materials-property prediction; implemented real GitHub dataset ingestion, cleaning, leakage-aware splitting, benchmark training, reporting, and visual evidence across SEM, tool-wear, force/vibration, and concrete-strength datasets.

