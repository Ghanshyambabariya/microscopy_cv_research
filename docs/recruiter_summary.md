# Recruiter Summary: MicroForge AI

## One-Line Pitch

Research-oriented materials-intelligence portfolio combining microscopy computer vision, high-frequency process-signal analysis, tool-wear prediction, materials-property ML, and multimodal benchmark automation.

## Why It Is Relevant

This project demonstrates practical ML work across the full data lifecycle: online dataset discovery, download, cleaning, preprocessing, train/test splitting, model training, metrics, visualizations, and reproducible reports.

It is designed to be easy to review: the root README gives the full story, while each project folder gives one focused role-specific case study.

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

Built MicroForge AI, a materials-intelligence portfolio integrating microscopy CV, high-frequency process-signal ML, tool-wear prediction, and materials-property regression; implemented real GitHub dataset ingestion, cleaning, leakage-aware splitting, benchmark training, reporting, and visual evidence across SEM, force/vibration, machining, and concrete-strength datasets.

## Reviewer Confidence Signals

- Clear project folders for different job/research directions.
- Real datasets are separated from simulated development scaffolding.
- Manufacturing benchmarks use grouped validation where leakage risk is high.
- Tests and config-driven scripts make the work reproducible instead of notebook-only.
