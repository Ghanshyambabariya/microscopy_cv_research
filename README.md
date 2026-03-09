# Microscopy CV Research

Research-structured computer vision project for microscopic images with three connected tracks:

1. Supervised learning with pretrained encoders for microscopy-style images
2. Synthetic data generation for balancing, augmentation, and stress testing
3. Hybrid learning that combines classification, regression, and synthetic mixing

## What Is Runnable Right Now

- `scripts/create_sample_dataset.py` generates a small microscopy-style dataset with class labels and a continuous property target
- `scripts/run_supervised.py` trains either classification or regression on the sample dataset
- `scripts/run_synthetic.py` generates extra synthetic microscopy images and a study report
- `scripts/run_hybrid.py` runs a multitask hybrid model with classification + regression heads

## Run Order

```powershell
C:\Users\ghans\AppData\Local\Programs\Python\Python312\python.exe scripts\create_sample_dataset.py
C:\Users\ghans\AppData\Local\Programs\Python\Python312\python.exe scripts\run_supervised.py --config configs\supervised_classification.json
C:\Users\ghans\AppData\Local\Programs\Python\Python312\python.exe scripts\run_supervised.py --config configs\supervised_regression.json
C:\Users\ghans\AppData\Local\Programs\Python\Python312\python.exe scripts\run_synthetic.py
C:\Users\ghans\AppData\Local\Programs\Python\Python312\python.exe scripts\run_hybrid.py
```
