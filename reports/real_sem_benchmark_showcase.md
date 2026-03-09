# Real SEM Benchmark Showcase

This report is built from NASA's `pretrained-microscopy-models` benchmark segmentation data bundled under the MIT license in that repository.

## What Is Integrated

- Modality: SEM-style materials microscopy
- Task: semantic segmentation
- Source: NASA MicroNet benchmark segmentation data
- Evidence in this repo: real benchmark sample inputs, masks, overlays, and split counts

## Split Summary

| Dataset | Train pairs | Val pairs | Test pairs |
|---|---|---|---|
| EBC1 | 18 | 3 | 3 |
| EBC2 | 4 | 3 | 3 |
| EBC3 | 15 | 3 | 6 |
| Super1 | 10 | 4 | 4 |
| Super2 | 4 | 4 | 4 |
| Super3 | 1 | 4 | 4 |
| Super4 | 4 | 4 | 5 |

## Portfolio Figure

![NASA SEM benchmark](reports/figures/real_sem_benchmark.png)

## Interpretation

- This is the first real microscopy benchmark integrated directly into the project portfolio.
- It is segmentation-focused, so it complements the repo's existing classification and regression starter tracks.
- TEM and EBSD are still benchmark targets rather than fully integrated evaluated tasks.