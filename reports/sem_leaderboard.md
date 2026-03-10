# SEM Benchmark Leaderboard

This table aggregates real SEM segmentation experiments tracked in this repo.

| Experiment | Dataset family | Pixel Acc | Mean IoU (fg) | Mean Dice (fg) | Notes |
|---|---|---|---|---|---|
| NASA EBC baseline | EBC1/2/3 | 0.9480 | 0.4334 | 0.5293 | UNetSmall, class-weighted CE, 10 epochs |
| NASA EBC active round 1 | EBC1/2/3 | 0.7649 | 0.1107 | 0.1813 | seed=6, acquisition=4, entropy selection |
| NASA EBC active round 2 | EBC1/2/3 | 0.7791 | 0.0126 | 0.0245 | after two acquisition rounds |

Planned additions (not yet integrated):
- MudrockNet SEM (pore/grain)
- EMPS particle SEM segmentation
- Automatic SEM agglomerated-particle dataset
- EBSD orientation/phase benchmarks (separate head)
