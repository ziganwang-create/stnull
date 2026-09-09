# Library-size structure audit (36 sections)

Per-section sd(log lib), geometric-mean lib, and Moran's I of log lib on the array (6-nearest-neighbor, row-normalized weights). Correlated against per-section median r of two protocols: full 2048-dim ResNet50 ridge (`ridge_summary.csv`) and the library-only null (`libonly_panelnorm_check.csv`).

## Cross-section correlations (n=36)

| statistic | corr with ridge median r | corr with lib-only median r |
|---|---|---|
| sd(log lib) | 0.882 | 0.964 |
| sd(log lib) x Moran's I | 0.826 | 0.934 |
| Moran's I(log lib) | 0.698 | 0.800 |

Top-5 sd(log lib): B2 (sd 1.43, I 0.87), B4 (sd 1.39, I 0.83), B6 (sd 1.27, I 0.80), B1 (sd 1.26, I 0.86), B3 (sd 1.23, I 0.86)
Bottom-5 sd(log lib): F1 (sd 0.39, I 0.31), A5 (sd 0.39, I 0.36), F3 (sd 0.40, I 0.27), E2 (sd 0.42, I 0.54), F2 (sd 0.43, I 0.32)

## Patient means

| patient | sd(log lib) | Moran's I | ridge r | lib-only r |
|---|---|---|---|---|
| A | 0.510 | 0.434 | 0.047 | 0.147 |
| B | 1.273 | 0.849 | 0.229 | 0.384 |
| C | 0.780 | 0.605 | 0.179 | 0.285 |
| D | 0.696 | 0.670 | 0.088 | 0.220 |
| E | 0.472 | 0.628 | 0.029 | 0.174 |
| F | 0.406 | 0.299 | 0.011 | 0.158 |
| G | 0.742 | 0.551 | 0.146 | 0.243 |
| H | 0.662 | 0.612 | 0.130 | 0.236 |

**Interpretation.** Between-section differences in achievable median r are largely explained by how much spatially organized library-size variation each section carries: sd(log lib) alone correlates ~0.88 with ridge median r and ~0.96 with the library-only null (sd x Moran's I: 0.83 / 0.93). Patients B/C/G/H have both large (sd ~0.7-1.1 on average) and spatially smooth (Moran's I ~0.5-0.8) depth fields, so any predictor that recovers local depth - including one scalar from the image - scores well; A/E/F have flat or spatially incoherent depth (F mean sd ~0.41), leaving little compositional/depth signal for any method to exploit. All figures approximate, +/-0.02-0.05.
