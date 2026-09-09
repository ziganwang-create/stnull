# SKCM -- zero-pixel audit results

HEST-Benchmark cohort `SKCM`, audited with `stnull`. Everything here is a
property of the data rather than of a model, except where a published
leaderboard value is quoted for comparison.

| | |
|---|---|
| Organ | skin |
| Platform | Xenium |
| Samples | 2 |
| Patients | 2 |
| Official split | official LOO=patient-level |
| Leakage verdict | no leak |
| sd(log lib) range | 1.642/3.566 (official set) |
| Sparsity (zero fraction) | 0.7251 |
| Zero-pixel null, official caliber | 0.4400 |
| Best published model | GenBio-PathFM |
| Its published r | 0.6715 |
| Models the null matches or beats | 0/26 |
| Null as pct of best | 66% |
| Ceiling r_tech (median) | ~0.975 (0.968/0.982) |
| r_nbr (median) | 0.769/0.760 |
| H2 zero-biology sim >= real | 16/16 |
| H3 ordering verdict | REPLICATES 8/8 cells |

Row source: `../hb_MASTER_TABLE.csv`, cohort `SKCM`.
Cross-cohort synthesis: `../hb_CROSS_COHORT.md`.

## Files

| file | what it holds |
|---|---|
| `hb_skcm_ceiling.csv` | H4 technical ceiling r_tech by split-half thinning, plus r_nbr where present |
| `hb_skcm_depth_pretable.csv` | pre-registered descriptive statistics of sequencing depth |
| `hb_skcm_leaderboard_vs_null.csv` | published models for this cohort placed next to the null |
| `hb_skcm_libnull.csv` | H1 zero-pixel library-size null, per sample x panel x spot set x caliber |
| `hb_skcm_prereg.md` | pre-registration written before the audit ran |
| `hb_skcm_structure.csv` | H3 per-sample sd(log lib), the HER2ST-fitted prediction, and the residual |
| `hb_skcm_structure_order.csv` | H3 predicted vs observed sample ordering |
| `hb_skcm_zerobio.csv` | H2 zero-biology simulation: the same null on counts with all spatial signal removed |

## Column glossary

The per-cohort CSVs share one schema. A column absent from a file simply did not
apply to that cohort -- for example, a cohort with a single spot set has no
`spotset` column, and a cohort whose official split has no folds has no `fold`.

### Keys that appear everywhere

| column | meaning |
|---|---|
| `sample` | HEST sample id: one tissue section, one slide. |
| `fold` | Index of the official HEST-Benchmark leave-one-out fold that this sample is the *test* side of. |
| `panel` | Gene panel. `hest_var50` = the 50 genes HEST-Benchmark selects. `trainHVG*` = a panel re-selected inside the training patients only, i.e. the leakage-free counterfactual. |
| `spotset` / `spot_set` | Which spots enter the readout. `all` = every spot in the `.h5ad`. `official` = only the barcodes that also have an image patch in `patches/*.h5`, which is the set the published benchmark actually scores. |
| `caliber` | Target transform, or "measurement convention". `log1p_raw` = HEST-Benchmark's official target, `log1p` of raw counts with **no** depth normalisation. `cp10k_panel` = counts per 10k using the panel's own row sums. `cp10k_full` = counts per 10k using whole-transcriptome row sums. Same biology, three rulers, and they do not agree. |
| `n_spots`, `n_genes` | Sizes actually used after filtering. `n_genes_nan` counts genes dropped for zero variance. |

### H1 -- the zero-pixel library-size null (`*libnull*.csv`)

A per-gene ordinary least squares of gene `g` on `[1, log(total_counts)]`. One
number per spot goes in and no pixel is ever read. It is the audit baseline, not
a competitor.

| column | meaning |
|---|---|
| `med_r_trainfit`, `mean_r_trainfit` | Median and mean over genes of the Pearson r between the null's fit and the truth, fitted on the same spots it is scored on. In-sample by construction, which is the point: it is the null under the *published* scoring convention, so it is the number to set beside a leaderboard cell. |
| `med_r_cv`, `mean_r_cv` | Same, but the null is fitted 5-fold within the section and scored out of fold. The gap `trainfit - cv` is the null's own overfitting, and it is small -- with two free parameters per gene there is nothing to overfit. |
| `*_foldconcat.csv` | Test samples of a fold concatenated *before* the per-gene correlation, which is how HEST-Benchmark's `merge_fold_results` aggregates. Concatenating across sections adds between-section depth variance, and that raises r. |
| `*_official.csv` | The single number that lines up with a published leaderboard cell: official panel, official spot set, official caliber, per-gene Pearson, mean over genes, mean over folds. |
| `*_folds.csv` | The per-fold values behind that mean. |

### H2 -- the zero-biology simulation (`*zerobio*.csv`)

Counts are re-simulated so that **no gene carries any spatial signal**: every
spot draws from one shared expression profile, scaled only by that spot's real
library size. If the null scores as high on the simulation as on the real data,
the score was an artifact of depth passing through the log transform, not
biology.

| column | meaning |
|---|---|
| `med_r_real`, `mean_r_real` | The H1 null's score on the real counts. |
| `med_r_sim`, `mean_r_sim` | The same estimator on the zero-biology simulation. |
| `sim_ge_real` | 1 when `sim >= real`, i.e. the simulation reproduces or exceeds the observed score with no biology present at all. |

### H3 -- does depth structure predict the score (`*structure*.csv`)

| column | meaning |
|---|---|
| `sd_loglib` | Standard deviation of `log(total_counts)` across the section: the one summary of how much depth structure a section carries. |
| `pred_r_her2stfit` | The null's score predicted from `sd_loglib` alone, through the line fitted on HER2ST (the discovery cohort) and never refitted here. |
| `obs_med_r` | What the null actually scored. |
| `resid` | `obs_med_r - pred_r_her2stfit`. |
| `extrapolated` | 1 when this sample's `sd_loglib` falls outside the HER2ST range, so the prediction is an extrapolation and should be read as a direction rather than a value. |
| `*_structure_corr.csv` | Pearson and Spearman of `sd_loglib` against the observed score across the cohort's samples, plus the top-3 samples by observed and by predicted score. |
| `*_structure_order.csv` | For two-sample cohorts, whether the predicted ranking matched the observed one (`order_match`). |

### H4 -- the technical ceiling (`*ceiling*.csv`, `*rnbr*.csv`)

`r_tech` is a split-half reliability. Counts are binomially thinned into two
independent half-depth replicates of the same tissue, each replicate is put
through the same caliber, and the two are correlated per gene. It is the
correlation a *perfect* predictor of the biology would reach against this
measurement, so it bounds any reported r from above.

| column | meaning |
|---|---|
| `rtech_med_mine` | Median `r_tech` over genes, from this cohort's own implementation. |
| `rtech_med_stnull` | The same quantity recomputed by `stnull.ceiling.r_tech`, as a cross-check. The two agree to about 0.02 wherever both ran. |
| `rtech_q25`, `rtech_q75` | Inter-quartile range over genes. |
| `method` | Which estimator ran. `split_half_full` = binomial thinning of the full-transcriptome counts. |
| `rnbr_med`, `rnbr_mean`, `rnbr_q25`, `rnbr_q75` | `r_nbr`: correlation between a spot and the mean of its k nearest neighbours (`k`, usually 6). A model that reproduced nothing but local spatial smoothness would land here, so `r_nbr` is the floor that autocorrelation alone buys. |

### Depth pre-table (`*depth_pretable.csv`)

Descriptive statistics computed **before** any null was fitted, so the
pre-registration could state a prediction rather than a postdiction.

| column | meaning |
|---|---|
| `lib_median`, `lib_p10`, `lib_p90` | Library size (`total_counts`) quantiles. |
| `frac_lib_lt100`, `frac_zero_lib` | Fraction of spots carrying almost no counts. |
| `sd_loglib`, `cv_lib` | Dispersion of depth, on the log scale and the raw scale. |
| `moran_I_loglib_k6` | Moran's I of `log(total_counts)` over the 6-nearest-neighbour graph: how spatially clustered the depth field is. A depth artifact looks like real spatial biology exactly to the extent this is high. |
| `zero_frac_matrix`, `zero_frac_fullgenome` | Fraction of zero entries in the count matrix. |
| `genes_detected_median` | Median genes detected per spot. |
| `protocol` | Assay chemistry, where it varies within a cohort (e.g. `ffpe`). |

### Leaderboard comparison (`*leaderboard*.csv`)

| column | meaning |
|---|---|
| `model` | Published model name, or the zero-pixel null itself as one row. |
| `<cohort>_r`, `<cohort>_pearson` | The published Pearson for that model on this cohort, at the official caliber. |
| `source` | Where the published number was read from: paper table and page. |
| `null_mean_r` | The null's score under the same convention. |
| `beats_null`, `gap_vs_null`, `rank` | Whether the published model exceeds the null, by how much, and its rank once the null is inserted into the leaderboard. |

### Pre-registration (`*prereg*`, `*struct*`, `*panels*`)

`*prereg*` was written and timestamped **before** the corresponding audit ran and
names the predicted direction for H1 to H5 on this cohort; `*_prereg_stamp.txt`
carries the UTC timestamp. `*_struct.json` records the split structure -- which
samples sit in which fold and which patient each belongs to -- which is what the
leakage verdict rests on. `*_panels.json` records the exact gene lists used, per
fold and spot set.


## Provenance

Each cohort was audited by a driver that loads the HEST-Benchmark release
(`Xiao_data_codes/data/hest_bench/<COHORT>/adata/*.h5ad` and `patches/*.h5`),
fits the nulls, and writes one CSV per hypothesis. Two of those drivers ship in
this bundle verbatim and the rest follow the same pattern:

- `hest_coad/hb_coad_audit_main.py`
- `hest_paad/hb_paad_audit_main.py`

Both import the reference implementation directly:

```python
from stnull.ceiling import r_tech as stnull_rtech
from stnull.spaces import TargetSpace
```

so the `rtech_med_stnull` column in every `*ceiling*.csv` is the shipped
package's own output rather than a re-derivation of it.

Reading discipline: aggregated numbers are approximate, plus or minus 0.02 to
0.05. Sample counts per cohort run from 2 to 24, so no p-values are reported.


Zigan Wang
