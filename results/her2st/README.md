# HER2ST -- zero-pixel audit results

HER2ST breast cancer spatial transcriptomics, 36 sections from 8 patients
(legacy ST arrays). This is the **discovery cohort**: the zero-pixel effect was
found here first, and the HEST-Benchmark cohorts in the sibling directories are
the out-of-sample replications. Read the HER2ST numbers as in-sample and the
HEST numbers as the test.

| | |
|---|---|
| Organ / platform | breast / legacy ST |
| Sections, patients | 36, 8 |
| Split | patient-level leave-one-out, 8 folds (ours, not the paper's) |
| Leakage verdict | no leak under our protocol |
| sd(log lib) range | 0.389 to 1.428 |
| Zero-pixel library null | median r 0.230 (737 HVG, cp10k caliber) |
| 2048-d image ridge, same protocol | median r 0.113 |
| Sections where the null wins | 36 / 36 |
| Technical ceiling r_tech | 0.51 median (0.36 to 0.65) |
| H2 zero-biology sim >= real | 34 / 36 |

Row source: `../hb_MASTER_TABLE.csv`, cohort `HER2ST`.

**One caution carried over from the analysis notes.** The zero-pixel library
null beating the image ridge does *not* mean the null recovers cell density. The
zero-biology simulation in `libnull_zerobiology_sim.csv` scores **higher** than
the real data (0.2658 vs 0.2298), which says the effect is mostly what
`log1p` of a counts-per-10k target does to sparse counts, i.e. an artifact of
the measurement convention rather than a biological signal the null has found.

## Files

### H1 -- the zero-pixel library-size null

| file | what it holds |
|---|---|
| `lib_structure_audit.csv` | Per section: `sd_loglib`, `geomean_lib`, `moran_I_loglib` (6-NN, row-normalised), and the per-section median r of two protocols -- `ridge_median_r` (full 2048-d ResNet50 ridge) and `libonly_median_r` (the library-only null). This is the file behind the claim that sd(log lib) alone tracks achievable r at 0.96. |
| `lib_structure_audit.md` | The written reading of that file: cross-section correlations, top and bottom five sections, per-patient means. |
| `libonly_panelnorm_check.csv` | `libonly` vs `ridge` per section under the panel-normalised (cp10k over the 737-gene panel) target, and their difference. |
| `libonly_independent_check.csv` | The same comparison with the null's depth variable taken from an independent source, so the null cannot be reading the target's own row sum. |
| `pilot_null_pergene_r.csv` | Per gene x section r for three **composition** nulls: `cross6` (cell-type proportions transferred from other patients), `binary` (tumour / not), `oracle` (same-section proportions). Zero-pixel, but composition rather than depth. |
| `pilot_null_summary_by_section.csv` | Per section medians of those three, plus `q90_r_crosspatient` and the fraction of genes above r = 0.3. |
| `pilot_null_winnerscurse.csv` | What top-N gene selection buys: `median_all_HVG`, `top100_winnerscurse` (genes picked on the test set), `top100_shuffle_floor_mean/max` (the same selection applied to permuted labels -- pure selection noise), `top100_honest_preselected` (genes picked on training patients only). On A1, 0.2425 against a 0.2384 shuffle floor: almost the entire top-100 number is selection noise. |
| `pilot_null_summary.json`, `pilot_compositional_null.py` | Summary of the pilot run and the script that produced it. |

### H2 -- zero-biology simulation

| file | what it holds |
|---|---|
| `libnull_zerobiology_sim.csv` | Per section: `real` (the null's score on real counts), `sim` (the same estimator on counts simulated with no spatial biology at all, only the real library sizes), and `sd_loglib`. `sim >= real` in 34 of 36 sections. |
| `libnull_zerobiology_sim.py` | The simulator. |

### Depth and the DRG ladder

`DRG` = depth-and-composition-residual gain: the part of a model's correlation
that survives after both the depth read-out and the cell-composition read-out
have been projected out. `DG` controls depth only, `CRG` composition only,
`r_full` controls nothing.

| file | what it holds |
|---|---|
| `depth1d_vs_full_check.csv` | Per section: `depth_r` (how well 2048-d image features predict `log(lib_size)`), `img_depth_1d` (expression predicted through that single depth number), `img_full_2048` (the full-feature ridge), and `gain` = the difference. The gain is small, which is the finding: most of what the image contributes routes through one scalar. |
| `drg_prior_validation_<backbone>.csv` | The full ladder for one image backbone, one row per section. Key columns: `n_spots`, `n_labeled`, `depth_alpha` (ridge alpha chosen for the depth read-out), `depth_r_patient`, then `r_full` / `dg` / `crg` / `drg` each in an `_all` (all spots) and `_lab` (annotated spots) variant with `_med` and `_mean`; `drg_ci_lo`/`drg_ci_hi` (500-sample bootstrap), `perm_p95`/`perm_max` (200 permutations -- the null of the DRG rule itself), and `g4_pass_p95`/`g4_pass_max` (whether DRG clears its own permutation floor, the pre-registered G4 gate). |
| `drg_prior_validation_<backbone>.md` | The same run written out: an `l_hat` self-check against `depth1d_vs_full_check.csv`, the three-rung table, and the gate verdict. Generated by `prototype/metric_drg.py`, seed 20260820, 200 permutations, 500 bootstrap. |
| `drg_resnet50_vs_phikon.csv` | ResNet50 (`_R` suffix) and Phikon (`_P` suffix) ladders side by side, one row per section. |
| `candor_drg.csv` | The same ladder computed inside the CANDOR method track, one row per `tag` x `section` x `model`, including the `depth` model row. |
| `drg_truelib_toroidal_v2.csv` | DRG with the **true** library size substituted for the estimated one, plus a stronger null: `iid_p95`/`iid_max` are the i.i.d. permutation floor, and `tor_p95`/`tor_max`/`tor_n` are a **toroidal shift** null that preserves spatial autocorrelation, which the i.i.d. permutation destroys. `drg_truelib_poly3` allows a cubic rather than linear depth read-out. |
| `drg_truelib_toroidal_v2_all4.csv`, `_midnight.csv` | The same for four backbones and for Midnight-12k. |
| `drg_truelib_toroidal_v2.py`, `_midnight.py` | The scripts. |

### H4 -- the technical ceiling

| file | what it holds |
|---|---|
| `ceiling_rtech_CORRECTED.csv` | Per section: `reliability`, `r_tech_correct`, `r_tech_old_biased`, `diff`. The corrected estimator drops the ceiling from about 0.78 to about 0.46 on A1 -- the earlier value applied a Spearman-Brown step that does not belong there. Use the `_correct` column. |
| `ceiling_rtech_rnbr_summary.csv` | Per section: `median_r_tech` and `median_r_nbr` with the gene counts each was computed over. Note that `median_r_tech` here is the **old, biased** value; `ceiling_rtech_CORRECTED.csv` supersedes it. |
| `ceiling_rtech_rnbr.csv` | The per-gene detail behind that summary (about 1.3 MB). |
| `rtech_splithalf_v2.csv` | An independent split-half re-derivation: `r_splithalf_med` (raw half-versus-half) and `r_sb_full` (the Spearman-Brown extrapolation to full depth). Produced by `rtech_splithalf_v2.py`. |

### Protocol levers

The "levers" are the reporting choices that move a headline number without
touching the model at all.

| file | what it holds |
|---|---|
| `protocol_levers.csv` | One row per section x `panel` x `target` x `smooth`. `panel` = which gene panel (`hvg737_train` selects inside the training patients only); `target` = the caliber (`cp10k`, `log1p_raw`, ...); `smooth` = whether the truth was spatially smoothed before scoring. Readouts: `med_all` / `mean_all` over all genes, `heg50_mean` (top 50 highest-expressed), `top50_testpcc` / `top100_testpcc` (genes ranked on the **test** set -- the winner's-curse readout). On A1 one set of predictions reads out as 0.018 or as 0.423 depending only on which panel, target, smoothing and readout rule is quoted. |
| `protocol_reporting_variants.csv` | The narrower version of that sweep: median, mean, HEG-50, HEG-100, test-top-50, test-top-100 per section. |

## How to read the levers table

Every number in `protocol_levers.csv` comes from **one** set of predictions.
Nothing about the model changes across its rows. What varies is the panel, the
target caliber, whether the truth was smoothed, and which readout rule is
quoted. The spread that opens up is the price list for reporting choices:

- across all four levers, the median section spans **0.407** (min 0.199, max 0.602
  over the 36 sections);
- holding panel, target and smoothing fixed and varying only the **readout rule**
  -- median over genes versus mean versus HEG-50 versus test-selected top-N --
  the median section still spans **0.256**.

Both are wider than most published gaps between competing architectures. That is
the argument for reporting a headline together with the permutation floor of its
own rule, which is what the `stnull` package does.

## Provenance

| output | script |
|---|---|
| `lib_structure_audit.*`, `libonly_panelnorm_check.csv` | `prototype/lib_structure_audit.py` |
| `libonly_independent_check.csv`, `depth1d_vs_full_check.csv`, `drg_prior_validation_*` | `prototype/metric_drg.py` |
| `protocol_levers.csv`, `protocol_reporting_variants.csv` | `prototype/analysis_out/eval_specs.py` |
| `ceiling_rtech_rnbr*.csv` | `prototype/ceiling.py` |
| `ceiling_rtech_CORRECTED.csv` | `prototype/make_figures_v3.py` |
| `rtech_splithalf_v2.csv` | `rtech_splithalf_v2.py` (shipped here) |
| `libnull_zerobiology_sim.csv` | `libnull_zerobiology_sim.py` (shipped here) |
| `drg_truelib_toroidal_v2*.csv` | `drg_truelib_toroidal_v2*.py` (shipped here) |
| `candor_drg.csv` | `prototype/candor.py` + `prototype/candor_report.py` |
| `pilot_*` | `pilot_compositional_null.py` (shipped here), originally `pilot/compositional_null.py` |

The `pilot_` files are the only ones that did not come from
`prototype/analysis_out/`; they are copied from `pilot/` because they carry the
composition-null side of the HER2ST result, which nothing in `analysis_out`
duplicates.

Reading discipline: all figures are approximate, plus or minus 0.02 to 0.05.

Zigan Wang
