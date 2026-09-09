# Changelog

Author: Zigan Wang.

All notable changes to `stnull` are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html).

Because this is an audit tool, **a change to a default is a breaking change**: it
changes the numbers a user reports. Such changes get a major or minor bump and an
explicit entry here, never a patch.

## [0.3.0] - 2026-09-08

Audit-of-the-auditor release. An adversarial line-by-line review of every module
produced fifteen confirmed defects, all of them fixed here, each pinned by a new
regression test in `tests/test_fixes_030.py` (26 tests). Several of the fixes are
estimator corrections, so **numbers move**; they are listed under *Numbers that
change* below. Author: Zigan Wang.

### Fixed (correctness)

* **`null_fit="train_only"` compared an in-sample model with an out-of-sample
  null.** The zero-pixel nulls were scored on `~is_train` while every model-side
  readout (`r_full`, DG/CRG/DRG, top-N, Moran's I, the smoothing lever and all
  of their floors) was computed on ALL spots of the section, including the ones
  the user flagged as training data. A predictor that simply memorised the train
  spots reported r ~ 0.76 against a null of ~0.03. The model side is now
  restricted to the same held-out spots; `per_section.n_eval` records how many,
  and the degradation ledger states the restriction.
* **`depth_proxy="observed"` published two readouts that measured nothing.**
  `corr(l_hat, log lib)` was printed as "depth carried by the model" while being
  1.0 by construction (l_hat *is* the observed log library size), with an
  analytic-zero floor, so `above_floor()` returned True even for pure-noise
  predictions; `depth_1d` was labelled "image -> 1 number -> genes" while being
  `depth_null` refitted on a second random fold split. Both are now N/A under
  `"observed"` with that reason attached, and are computed only when l_hat comes
  from the model (`"from_pred"` / `"given"`).
* **Option strings were not validated.** `null_fit="orcale"`,
  `depth_proxy="obsevred"` and `perm_kind="blok"` all fell through to a default
  estimator while `RunMeta` printed the string that was passed: an archived
  report naming a protocol the run did not use. All three are now closed sets
  that raise `BadInput`, as is `fit_mode` inside `nulls.linear_null_pred`.
* **`depth_proxy="given"` without `lib_pred`** silently switched the DG and DRG
  rungs off. It now raises under `strict=True` and falls back to `"observed"`
  with a ledger entry otherwise. Supplying `lib_pred` together with
  `"from_pred"` still downgrades the request to `"given"`, but the downgrade
  (and the fact that the `"given"` path carries no permutation guard) is now
  warned and booked.
* **A depth proxy rejected by its own guard was republished.** When the
  `from_pred` permutation guard refused an l_hat, its `best_r` was still stored
  and printed against an analytic-zero floor, and the guard's explanation went to
  a key nothing read. The rejected value is now N/A and the reason reaches the
  ledger.
* **Non-finite inputs crashed deep inside numpy/scipy.** A single NaN in
  `lib_size`, `coords`, `composition` or `lib_pred` passed `check_inputs` and
  came back as `LinAlgError: SVD did not converge` or a `cKDTree` `ValueError`
  naming neither the input nor the section; a NaN in `lib_size_full` became
  INT64_MIN and surfaced as `ValueError: n < 0` from `rng.binomial`. All are now
  validation errors under `strict`, and every consumer degrades to N/A with a
  reason otherwise (`linear_null_pred`, `ladder_rungs`, `perm_plan`, `r_tech`).
* **`lib_size_full` was never validated.** It was absorbed by `**_ignored` in
  `check_inputs`, so a wrong-length vector surfaced as a raw `IndexError` from
  inside the ceiling loop.
* **Sparse `y_true` / `y_pred` crashed `check_inputs`** with `TypeError: ufunc
  'isfinite' not supported`, blocking the sparse path `audit` itself implements.
* **Every null reported the MODEL's gene count** (`n_genes=0` in nulls-only mode,
  and wrong whenever a gene scored for one side but not the other). Each null now
  counts its own finite genes, as `docs/API.md` always claimed.
* **`headline.r_mean` carried the floor of the median rule.** The mean-over-genes
  value was floored by the null of the median-over-genes readout, a different
  distribution whenever the per-gene r spread is skewed (28% apart in a measured
  example). A second null with `how="mean"` is now computed.
* **`share_depth` could explode.** The observed ratio was gated at
  `|r_full| > 1e-6` while its own permutation null used `0.01`; a section at
  r_full = 2e-6 contributed a share of -12499. Both now use the same threshold.
* **A null that could not be fitted vanished silently.** `linear_null_pred`
  returned an all-NaN result with no reason whenever the design was wide relative
  to the section (n < 2(p+1)), so the section dropped out of the across-section
  median with nothing in `warnings`, `na` or the ledger. Every bail-out now
  carries a sentence, and it reaches the report.
* **The composition oracle saturated at r = 1.0 on pure noise** once the design
  had as many columns as the section had spots (measured: K=79 on n=80 gives
  1.0000), and that number was printed as "the ceiling of any pure composition
  model". The oracle now refuses `p >= n-2` and says why.

### Numbers that change

* **The 3x3 smoothing lever.** `smooth_matrix` row-standardised twice: it added
  the identity to an already standardised W, giving the centre spot weight 1/2
  and each neighbour 1/16 instead of the 1/9 box mean its docstring, the API
  reference and the literature all describe. The lever it prices was therefore
  understated severalfold (a factor of 4.6 on a synthetic check). The operator is
  now built unstandardised and standardised once. **Any previously quoted
  `smooth_lever` / `smooth_delta` must be re-run**, including the 0.027 in
  `docs/INTERPRETATION.md`, which has been withdrawn.
* **The `r_tech` permutation floor.** The value averages the raw half-half r over
  `n_rep` replicates before the Spearman-Brown step, but the floor applied
  Spearman-Brown to a single replicate and reused replicate 0 only, which
  inflated it by about 38% on zero-signal data and made it independent of
  `n_rep`. The floor now follows the value's rule exactly, reusing the same
  permutations across replicates. **r_tech point estimates are unchanged
  bit-for-bit** at a fixed seed: the permutations are still drawn at the same
  point in the RNG stream.
* **The cross-fitted nulls.** Each null used to draw its own K-fold partition
  from an advancing RNG, so a difference between two nulls carried fold noise of
  the same order as the effects being read (sd 0.005-0.006 per null,
  peak-to-peak 0.03), and adding or removing a null moved the others. One
  partition is now drawn per section and shared by every null. Depth-null values
  are unchanged at a fixed seed; composition and depth+composition nulls move by
  fold noise.
* **`headline.r_mean.floor`** now comes from the mean rule (see above).
* **Everything under `null_fit="train_only"`**, which is now scored on the
  held-out spots.
* **`depth.depth_1d` and `depth.depth_r_of_pred`** are N/A under the default
  depth proxy instead of reporting a tautology.
* **Null `n_genes`** is each null's own count.
* `r_tech` notes now carry the mean count per spot-gene and, below 5, a measured
  caveat: log1p breaks the parallel-test linearity Spearman-Brown assumes, and
  the estimate runs about +0.06 to +0.11 high in that regime (Poisson simulation
  against a Monte-Carlo Bayes-optimal bound). The ceiling is an approximate bound
  at ST-typical sparsity, which is where this package is used.

### Documentation

* Every module now carries a module docstring (what it is, who calls it, key
  statistical assumptions), section separators, complete public docstrings with
  units, statistical premises and "not applicable when" clauses, and inline
  comments that explain *why* a step is done rather than restating it. That pass
  changed no behaviour.
* Authorship: Zigan Wang, in every source file, `pyproject.toml`, the README and
  the two documents in `docs/`.

### Known issues (reviewed, not fixed here)

1. A two-section run labels its CI `ci_kind=spot (single section)` and quotes the
   first section's spot bootstrap; the second section's spread is dropped.
   Section-level bootstrap needs >= 3 sections (README defect #5).
2. `boot_ci(groups=...)` is dead code: `patient=` never widens the CIs, although
   the ledger entry for a missing `patient` implies that supplying it would. The
   bootstrap is spot/section level unconditionally.
3. `depth_proxy="from_pred"` does not honour `is_train`: l_hat is cross-fitted
   within the section regardless of `null_fit`, so under `train_only` the
   standardisation, the kernel and the alpha search still see the scored spots.
4. The one-hot class vocabulary and the `min_class_spots` merge are computed over
   the full section, including held-out spots, even under `train_only`.
5. `perm_null_of_rung` permutes an already-residualised y_hat without
   re-projecting, which attenuates the null slightly (under 1% at n=300, q=5;
   it grows with the number of composition columns).
6. Under `strict=False`, sections below `min_spots` are dropped from
   `per_section` entirely, though `docs/API.md` says they are kept there.
7. `neighbour_matrix` keeps only the last spot per rounded grid cell, so a
   duplicated cell removes the earlier spot from every neighbour list;
   `perm_plan` already detects duplicates and refuses, `neighbour_matrix` does
   not.
8. `fit_depth_proxy` re-factorises the same fold kernel for every alpha and every
   guard permutation (930 dense solves per section). Caching one `eigh` per fold
   is bit-identical and about 100x faster; `depth_proxy="from_pred"` is slow on
   large sections until then.
9. `r_tech` point estimates still depend on `n_perm`, because the permutations
   are drawn from the same stream as the thinning replicates.
10. `neighbour_matrix` ignores `mode` in the micron branch (kNN, not rook/queen)
    and raises on an empty coordinate array there.

### Validation at release

* `python -m pytest tests/ -q` -> **64 passed** (38 existing + 26 new), 1
  unrelated pandas/bottleneck warning.
* The synthetic ground-truth suite is unchanged and still passes at its 0.2.0
  tolerances: `drg_observed` within 0.01 of the oracle ladder in every scenario,
  no false positives on pure depth / composition / noise.
* Model-side point estimates on the toy fixture are bit-identical to 0.2.0
  (r_median, DG, CRG, DRG, drg_observed, top-N, Moran's I, the leakage lever);
  the moved numbers are exactly those listed above.

## [0.2.0] - 2026-08-24

Audit-critical release: two estimator corrections that change reported numbers,
plus the closure of every "Known defect" that could silently misreport. Because
defaults changed, numbers from 0.1.0 runs are not comparable without re-running.

### Changed (breaking: defaults and estimators)

* **`r_tech` estimator replaced.** 0.1.0's preferred `multinomial_resample`
  correlated the observed profile with a replicate resampled *from the observed
  profile*, sharing the observation's own sampling noise: its zero-signal limit
  was 1/sqrt(2) = 0.707 (measured 0.67 on iid-Poisson data) and it reported
  ~0.786 on HER2ST whose corrected ceiling is 0.507. The estimator is now an
  independent binomial split-half of the counts (thinning the
  'rest-of-transcriptome' column too when `lib_size_full` is given), Spearman-
  Brown corrected, with **r_tech = sqrt(reliability)**, the bound on
  corr(prediction, noisy y). Zero-signal readout ~0.08 and below its own
  permutation floor (guarded by a test); reproduces the frozen
  `ceiling_rtech_CORRECTED.csv` HER2ST values to within 0.007 per section.
  Method strings are now `split_half_full` / `split_half_panel`.
* **`depth_proxy` default is `"observed"`** (was `"from_pred"`). The old default
  attenuated depth-independent signal (46% recovery at 60 genes, 28% at 200:
  the documented false negative). The default now matches the oracle ladder to
  +/-0.0011 across the depth/signal/mixed/noise scenarios (6 seeds, default
  config: oracle DRG 0.0082/0.1367/0.0892/0.0064 vs tool 0.0073/0.1359/
  0.0903/0.0054). `"from_pred"` remains available and now carries a permutation
  guard on its alpha selection: a proxy indistinguishable from noise sets the
  depth rung to N/A instead of absorbing the model's signal. The CLI default
  was aligned.
* **`top_n` default adapts to the panel.** `top_n=None` (the new default) means
  "(10, 50, 100, 250) trimmed to entries <= n_genes", with a warning when
  trimming happens, so a 50-gene HEST panel runs out of the box. An *explicit*
  `top_n` wider than the panel still raises under `strict=True`. CLI `--top-n`
  default aligned.

### Fixed

* **Micron coordinates no longer degrade silently** (0.1.0 defect #3).
  `coord_kind="grid"` now runs a lattice check (median nearest-neighbour
  spacing > 2 or rook-neighbour hit rate < 20% fails); coordinates that fail
  are warned about, booked in the degradation ledger, and handled as
  `"micron"`. Block-permutation fallbacks to free are booked per section, and
  duplicate coordinates within a section are detected and booked.
* **kNN neighbourhoods exclude self explicitly.** With coincident spots the
  KD-tree tie-breaking could leave a spot as its own neighbour (self-weight
  1/k), polluting `r_nbr`/Moran; self is now dropped by index, not position.
* **Torus block permutation can no longer return the identity.** On a
  single-column (W=1) grid, dx=0 with dy drawn from [0, H) produced identity
  permutations (measured 6/200), deflating the floor; (dx, dy) = (0, 0) is now
  impossible, a 1x1 grid falls through to free permutation, and duplicate grid
  cells (where a "shift" is not a permutation) also fall through, and the ledger
  discloses it.
* **`null_fit="oracle"` is marked on the numbers themselves.** depth/cross/
  depth+comp nulls fitted in-sample carry an `UPPER_BOUND` note, are excluded
  from `headline.strongest_null` (same rule as the composition oracle, D7), and
  the run is booked in the degradation ledger.
* **Degradation ledger no longer misstates `null_fit`.** With `is_train`
  missing the ledger used to claim "nulls are cross-fitted within section
  (null_fit=oracle)", which was false. The wording now follows the actual
  fit mode, and `null_fit="train_only"` without `is_train` raises a clear
  `BadInput` (strict) instead of a deep `ValueError`.
* **`n_perm=0` returns Stats with `status='na'`** (value computed, floor absent,
  note says why) instead of crashing in `np.stack`.
* **The per-section ceiling loop skips-and-counts instead of `break`.** One
  section without an `r_tech` estimate no longer silently drops every later
  section; skipped sections are counted in the note and warned about, and the
  n_spots weighting stays aligned to the surviving sections.

### Validation at release

* Synthetic, default configuration, 6 seeds x 4 scenarios: oracle DRG
  0.0082/0.1367/0.0892/0.0064 (depth/signal/mixed/noise) vs tool default DRG
  0.0073/0.1359/0.0903/0.0054, |diff| <= 0.0011, and the drg_observed
  tolerance in `tests/test_synthetic.py` was tightened from 0.02 to 0.01.
* Ceiling: zero-signal (iid Poisson) r_tech = 0.083, below its own permutation
  floor 0.195 (the removed estimator read 0.671 on the same data); HER2ST
  spot-checks vs the frozen corrected values: A1 0.4524/0.4552,
  B2 0.6417/0.6469, F3 0.3551/0.3620.
* README quickstart re-run verbatim on the 50-gene panel: headline 0.0599 vs
  depth null 0.2944, model beats it in 0/3 sections, default top_n trims to
  (10, 50).
* Test suite: 38 passed (`tests/final_pytest.log`), including 19 new
  regression tests pinning every fix in this release.

### Documentation

* README quickstart now runs verbatim on a 50-gene panel (numbers refreshed);
  Known defects #1/#3/#4 marked fixed with the new behaviour; r_tech section
  rewritten for the corrected estimator (HER2ST 0.362-0.647, median 0.507,
  source `ceiling_rtech_CORRECTED.csv`); sample report regenerated
  (r_tech 0.6649 on the 100-HVG example; ladder ends at DRG 0.0037 under the
  new default). `docs/INTERPRETATION.md` worked example updated to the new
  report.

## [0.1.0] - 2026-08-21

First release.

### Added

* `audit()`: the orchestrator. Per-gene Pearson r computed within a section,
  aggregated over genes (median and mean) and over sections (median and mean).
  `space` is required and has no default (decision D1); `section` is the only
  grouping that cannot be omitted; `y_pred` is optional and its absence switches
  the run to nulls-only mode.
* `Stat`: the single frozen number container. Value, permutation floor of the
  same readout rule, bootstrap 95% CI, n, status and note, printed together by
  `__str__`. No renderer can hide the floor column.
* Zero-pixel null models: `depth_null` (least squares on `[1, log lib]`),
  `cross_null` and `oracle_null` (class-mean lookup, cross-fitted and in-section),
  `depth_comp_null`, and `r_nbr` (mean of the neighbours' true expression).
  `headline.strongest_null` is rendered on the headline row and
  `frac_sections_model_beats_null` beside it.
* Attribution ladder `r_full -> DG -> CRG -> DRG` with a conservative
  `drg_observed` variant, per section, each rung carrying a floor and a bootstrap
  CI. Residualisation uses an SVD orthonormal basis and is rank-deficiency safe.
* Four named permutation floors: `perm_free`, `perm_block` (torus shift on the
  lattice, default when `coords` are given), `perm_selection` (the same top-N rule
  applied to permuted r), `analytic_zero`. Residualised rungs permute only the
  residual spot order with the design matrix held fixed.
* Protocol lever price list: test-set gene selection, ground-truth smoothing,
  split-granularity leakage (measured on a zero-pixel null), target-space change
  (N/A by design D2).
* Measurement ceiling `r_tech` by exact multinomial resampling (counts plus
  full-transcriptome depth) or binomial split-half with Spearman-Brown
  correction; N/A rather than a Poisson approximation when counts are absent.
* Report sections 0-10 with a degradation ledger that records the direction of
  bias for every missing input; `claims()` with fixed template sentences and
  four permanently non-supported readings.
* Renderers: `summary()` (ASCII), `to_html()` (single self-contained file),
  `to_markdown()`, `to_json()`, `to_csv_dir()`, `plot()`.
* Helper entry points `nulls_only()`, `ladder()`, `perm_floor()`, `compare()`
  (raises `NotComparable`), `check_inputs()`, `cite_text()`.
* CLI `stnull audit | nulls | levers | check | cite` with exit codes
  0 / 2 / 3 / 4.
* Dependencies: numpy, pandas, scipy. matplotlib optional (figures only);
  sklearn not used.
* Documentation: `README.md`, `docs/API.md`, `docs/INTERPRETATION.md`, a real
  HER2ST report at `examples/sample_report.html`.

### Fixed during development

Both of these were reproducibility bugs found by the test suite, and both are
worth checking for in any similar codebase:

* **Per-section RNG seeds derived from `hash()`** made results drift between
  processes, because CPython salts string hashing per process. Point estimates
  moved by up to 0.03 between two runs of the same script. Seeds are now derived
  from `sha1(section_name)`; `tests/test_smoke.py::test_seed_is_process_independent`
  runs two subprocesses under different `PYTHONHASHSEED` values and compares.
* **Point estimates moved when `n_perm` changed**, because permutation,
  null cross-fitting and bootstrap shared one RNG stream, so altering the
  permutation count also altered the K-fold splits. Each section now spawns three
  independent streams (perm / fit / boot);
  `test_point_estimates_do_not_move_with_n_perm` guards it.

### Known issues

Listed in full under *Known defects* in the README. In short:

1. `depth_proxy="from_pred"` (the default) attenuates depth-independent signal
   (46 % recovery in synthetic tests at 60 genes, 28 % at 200 genes) because the
   ridge penalty is chosen by maximising `corr(l_hat, log lib)`, which is a
   lottery when no depth signal exists. Use `lib_pred=` or read `drg_observed`.
2. `null_fit="cv_within_section"` (the default) reports the depth null about
   0.028 lower than an in-section fit, and `Stat.floor` is a block-permutation
   95th percentile where much of the literature quotes a free-permutation median.
   Both break naive comparison with published numbers.
3. Micron coordinates passed with the default `coord_kind="grid"` degrade
   silently: `r_nbr` and Moran's I go N/A and floors fall back to `perm_free`,
   without a warning or a ledger entry. This violates decision D9.
4. The default `top_n=(10, 50, 100, 250)` raises under `strict=True` on panels
   smaller than 250 genes, which includes common HEST configurations.
5. Two-section runs label the CI `ci_kind=spot (single section)` while actually
   using the first section's spot bootstrap.
6. DRG point estimates run about +0.013 high in the pure-depth scenario; the
   decision rule was not fooled (0/60 sections), but the point estimate alone
   overstates.

Accepted and not implemented: `n_jobs` (single-process only), `spot_ids`
(unused), `perm_floor(rule=...)` beyond `"all"` and `"top_n"`, `--lang zh` for
console output, patient-level bootstrap in the headline, per-number `degraded`
marking under `strict=False`, and automatic pricing of the target-space lever.

### Validation at release

* Synthetic: 20 seeds x 3 sections x 60 genes across 6 scenarios, judged against
  an oracle ladder computed from the true latent depth and true labels. No false
  positives (0/60 flagged sections on pure depth, pure composition and pure
  noise). One false negative, issue 1 above. Data: `tests/synth_results.csv`.
* HER2ST regression against independently computed frozen results: the ResNet50
  and Phikon ladders reproduce to 0.0000 per section (32 values), `r_nbr`
  reproduces to 0.00000 across 36 sections, `r_tech` differs by +0.0009 and the
  top-100 selection floor by +0.0012. Data: `tests/regress_her2st.py` and the
  `tests/regress_*.csv` files.
* Test suite: 19 passed (`tests/final_pytest.log`).

### Not timed

The full 13,580 spots x 737 genes x 36 sections case has never been run
end to end. The measured 812 x 100 x 3 case takes about 90 s at
`n_perm=200, n_boot=200, n_rep_ceiling=10`; extrapolation suggests the full case
exceeds the ten-minute design target, dominated by the ladder bootstrap. Start
with `n_boot=100`.

[0.2.0]: https://github.com/ziganwang-create/stnull/releases/tag/v0.2.0
[0.1.0]: https://github.com/ziganwang-create/stnull/releases/tag/v0.1.0
