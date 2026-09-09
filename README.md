# stnull

**Audit H&E to spatial-gene-expression predictions against zero-pixel null models.**

On HER2ST (8 patients, 36 sections, 737-gene panel), a null model that sees **no
pixels at all** (one scalar per spot, the total UMI count) reaches a median
per-gene Pearson **r = 0.230**. A 2048-dimensional ResNet50 ridge probe on the
same split reaches **r = 0.113**, and loses on **0 of 36 sections**.

<sub>Source: <code>prototype/analysis_out/libonly_panelnorm_check.csv</code>,
median of the <code>libonly</code> and <code>ridge</code> columns over its 36
rows; <code>ridge &gt; libonly</code> in 0 rows.</sub>

The field ranks itself by that same per-gene r. It can be fooled by three things,
and no published paper checks all three:

| confounder | what buys the r | zero-pixel demonstration |
|---|---|---|
| **sequencing depth** | spots with more reads correlate with everything | 0.230 from one scalar |
| **tissue composition** | a class-mean lookup reproduces much of the map | 0.004-0.111 from a 6-class table |
| **protocol choices** | picking genes on the test set, smoothing the truth, splitting by section | 0.117 to 0.269 from gene picking alone |

`stnull` measures how much of *your* reported r each of them buys. It refuses to
print any value without the permutation floor of the **same readout rule** beside
it, and it never transforms your data, trains a model, or reads an image.

- **Sample report (real HER2ST data):** [`examples/sample_report.html`](examples/sample_report.html)
- **API reference:** [`docs/API.md`](docs/API.md)
- **How to word your conclusions:** [`docs/INTERPRETATION.md`](docs/INTERPRETATION.md)
- **Code walkthrough (read the source like a book, with the full variable lineage table):** [`docs/CODE_WALKTHROUGH.md`](docs/CODE_WALKTHROUGH.md)
- **Results on 11 datasets (HER2ST and the ten HEST-Benchmark cohorts):** [`results/`](results/)

Version 0.3.0. Author: Zigan Wang (`zigan.wang@uq.edu.au`).

> **0.3.0 moves numbers.** The 3x3 smoothing lever, the `r_tech` permutation
> floor, the cross-fitted composition nulls, the floor under `r_mean`, and every
> readout under `null_fit="train_only"` all change. See
> [`CHANGELOG.md`](CHANGELOG.md) before comparing a 0.3.0 report with an older
> one.

---

## Install

```
pip install -e .            # numpy, pandas, scipy
pip install -e .[plot]      # + matplotlib, used only for figures inside to_html()
```

Python >= 3.9. `to_html()` writes a single self-contained file (CSS inline,
figures embedded as base64, no external requests). Without matplotlib it degrades
to tables and stays complete.

The library sets no `OMP_NUM_THREADS`; that is the caller's business. If you
share the machine, set it before importing numpy.

---

## 60-second quickstart

Copy-paste and run. It builds a toy dataset whose only real signal is depth, and
shows the audit catching it. Takes about 14 s.

```python
import os
os.environ["OMP_NUM_THREADS"] = "3"          # be polite to the rest of the box
import numpy as np
from stnull import audit

# ---- a toy dataset with a depth confounder baked in ----------------------
rng = np.random.default_rng(0)
G, blocks = 50, []                           # a 50-gene, HEST-style panel
for side in ("S1", "S2", "S3"):
    gx, gy = np.meshgrid(np.arange(20), np.arange(20))
    xy = np.c_[gx.ravel(), gy.ravel()]                    # 400 spots on a lattice
    lib = np.exp(rng.normal(8.5, 0.5, len(xy)))           # sequencing depth
    lab = np.where(xy[:, 0] < 10, "tumour", "stroma")     # 2 tissue classes
    load = rng.normal(size=G)                             # every gene tracks depth
    y = np.log(lib)[:, None] * load + rng.normal(0, 1, (len(xy), G))
    p = np.log(lib)[:, None] * load + rng.normal(0, 3, (len(xy), G))  # "the model"
    blocks.append((np.full(len(xy), side), xy, lib, lab, y, p))

sec, xy, lib, lab, Y, P = (np.concatenate([b[i] for b in blocks]) for i in range(6))

# ---- the audit -----------------------------------------------------------
rep = audit(y_true=Y, y_pred=P,
            space="log1p_cp10k:panel",   # required, no default (see D1)
            section=sec, lib_size=lib, coords=xy, labels=lab,
            n_perm=100, n_boot=100, seed=0)
# top_n was left at its default: (10, 50, 100, 250) trimmed to the panel,
# so this 50-gene run reads out top-10 and top-50 and does not raise.

print(rep.summary())
rep.to_html("audit.html")
```

Here are the first three lines of the output. The two numbers that matter are
glued together on purpose:

```
model median r = 0.0599
  vs strongest zero-pixel null (depth_null) = 0.2944
  model beats that null in 0/3 (0%) of sections
```

**On your own data, the two arguments people get wrong:**

* `top_n`: the default `(10, 50, 100, 250)` is trimmed to the entries that fit
  your panel (with a warning). An **explicit** `top_n` wider than the panel still
  raises under `strict=True`: you asked for a readout that cannot exist.
* `coord_kind`: the default `"grid"` assumes integer array coordinates. Since
  0.2.0, coordinates that fail a lattice check (median spacing >> 1 or rook
  neighbour hit rate < 20%) are detected, warned about, booked in the
  degradation ledger, and handled as `"micron"` (kNN neighbourhoods,
  pitch-estimated block permutation). Passing `coord_kind="micron"` explicitly
  is still the honest way to say what your coordinates are.

Minimum viable call. Everything else degrades gracefully, one report section at
a time:

```python
rep = audit(y_true=Y, y_pred=P, space="log1p_cp10k:panel", section=sec)
```

No `y_pred` at all runs in **nulls-only** mode, which answers "how much r can
zero pixels buy on my data?" before you train anything:

```python
from stnull import nulls_only
rep = nulls_only(y_true=Y, space="log1p_cp10k:panel", section=sec, lib_size=lib)
```

---

## What each number means, and how not to read it

Every metric below is a `Stat`: value, permutation floor of the same readout
rule, bootstrap 95% CI, n, status. `print(stat)` always shows all of them.

### Headline r: `rep.headline.r_median` / `.r_mean`

Per-gene Pearson r, then within-section median **and** mean, then across-section
median **and** mean. All four are printed because the literature mixes the two
aggregations, and a single number is not comparable across papers.

* Read as: how well these predictions track the truth under this protocol.
* Do **not** read as: model quality, until you have looked at
  `headline.strongest_null` on the same line. In the HER2ST reference the
  headline is 0.113 and the strongest null is 0.230.
* Also read `frac_sections_model_beats_null`. A median can hide "the model won
  zero sections".

### `depth_null`: the library-size null

Per-gene least squares of y on `[1, log lib_size]`. Zero pixels, one scalar per
spot.

* Read as: the floor any imaging model has to clear before it is interesting.
* Do **not** read as: proof that depth is a technical artefact. Depth correlates
  with cellularity, which is biology. The point is that *the image is not needed*
  to obtain it.
* If `space="log1p:raw"` the report prints a **`TARGET_CARRIES_DEPTH`** banner:
  in that space a high `depth_null` is a definition, not a finding, and only DG
  is comparable.

### `depth_1d`: image to one number to genes

Take the model's own prediction, compress it to a single predicted depth, expand
back to all genes. On HER2ST this scores **0.141** versus **0.113** for the full
2048-dimensional model.
<sub>Source: <code>prototype/analysis_out/depth1d_vs_full_check.csv</code>,
median of <code>img_depth_1d</code> and <code>img_full_2048</code> over 36 rows.</sub>

* Read as: an estimate of how much of the model's output is one-dimensional.
* Do **not** read as: the model *only* predicts depth. It bounds a rank-1
  summary, not the model.
* Since 0.3.0 this row is **N/A under the default `depth_proxy="observed"`**,
  with that reason printed on the row: there l_hat is the measured log library
  size, so "compress the model to one number" would just be `depth_null` refit
  on another fold split, and `corr(l_hat, log lib)` would be 1.0 by
  construction. Use `depth_proxy="from_pred"` (guarded) or pass `lib_pred=` to
  get the image version, which is what the 0.141 above was computed with.

### `cross_null` / `oracle_null`: the composition nulls

Class-mean lookup from your labels or deconvolution proportions. `cross_null` is
cross-fitted (honest); `oracle_null` reads the class means off the section being
scored.

* `oracle_null` is the **ceiling of any pure composition model**. A model scoring
  below it does not beat a lookup table. It is labelled `UPPER_BOUND` and is
  deliberately barred from being reported as `strongest_null`.
* Do **not** quote `oracle_null` as a competitor's score. It is not achievable
  out of sample.
* Since 0.3.0 the oracle refuses a design as wide as the section: an in-sample
  fit with as many columns as spots interpolates and reads r = 1.0 on pure
  noise, which is not the ceiling of anything. It returns N/A with that reason
  instead.

### `dg` / `crg` / `drg`: the attribution ladder

Residualise **both** y and y_hat on `X`, then correlate:
`r_full` (X = intercept), `DG` (X = [1, l_hat]), `CRG` (X = [1, pi]),
`DRG` (X = [1, l_hat, pi]).

* Read as: *after controlling for depth and composition, **the predictions of
  this model** retain r = ...*
* Do **not** read as: an upper bound on the information in H&E. DRG measures a
  model, not the data. On HER2ST it **rose** with backbone quality (ResNet50
  0.0173 to Phikon 0.0295, on 8/8 sections), which cannot happen if it were a
  property of the data.
* Do **not** read `DRG > floor` as "the model learned morphology". Above-floor
  residual information is statistically distinguishable; whether it is
  morphological needs a separate experiment.
* Since 0.2.0 the default (`depth_proxy="observed"`) matches the oracle ladder
  to +/-0.0011 on all four synthetic scenarios; the opt-in `from_pred` proxy is
  **biased low** (see *Known defects* #1) and is guarded.

### `r_nbr`: the neighbour null

Predict each spot's true value by the mean of its lattice neighbours' **true**
values. No pixels, no model, pure spatial smoothing. On HER2ST it scores a median
0.128 and **beats the image ridge probe on 23 of 36 sections**.
<sub>Source: <code>prototype/analysis_out/ceiling_rtech_rnbr_summary.csv</code>
(<code>median_r_nbr</code>) joined to
<code>libonly_panelnorm_check.csv</code> (<code>ridge</code>).</sub>

* Read as: how much of the score is available from spatial autocorrelation alone.
* Do **not** read as: a competing method. It uses the test labels.

### `by_n`: top-N gene selection

Each top-N readout is paired with the floor of the **same selection rule**
applied to permuted predictions (`perm_selection`). On HER2ST, selecting the top
100 genes on the test set lifts 0.117 to 0.269, of which **0.081 is pure
selection noise**.
<sub>Source: <code>prototype/analysis_out/ridge_protocol_compare.json</code>, keys
<code>aggregate/patient/median_r_all/median</code>,
<code>top100_test_median_r/median</code>,
<code>perm_align_top100_test/median</code>.</sub>

* Read as: what your gene-selection rule is worth. If you have `is_train`,
  `selection.train_selected` gives the legal version.
* Do **not** quote a top-N number without its `perm_selection` floor. That is the
  single failure mode this package exists to stop.

### `r_tech`: the measurement ceiling

`sqrt(reliability)`, where reliability is the Spearman-Brown-corrected
correlation between two independent binomial split-halves of the raw counts
(the "rest of transcriptome" column is thinned too when `lib_size_full` is
given). This bounds the metric itself: no predictor can beat
`corr(truth, y_obs) = sqrt(reliability)`. HER2ST: **0.362-0.647 across the 36
sections, median 0.507**.
<sub>Source: <code>prototype/analysis_out/ceiling_rtech_CORRECTED.csv</code>,
<code>r_tech_correct</code>; the stnull implementation reproduces it to
within 0.007 per section (A1/B2/F3 spot-checked).</sub>

* Read as: "relative to the measurement noise ceiling of this section".
* Do **not** read as: the maximum achievable biological accuracy, and do not
  compute it without `counts`; `stnull` returns N/A rather than substituting a
  Poisson approximation (D8).
* History: 0.1.0 shipped a `multinomial_resample` estimator that correlated the
  observed profile with a replicate resampled **from the observed profile**, so
  the replicate shared the observation's own sampling noise; its zero-signal
  limit was 1/sqrt(2) = 0.707 and it reported ~0.786 on these sections. That
  estimator is removed in 0.2.0, and the old 0.766-0.823 numbers are void.

### `perm_block_vs_free`: the two floors

`perm_free` shuffles spots and destroys spatial autocorrelation, so it is
**anticonservative** on this kind of data. `perm_block` shifts the whole array on
a torus and preserves it. Measured across the 36 HER2ST sections: the all-gene
floor is **0.049 (block)** versus **0.015 (free)**, and the block floor exceeds
the section's own r_full on **5/36 sections** (free: 1/36).
<sub>Source: <code>tests/regress_selection.csv</code>, produced by
<code>tests/regress_her2st.py</code>.</sub>

* Read as: evidence that the free-permutation tests common in this literature are
  too lenient.
* Do **not** silently switch to whichever floor is lower. The report names the
  `floor_kind` on every number for exactly this reason.

---

## Known defects

These are real and reproducible. Read them before you quote a number.

**1. `depth_proxy="from_pred"` attenuates depth-independent signal** (FIXED as
the default in 0.2.0; the option remains, guarded). `drg.fit_depth_proxy` picks
its ridge penalty by maximising `corr(cross-fitted l_hat, log lib)`. When the
data contain no depth signal that criterion is a lottery, and the winning
`l_hat` is often just the dominant direction of `y_pred`, the thing you were
trying to certify. Measured: `|corr(l_hat, true signal)| = 0.750` while
`corr(l_hat, log lib) = 0.057`; recovery ratio 0.64 at G=60, **0.28** at G=200.
Since 0.2.0 the default is `depth_proxy="observed"` (which matched the oracle
ladder to +/-0.0011 in all four synthetic scenarios) and `"from_pred"` carries a
permutation guard on the alpha selection: a proxy indistinguishable from noise
sets the depth rung to N/A instead of deleting the signal.

**2. `null_fit="cv_within_section"` (the default) reports the depth null about
0.028 lower than an in-section fit** (0.2021 versus 0.2298 on HER2ST).
Cross-fitting is the more honest estimator, but it will not reproduce published
in-section numbers. Similarly, `Stat.floor` is the **95th percentile of a block
permutation**, so a floor that is 0.081 under the literature's
free-permutation-median convention prints as 0.155 here. Neither is a bug; both
break naive comparison.
**Mitigation:** rerun with `null_fit="oracle"` and report both.

**3. Non-lattice coordinates degrade silently.** FIXED in 0.2.0: coordinates
that fail the lattice check under `coord_kind="grid"` (median nearest-neighbour
spacing > 2 or rook-neighbour hit rate < 20%) now emit a warning, book a
degradation-ledger entry, and are handled as `"micron"`. Free-permutation
fallbacks and duplicate coordinates are also booked per section. Still check
`rep.run.perm_kind` in the run card.

**4. `top_n` defaults exceed small panels.** FIXED in 0.2.0: the default is now
trimmed to the entries that fit the panel (with a warning). An **explicit**
`top_n` wider than the panel still raises under `strict=True`.

**5. Two-section runs mislabel the CI.** Section-level bootstrap needs at least 3
sections; with 2 it falls back to the first section's spot bootstrap but still
prints `ci_kind=spot (single section)`. Still open in 0.3.0.

**6. DRG point estimates under `from_pred` run about +0.013 high** on the
pure-depth scenario. The decision rule was not fooled (0/60), but the point
estimate alone would suggest a gain that is not there. The 0.2.0 default
(`observed`) shows no such offset (-0.0009 on the same scenario).

**7. Fixed in 0.3.0** (listed so that older reports can be recognised): under
`null_fit="train_only"` the nulls were scored on held-out spots while the model
was scored on all spots, including the ones it may have memorised; under the
default `depth_proxy="observed"` the report printed `corr(l_hat, log lib) = 1.0`
as "depth carried by the model" and `depth_1d` as an image readout, though
neither touched the model; `smooth_matrix` was row-standardised twice, so the
3x3 lever was priced at roughly a quarter of the box mean it names; the `r_tech`
floor was built from one thinning replicate while the value averaged all of
them; `headline.r_mean` carried the floor of the *median* rule; every null
reported the model's gene count (0 in nulls-only mode); an unrecognised
`null_fit` / `depth_proxy` / `perm_kind` silently selected a different estimator
while the run card printed the string you passed; a non-finite `lib_size`,
`coords`, `composition` or `lib_size_full` passed validation and then raised
`LinAlgError` from inside numpy; a null that could not be fitted, and a depth
proxy rejected by its own guard, vanished without a reason.

**Accepted but not implemented:** `n_jobs` is accepted and ignored (runs are
single-process); `spot_ids` is accepted and unused; `perm_floor(rule=...)`
supports only `"all"` and `"top_n"`; `--lang zh` affects the HTML/Markdown files
only, console output is always ASCII English; the split-granularity lever is
measured on a zero-pixel null (stnull never retrains your model) and is therefore
a **lower bound** carrying no permutation floor; the target-space lever is not
priced automatically (run `audit()` twice with different `space` values and diff
the headlines).

---

## CLI

```
stnull audit  --true Y.npy --pred P.npy --obs obs.parquet \
              --space log1p_cp10k:panel --section-col section --lib-col lib_size \
              --coord-cols arr_x,arr_y --coord-kind grid \
              --label-col label --patient-col patient --counts counts.npz \
              --top-n 10,50 --perm 200 --boot 500 \
              --out audit.html --csv-dir out/ --json audit.json
stnull nulls  ...      # same flags without --pred
stnull levers ...      # the protocol lever price list
stnull check  ...      # validate inputs in seconds
stnull cite            # methods paragraph + defaults, for your Methods section
```

Exit codes: `0` ok, `2` strict validation failed, `3` degraded with
`--fail-on-degraded`, `4` inputs unreadable. `--out` dispatches on the suffix
(`.html` / `.md` / `.json`); `--csv-dir` writes `per_section.csv`,
`per_gene.csv`, `ladder.csv`, `claims.csv`.

---

## The decisions this package is opinionated about

Full text with the measured justification for each is in
[`docs/API.md`](docs/API.md); the short version:

**D1** `space` has no default: the same composition null moves 0.007-0.083 r
between a panel-row-sum and a full-transcriptome denominator, larger than most
single-step advances in this field. **D2** stnull never transforms your data.
**D3** `Stat` is the only number container and there is no value without a floor;
the renderers have no `hide_nulls=` switch. **D4** Floors are named
(`perm_free` / `perm_block` / `perm_selection` / `analytic_zero`) because they are
not interchangeable. **D5** The DRG wording is hard-coded and three readings are
permanently listed as `out_of_scope`. **D6** The strongest null is glued to the
headline. **D7** The oracle null is reported but labelled `UPPER_BOUND` and barred
from the headline. **D8** The ceiling is N/A rather than approximated. **D9**
Degradations are booked with a direction of bias. **D10** numpy/pandas/scipy only;
no sklearn; matplotlib optional; no per-gene Python loops.

**Deliberately absent:** model training, feature extraction, image reading, a
composite score, a built-in table of published numbers to compare against, and
any API that collapses the report into one number.

---

## Sample report

[`examples/sample_report.html`](examples/sample_report.html) is a complete run on
real HER2ST data: 812 spots x 100 genes x 3 sections (A1/B1/C1, one per patient),
targets rebuilt from raw counts, predictions from frozen patient-level ridge
folds. Headline r = 0.1051 against a depth+composition null of 0.3169, the model
winning 0/3 sections; r_tech = 0.6649 (split-half sqrt-reliability, 100
HVG-selected genes; the full 737-gene panel sits lower, median 0.507); the
ladder collapses 0.1051 to 0.0037.

Regenerate it with `python examples/her2st_example.py` (about 90 s), which also
writes `examples/out/her2st_audit.{html,md,json}` and `examples/out/csv/*.csv`.
That script is **local**; it reads one machine's prototype directory. Nothing
inside `src/stnull` refers to it or to any project path.

---

## Provenance of every number in this README

| number | file |
|---|---|
| 0.230 / 0.113 / 0-of-36 | `prototype/analysis_out/libonly_panelnorm_check.csv` |
| 0.141 image-to-1D depth | `prototype/analysis_out/depth1d_vs_full_check.csv` |
| r_nbr 0.128 and 23/36 | `prototype/analysis_out/ceiling_rtech_rnbr_summary.csv` |
| r_tech 0.362-0.647, median 0.507 (corrected) | `prototype/analysis_out/ceiling_rtech_CORRECTED.csv` |
| 0.117 to 0.269, floor 0.081 | `prototype/analysis_out/ridge_protocol_compare.json` |
| composition null 0.004-0.111 (8 sections) | `prototype/analysis_out/candor_selftest_composition.csv` |
| DRG 0.0173 / 0.0295 by backbone | `prototype/analysis_out/drg_prior_validation_{resnet50,phikon_p50}.md` |
| block versus free floors, 36 sections | `tests/regress_selection.csv` |
| 64 passed (38 + 26 new in 0.3.0) | `python -m pytest tests/ -q` |
| sample-report figures | `examples/sample_report.html`, `examples/out/csv/ladder.csv` |
| quickstart output | reproduce by running the block above |

The `prototype/analysis_out/` files belong to the research project this package
was extracted from; they are not shipped with `stnull` and nothing in
`src/stnull` reads them.

---

## Citing

`stnull cite` prints a Methods paragraph populated with the parameters of your
actual run. Please paste that rather than paraphrasing, and keep the sentence
stating that DRG is a property of the audited predictions.

```
Wang, Z. stnull 0.3.0. Audit of spatial-expression predictions against
zero-pixel null models. https://github.com/ziganwang-create/stnull
```

## License

MIT. See [`LICENSE`](LICENSE).
