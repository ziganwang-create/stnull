# How to read an stnull report honestly

Author: Zigan Wang. For stnull 0.3.0.

This document is about **wording**. The arithmetic is in
[`API.md`](API.md); what follows is the discipline that keeps a defensible number
from turning into an indefensible sentence.

Contents: [the one rule](#the-one-rule) · [reading a Stat](#reading-a-stat) ·
[the wording table](#the-wording-table) · [permanently out of scope](#permanently-out-of-scope) ·
[comparing across papers](#comparing-across-papers) ·
[when the audit is unflattering](#when-the-audit-is-unflattering) ·
[reporting checklist](#reporting-checklist) ·
[worked example](#worked-example-reading-the-sample-report) ·
[biases that change your wording](#estimator-biases-that-change-your-wording)

---

## The one rule

> **An stnull report is a statement about one set of predictions. It is never a
> statement about H&E, about the task, or about anybody else's model.**

Every failure of interpretation this document guards against is the same
substitution: a property of *these predictions* being promoted to a property of
*the data*. `rep.claims()` enforces the rule in the sentences it generates, and
permanently lists the three most tempting promotions as `out_of_scope`.

---

## Reading a `Stat`

```
0.1051  [perm_block floor 0.0319]  CI[0.0311,0.1086]  n=812/100
```

Read it left to right as a single sentence, never as a number with decorations:

1. **value**: what the readout rule produced.
2. **floor**: what the *same readout rule* produces on permuted data. This is
   the 95th percentile of the null distribution; the note carries its median and
   max. A value below its floor is not a small result, it is no result.
3. **CI**: bootstrap 95%. Check `ci_kind` in the note: `section` resamples
   sections, `spot` resamples spots and is narrower than a patient-level
   bootstrap would be.
4. **n**: spots / genes that actually produced a finite r. If `n_genes` is well
   below your panel size, genes were dropped for zero variance and your effective
   panel is smaller than you think.
5. **status**: anything other than `ok` (`na`, `degraded`, `underpowered`,
   `derived`) changes what the number is allowed to support.

The decision rule used throughout the report is **CI lower bound above the
floor**, not "value above floor". It is what `Stat.above_floor()` returns and
what `ladder.drg_above_floor` records.

---

## The wording table

Each row: a sentence you may write, a sentence you may not, and why.

### Headline r

| | |
|---|---|
| YES | "Under a patient-level split in the `log1p_cp10k:panel` space, the median per-gene Pearson r of our predictions is 0.113 (block-permutation floor 0.049)." |
| NO | "Our model achieves r = 0.113." |
| why | An r without its space, split, aggregation and floor is not a measurement. Four of those five words change the number. |

| | |
|---|---|
| YES | "Our median r (0.113) is below the strongest zero-pixel null we ran (library-size null, 0.230), which the model beats on 0 of 36 sections." |
| NO | "Our model outperforms the baselines." |
| why | `strongest_null` is rendered on the headline row precisely so it cannot be dropped, and `frac_sections_model_beats_null` exists because a median hides "won zero sections". |

### Depth

| | |
|---|---|
| YES | "A null model using only each spot's total UMI count (no pixels) reaches a median r of 0.230 on the same split." |
| NO | "Sequencing depth is a technical artefact that inflates published scores." |
| why | Depth tracks cellularity, which is biology. The finding is that **the image is not needed** to obtain it, not that the signal is fake. |

| | |
|---|---|
| YES | "After residualising both truth and prediction on the model's own predicted depth, the predictions retain r = 0.038 on section C1 (block-permutation floor 0.033, 95% CI [0.006, 0.065])." |
| NO | "Nearly all of our model's performance is a depth artefact." |
| why | `share_depth` is a ratio of two floored estimates, is marked `status="derived"`, and inherits the low bias of the default depth proxy. Quote DG and r_full, and let the reader divide. |

| | |
|---|---|
| YES | "In the `log1p:raw` space the target is itself monotone in depth, so a high depth null is definitional; we therefore report DG as the comparable quantity." |
| NO | "We report r in `log1p:raw` and the depth null is only 0.05, so depth is not an issue here." |
| why | The `TARGET_CARRIES_DEPTH` banner exists because the depth null is uninformative in that space in *either* direction. |

### Composition

| | |
|---|---|
| YES | "A class-mean lookup table over 6 pathologist annotations, cross-fitted within section, reaches r = 0.004-0.111 across 8 sections." |
| NO | "Tissue composition explains most of the field's reported performance." |
| why | Your composition null is only as strong as your labels. Coarse annotations understate it; that is a limitation of the audit, not a finding about the field. |

| | |
|---|---|
| YES | "The oracle composition null (class means read off the section being scored) reaches 0.240 and is an upper bound on any pure composition model." |
| NO | "The composition baseline scores 0.240, higher than our model." |
| why | The oracle is not achievable out of sample. It is labelled `UPPER_BOUND` and barred from `strongest_null` for this reason. |

### The ladder (DG / CRG / DRG)

This is the wording stnull hard-codes, because it is the one people get wrong:

| | |
|---|---|
| YES | "After controlling for depth and composition, **the predictions of this model** retain a median r of 0.0173 across 8 sections; on 0 of those 8 does the bootstrap CI lower bound clear the permutation maximum by the pre-registered margin. This is a property of these predictions, not an upper bound on the information content of H&E." |
| NO | "H&E contains no morphological signal beyond depth and composition." |
| why | DRG measures a model. On HER2ST it **rose** with backbone quality on every one of the 8 annotated sections (median 0.0173 for ResNet50 versus 0.0295 for Phikon), which is impossible if it were a property of the data. (Per-section values: `prototype/analysis_out/drg_prior_validation_resnet50.md` and `..._phikon_p50.md`.) |

| | |
|---|---|
| YES | "The residual information is statistically distinguishable from the permutation null." |
| NO | "The model learned genuine morphological features." |
| why | Above-floor residual correlation says the residual is not noise. Attributing it to morphology requires an experiment that varies morphology. |

| | |
|---|---|
| YES | "Our DRG is near zero, so on this dataset with these controls we could not demonstrate depth- and composition-independent signal from this model." |
| NO | "DRG is near zero, so the task is not solvable from H&E." |
| why | A null result for one predictor is not a null result for the hypothesis class. And with the default depth proxy a real signal can be attenuated by half (see [below](#estimator-biases-that-change-your-wording)). |

### Spatial

| | |
|---|---|
| YES | "A neighbour null that averages the *true* expression of each spot's four lattice neighbours (using no image) reaches a median r of 0.128 and beats our image model on 23 of 36 sections." |
| NO | "Spatial smoothing is a competitive method for expression prediction." |
| why | The neighbour null reads the test labels. It is a diagnostic of how much of the score is spatial autocorrelation, not a method. |

| | |
|---|---|
| YES | "Smoothing the ground truth 3x3 raises our headline by <the delta your own run prints> without retraining; we report the unsmoothed number." |
| NO | (reporting only the smoothed number) |
| why | The lever price list exists to make this visible. Pricing a lever and then pulling it is worse than not measuring it. The 0.027 quoted here before 0.3.0 came from a smoother that gave the centre spot weight 1/2 instead of 1/9; the operator is now the unweighted 3x3 box mean the literature applies, so re-run to get your number. |

| | |
|---|---|
| YES | "We use block permutation, which preserves spatial autocorrelation; the free-permutation floor for the same readout is 0.015 versus 0.049 for block." |
| NO | "Our result is significant (permutation p < 0.05)." |
| why | Unqualified "permutation" usually means free permutation, which is anticonservative here. Across 36 HER2ST sections the block floor exceeds the section's own r_full on 5 sections, versus 1 under free permutation. |

### Selection and levers

| | |
|---|---|
| YES | "Restricting to the 100 best genes selected on the test set raises the median r from 0.117 to 0.269; the same selection rule applied to permuted predictions yields 0.081, so 0.081 of the 0.152 gain is pure selection noise." |
| NO | "On the top 100 genes our model reaches r = 0.269." |
| why | This is the single failure mode the package exists to stop. `by_n` entries always carry a `perm_selection` floor; a top-N number without it is not interpretable. |

| | |
|---|---|
| YES | "Genes were selected on the training patients (`is_train`) and scored on held-out patients." |
| NO | "We report performance on the top-N most predictable genes." (with no statement of where N was chosen) |
| why | `selection.train_selected` is the legal readout; `by_n` is the diagnostic one. Say which you used. |

| | |
|---|---|
| YES | "A zero-pixel null fitted on other sections of the same patient outperforms one fitted on other patients by X; this is a **lower bound** on the leakage a section-level split would grant a real model." |
| NO | "Section-level splitting inflates results by X." |
| why | stnull never retrains your model, so it can only measure the leakage available to a pixel-free null. The lever carries no permutation floor and is marked `status="degraded"`. |

### Ceiling

| | |
|---|---|
| YES | "Our r is 21 % of the measurement-noise ceiling of the same sections (r_tech 0.507 = sqrt of the split-half Spearman-Brown reliability of the raw counts)." |
| NO | "We reach 13 % of the maximum achievable accuracy." |
| why | r_tech bounds *counting* noise only. Biological, registration and annotation limits are additional and are not measured here. |

| | |
|---|---|
| YES | "r_tech is an approximate bound: at this depth (mean 1.6 counts per spot-gene) the Spearman-Brown step is biased upward by roughly 0.06-0.11." |
| NO | "No predictor can exceed 0.507 on these data." |
| why | Spearman-Brown assumes the two half-depth replicates are parallel tests on a linear scale, and log1p is not linear. Below about 5 counts per spot-gene (most Visium data) the estimate runs high, and since 0.3.0 the Stat's own note says so with the measured mean count. Dividing by an inflated ceiling understates your fraction of it, so the direction is safe for the model but the ceiling itself must not be quoted as exact. |

| | |
|---|---|
| YES | "Raw counts were unavailable, so the ceiling is N/A." |
| NO | (substituting a Poisson approximation) |
| why | D8. The approximation biases against the model, which feels safe, but it is still a fabricated number. |

---

## Permanently out of scope

`rep.claims()` always returns these four, whatever your data says. They are in
the report so a reviewer can see that you saw them.

| claim | flag | why |
|---|---|---|
| "H&E carries no information beyond depth and composition." | `out_of_scope` | DRG measures a model, not the data; it rose with backbone quality on 8/8 sections. |
| "DRG close to zero proves the task is impossible." | `out_of_scope` | A null result for one predictor is not a null result for the hypothesis class. |
| "DRG above its floor proves the model learned genuine morphology." | `out_of_scope` | Statistical distinguishability is not mechanism. |
| "Our r is higher than the number reported in paper X." | `not_supported` | Requires identical space, panel, split and aggregation. Use `compare()`. |

An upper bound on what H&E can support would need a different experiment: a
family of models, a capacity argument, or an information-theoretic bound. One
audited model cannot produce one.

---

## Comparing across papers

`compare()` raises `NotComparable` when `space`, gene aggregation, section
aggregation, gene count or section count differ. There is no approximate mode,
because the differences are large enough to reverse conclusions:

* **denominator**: panel row sum versus full transcriptome moves the same
  composition null by 0.007-0.083 r
* **aggregation**: median-over-genes and mean-over-genes are both in common use
  and differ materially on a skewed r distribution; stnull prints both
* **panel**: a fixed 785-gene panel versus per-fold HVGs is a different task
* **split**: section-level versus patient-level is a different task
* **selection**: test-set top-N versus train-set top-N is a different task

If you must place your number beside a published one, state all five and say
explicitly that the comparison is indicative. Do not write "state of the art".

---

## When the audit is unflattering

A model below its strongest null is a publishable finding, provided it is worded
as one.

* **Say**: "Under this protocol our predictions do not exceed a zero-pixel
  library-size null; we report the audit alongside the model."
* **Do not** switch to the readout that wins. Every such switch is priced in
  `selection.lever_price`, and the price list is in the report you are shipping.
* **Do not** drop the floor column. There is no renderer option to do it, on
  purpose.
* **Do not** re-run with `perm_kind="free"` because the floor is lower. If you
  change the floor kind, report both; `perm_block_vs_free` gives you the pair.
* **Do** report `drg_observed` next to `drg`; if they disagree, say so and say
  which one your conclusion rests on.
* **Do** state which inputs were missing. `rep.caveats()` lists them with the
  direction each omission biases the headline, and `--fail-on-degraded` lets a CI
  job refuse an under-specified audit.

---

## Reporting checklist

Nine lines that make a spatial-expression result auditable. `stnull cite`
produces the Methods paragraph; this is what a reader should be able to find.

1. **Target space and denominator**: panel row sum or full transcriptome.
2. **Split granularity**: patient, section, or spot (and how gene selection
   relates to it).
3. **Aggregation**: median or mean over genes, median or mean over sections.
   Report both.
4. **The strongest zero-pixel null you ran**, next to the headline, plus the
   fraction of sections the model beats it on.
5. **The floor of every quoted readout**, named by kind (`perm_free` /
   `perm_block` / `perm_selection`).
6. **Uncertainty**: bootstrap CI and what it resamples (spot, section,
   patient).
7. **Gene selection**: where N was chosen, with the `perm_selection` floor of
   that rule.
8. **Ceiling**: r_tech with its estimator, or an explicit N/A.
9. **The attribution ladder**: r_full, DG, CRG, DRG, each with floor and CI,
   and the sentence that DRG is a property of the predictions.

---

## Worked example: reading the sample report

From [`../examples/sample_report.html`](../examples/sample_report.html)
(812 spots x 100 genes x 3 HER2ST sections; numbers also in
`examples/out/csv/ladder.csv`):

```
model median r = 0.1051
  vs strongest zero-pixel null (depth_comp_null) = 0.3169
  model beats that null in 0/3 (0%) of sections
```

**Line 1 alone would be publishable and misleading.** Lines 2 and 3 are why they
are printed as one block.

Then, in order:

* `depth_null 0.3029` and `depth_1d(img) 0.2876`: most of what the model
  outputs is recoverable from one number per spot.
* `DG 0.0288 [floor 0.0223]`, CI lower bound -0.0294: the CI straddles the
  floor. Controlling depth alone leaves nothing demonstrable. Correct sentence:
  "we could not demonstrate depth-independent signal", not "there is none".
* `cross null 0.1087` versus `oracle null 0.2397 <UPPER_BOUND>`: the model
  (0.1051) is level with an honest lookup table and far below the oracle. The
  oracle is not a competitor and must not be quoted as one.
* `r_nbr 0.2020`, beating the model on 3/3 sections: a large part of the
  achievable score here is spatial autocorrelation, available without pixels.
* `top-10 = 0.3491 [perm_selection floor 0.2336]`: the headline could be
  tripled by gene picking, and two thirds of the 0.349 is what the same
  selection rule produces on permuted predictions.
* `r_tech 0.6649` (split-half, sqrt-reliability): counting noise is not the
  binding constraint.
* Ladder: `0.1051 -> DG 0.0288 -> CRG 0.0435 -> DRG 0.0037`, with
  `drg_above_floor = False` on all three sections.

**A defensible paragraph:**

> On three HER2ST sections our ridge predictions reach a median per-gene r of
> 0.105 (block-permutation floor 0.032). A zero-pixel library-size null reaches
> 0.303 on the same spots and the model does not beat it on any section. After
> residualising both truth and prediction on observed depth and on
> pathologist-annotated composition, the predictions retain a median r of 0.004,
> whose bootstrap CI lower bound does not exceed the permutation floor in any
> section. We therefore do not claim demonstrable depth- and
> composition-independent signal from this model on this dataset; this is a
> statement about these predictions and not about the information content of H&E.

**An indefensible one, from the same run:**

> Our model reaches r = 0.35 on the top 10 genes, approaching the measurement
> ceiling of the assay and demonstrating that H&E encodes spatial expression.

Every clause of that sentence is contradicted by a number in the same report:
the 0.35 has a selection floor of 0.234, 0.35 is 53 % of the 0.665 ceiling rather
than "approaching" it, and the model loses to a null with no pixels.

---

## Estimator biases that change your wording

Two defaults will change what your report says. Both are documented in the README
under *Known defects*; here is what they do to the sentences.

**The `from_pred` depth proxy attenuates real signal.** With
`depth_proxy="from_pred"` (the 0.1.0 default; since 0.2.0 the default is
`"observed"`), a depth-*independent* signal was recovered at only 46 % of its
true strength in synthetic tests (28 % at 200 genes), and the decision rule
missed it on 30 of 60 sections (`tests/synth_results.csv`).

* The 0.2.0 default (`observed`) matched the oracle ladder to +/-0.0011 in all
  four synthetic scenarios, so a DRG from a default run can be read at face
  value (with its floor).
* If you opt into `from_pred`, a **low DRG is weak evidence of absence** --
  write "we could not demonstrate" and report `drg_observed` next to it. Its
  permutation guard now sets the depth rung to N/A when the fitted proxy is
  indistinguishable from noise, instead of silently deleting signal.
* A **high DRG under `from_pred` is not weakened by this bias**; the bias runs
  downward.

**Default null fitting understates the depth null.** With
`null_fit="cv_within_section"` the HER2ST library-size null reads 0.2021 where an
in-section fit reads 0.2298. The cross-fitted number is the more honest one, but
if you are comparing with a published in-section figure, rerun with
`null_fit="oracle"` and report both. The same applies to floors: this package's
default is a block-permutation 95th percentile, while much of the literature
quotes a free-permutation median (0.155 versus 0.081 for the same HER2ST
top-100 readout).

**Two readouts that exist only under a model-fitted depth proxy.** Under the
default `depth_proxy="observed"`, `depth_1d` and `corr(l_hat, log lib)` are
reported as N/A. That is not a missing computation: l_hat *is* the measured log
library size there, so the correlation would be 1.0 by construction and
`depth_1d` would be `depth_null` refitted on another fold split. Before 0.3.0
both printed as if they described the image pipeline. If you want to state how
much depth your model's own output carries, run with `depth_proxy="from_pred"`
(guarded) or supply `lib_pred=`.

All of these belong in a Methods sentence, not a footnote.
