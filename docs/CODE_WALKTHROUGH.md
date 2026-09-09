# stnull code walkthrough

Author: Zigan Wang (`zigan.wang@uq.edu.au`). Written against version 0.3.0.

This document is the reading companion for the source tree under
`src/stnull/`. Every function body in the package carries numbered block
comments in a fixed WHAT / WHY / WHERE format (what the block does, why it is
done that way statistically, where its inputs come from and where its outputs
go). This file gives you the map: the one-page story, a glossary, a guided
tour of each module keyed to those block comments by line number, the full
variable lineage table, and three reading paths of different lengths.

Two conventions used throughout:

- `[L##]` is a lineage id. Section 4 defines all 56 of them. The same ids
  appear inside the source comments, so
  `grep -rn "\[L05\]" src/stnull/` lists every place a given variable is
  born or consumed. Lineage rows cite `file:function` anchors rather than
  raw line numbers, because comments shift lines and function names do not.
- Line numbers in the module tours (Section 3) were read from the v0.3.0
  sources as annotated; they mark where a numbered section comment starts.
  If they drift after an edit, the `[L##]` tags and function names still hold.

Shape shorthand: `n` = spots of one tissue section, `n_all` = spots of the
whole run, `g` = genes, `K` = composition classes, `B` = permutation draws.

---

## 1. The story in one page

Somewhere upstream, a model looked at small square crops of an H&E-stained
tissue photograph (one crop per measured location, called a spot) and
predicted, for each spot, the expression level of a few hundred genes. The
paper then reported a correlation, Pearson r, between predicted and measured
expression, and the number looked respectable, say 0.23.

`stnull` never sees the image. It takes only four things: the measured
expression matrix (`y_true`, one row per spot, one column per gene), the
model's predictions (`y_pred`, same shape), a declaration of which
normalisation the numbers are in (`space`), and some per-spot bookkeeping
(which tissue section each spot belongs to, its total molecule count, its
coordinates, its tissue class). From these it asks one question: how much of
that 0.23 would a competitor get who is not allowed to look at any pixel?

An exam analogy, used once and then dropped: if a student scores 62%, you
want to know that a coin-flipper scores 48% on the same paper before you
praise the student. `stnull` builds several such coin-flippers, each blind
to the image but allowed one piece of bookkeeping:

- the **depth null** predicts every gene from the spot's total molecule
  count alone (`nulls.depth_null`);
- the **composition null** predicts every gene as the average of its tissue
  class (`nulls.composition_null`);
- the **neighbour null** predicts each spot as the mean of the measured
  values at its physical neighbours (`nulls.neighbour_null`);
- the **permutation floor** shuffles the spot order and re-scores, giving
  the distribution of r under pure chance (`core.perm_plan` plus the
  `_perm_*` helpers in `audit.py`).

Every null is scored with exactly the same rule as the model: per-gene
Pearson r, median over genes, median over sections. The result of the run is
an `AuditReport` in which no number travels alone: each readout is a `Stat`
triple of (value, floor, confidence interval), where the floor is the 95th
percentile of that same readout under permutation. Above the nulls sits the
**attribution ladder** (`drg.ladder_rungs`): the model's r re-computed after
depth, composition, or both are mathematically removed from both sides,
which says how much of the score survives once the bookkeeping is credited
to the bookkeeping. Below everything sits the **measurement ceiling**
(`ceiling.r_tech`): how well the assay agrees with itself when you split
each spot's molecules into two random halves, which bounds what any model
could honestly reach.

The journey of one data point, end to end: a raw count enters as a cell of
`counts` or, normalised, as a cell of `y_true`; the coercion layer in
`audit.audit()` turns it into row i, column j of the dense matrix `Y`
[L01]; the per-section loop slices it into `Y[m]` and hands it to
`_section_block` as `y`; there it is correlated against the model
(`core.colcorr`), against every null (`nulls.py`), residualised on the
ladder designs (`drg.py`), shuffled B times for the floors, and resampled
for the intervals; the per-gene r that contains it is collapsed to a section
median (`core.agg`), combined across sections (`audit._combine`), frozen
into a `Stat` (`core.make_stat`), and finally printed by `report.py` as one
cell of a table with its floor beside it. That path, value in, floored Stat
out, is the entire package.

---

## 2. Glossary

Terms in the order you will meet them. Each entry: plain words first, then a
small numeric example.

**Spot.** One measured location on a tissue section, a circle of roughly
55 to 100 micrometres holding a handful of cells. Example: a HER2ST section
has about 300 to 700 spots on a hexagonal grid.

**Section.** One tissue slice on one glass slide. All correlations in
`stnull` are computed within a section, then aggregated across sections.
Example: 36 sections from 8 patients means 36 within-section medians and one
across-section median.

**UMI (unique molecular identifier).** A barcode attached to each captured
RNA molecule so duplicates from PCR copying are counted once. "Counts" in
this package always means UMI counts. Example: gene ERBB2 in spot 17 has a
count of 4 means 4 distinct molecules were captured there.

**Library size (sequencing depth).** The total UMI count of one spot across
the whole transcriptome. It varies for technical reasons (tissue thickness,
permeabilisation) and drags many genes up and down together. Example: spot A
has 12,000 total UMIs, spot B has 3,000; after CP10K normalisation a gene
with the same true concentration still tends to differ between them because
of sparsity. `stnull` treats depth as the largest known confounder [L04],
[L05].

**Gene panel.** The subset of genes actually scored, often a few hundred
highly variable genes out of about 20,000. The panel sum is not the library
size: a panel of 100 genes may hold only 5% of a spot's molecules, which is
why `check_inputs` traps `lib_size` values that look like panel sums
(audit.py:278).

**Space (target normalisation).** The transform the expression numbers are
in: raw counts, log1p of counts-per-10k with a panel or full denominator,
ST-Net-style median scaling, or z-scored log values. Declared explicitly via
`spaces.TargetSpace` [L12]; there is no default because the ceiling and the
depth interpretation change with it. Example: `"log1p_cp10k:panel"` means
`log(1 + 10000 * count / panel_sum)`.

**Pearson r.** The standard linear correlation between two lists of numbers,
between -1 and 1. Computed per gene across the spots of one section by
`core.colcorr`. Example: if predicted and measured values of GNAS rise and
fall together across 400 spots, r might be 0.31; if unrelated, r hovers near
0 with a spread of about 1/sqrt(400) = 0.05.

**Permutation floor.** The chance level of a readout, obtained by shuffling
which spot gets which prediction B times (default 1000), recomputing the
identical readout each time, and taking the 95th percentile of the B
results. A value below its floor is indistinguishable from luck. Example: a
median-r readout of 0.06 with a floor of 0.08 clears nothing. Built by
`core.perm_plan` [L19] and `core.make_stat` [L39].

**Block (torus-shift) permutation.** A shuffle that respects spatial
structure: instead of scrambling spots independently, the whole grid of
spots is shifted by a random offset with wrap-around, so neighbours stay
neighbours. This gives an honest floor when the data are spatially smooth; a
free shuffle destroys the smoothness and makes chance look weaker than it
is. `stnull` prints both floors side by side (`perm_block_vs_free` [L37]).

**Cross-fitting.** Fitting a null on some spots and scoring it on others, in
k folds (here k=5, `core.kfold_indices` [L20]), so the null cannot memorise
the spots it is judged on. Example: the composition null learns class means
on folds 1 to 4 and predicts fold 5, rotating five times; the "oracle"
variant that fits and scores on the same spots is reported only as an upper
bound.

**Residualisation (partialling out).** Removing the part of a variable that
a set of controls can linearly explain, keeping the remainder. `stnull`
residualises both truth and prediction on the same design before
correlating (`core.resid`, `core.partial_r`), so the resulting r credits
neither side with the controls. Example: if log depth explains half of a
gene's variance on both sides, the DG rung's r drops accordingly.

**Attribution ladder.** Four readouts of the same model: `r_full` (nothing
removed), `dg` (depth removed), `crg` (composition removed), `drg` (both
removed), built by `drg.ladder_rungs` [L31]. The gaps between rungs price
each confounder.

**Depth proxy (`lhat`).** The per-spot depth axis the ladder controls for.
Three routes [L21]: the observed log library size, a user-supplied vector,
or a cross-fitted kernel ridge estimate of depth from the predictions
themselves, kept only if it beats its own permutation guard
(`drg.fit_depth_proxy`).

**Split-half reliability and Spearman-Brown.** To price measurement noise,
each spot's molecules are randomly dealt into two half-depth pseudo-spots;
the two halves are put into the declared space and correlated per gene. The
Spearman-Brown step (`ceiling._sb_sqrt`) converts the half-half r into the
full-depth reliability `2r/(1+r)` (the prophecy formula for doubled test
length), and the ceiling is the square root of that reliability. Example:
half-half r of 0.33 gives reliability 0.50, so the ceiling on any model's r
for that gene is about sqrt(0.50) = 0.70. A model above the ceiling is
fitting noise or leaking.

**Bootstrap confidence interval.** Resample the units (spots within a
section, or whole sections across the run) with replacement, recompute the
readout, repeat, and take the 2.5th and 97.5th percentiles
(`core.boot_ci` [L34]). It answers "how much would this number wobble with
different spots", which is a different question from the floor's "could
chance produce it".

**Zero-pixel null.** Any competitor above that uses no image information.
The headline verdict compares the model against the strongest of them
[L41].

**Selection lever.** The score inflation from reporting only the best N
genes chosen on the test data. `levers.topn_value` applies the rule to the
real predictions, `levers.topn_null` applies the identical rule to permuted
ones. Example: top-100 chosen on test moves 0.065 to 0.250 while the
permuted rule alone reaches 0.217, so most of the gain is selection noise.

**Moran's I.** A single number for spatial autocorrelation, roughly the
correlation between a value at a spot and the mean at its neighbours. Used
on the residuals to show how much spatial structure survives the ladder
(`nulls.morans_I`).

---

## 3. Module tours

Ten package modules plus the example script, in the order a build would
naturally layer them. Each tour opens with an input / processing / output
table, then walks the file by its numbered section comments (line numbers
from the annotated v0.3.0 sources).

### 3.1 `spaces.py` (146 lines): the normalisation declaration

| In | Processing | Out |
|---|---|---|
| user string or enum via `audit(space=...)` [L12] | validate against a closed vocabulary | `TargetSpace` member into `audit.py` and `ceiling.apply_space` |

- `TargetSpace` (line 54): a `str` Enum of the six recognised spaces plus
  `CUSTOM`; the docstring explains each transform in one line. The
  `DEPTH_CARRYING` frozenset marks which spaces still carry library size,
  feeding `DepthResult.target_carries_depth`.
- `coerce_space` (line 95, block at line 129): accepts an enum member
  unchanged or a lowercase string, raises `MissingSpace` on None (no
  default, by design decision D1) and lists the vocabulary on a typo.
  `CUSTOM` demands a non-empty `space_note`.

### 3.2 `core.py` (847 lines): Stat, linear algebra, permutations

| In | Processing | Out |
|---|---|---|
| scalars, per-gene r vectors, designs, coords, rng from `audit.py` and `drg.py` | freeze value+floor+CI triples; correlation, SVD bases, folds, permutation plans, bootstrap | `Stat` objects to `report.py`; `pidx`/`pmask`, folds, bases to every consumer |

- Exceptions (from line 100): `StnullError` and its five children
  (`MissingSpace`, `NotComparable`, `LibSizeLooksLikePanelSum`,
  `InsufficientData`, `BadInput`), each raised at exactly one kind of gate.
- `Stat` (line 137) [L40]: the frozen triple. Methods: `excess` (line 196),
  `above_floor` (line 213, three-valued), `_fmt` (line 227), `__float__`,
  `__str__` (line 243, the console renderer), `to_dict` (line 261, the JSON
  renderer). `na_stat` (line 283) is the "readout absent, here is why"
  constructor.
- `make_stat` (line 307) [L39]: section 1 (line 346) turns the (B,) draws
  into the 95th-percentile floor; section 2 (line 378) degrades when fewer
  than 5 finite draws exist; section 3 (line 387) handles no-null-offered;
  section 4 (line 393) freezes the triple.
- `colcorr` (line 404) [L25]: section 1 (line 433) coerces to float64;
  section 2 (line 442) centres columns and forms the Pearson ratio; section
  3 (line 451) maps constant columns to NaN rather than 0.
- `ortho_basis` (line 460) [L32]: section 1 (line 487) normalises the
  design; section 2 (line 494) takes a thin SVD and keeps only
  well-conditioned directions, which is how the one-hot-plus-intercept
  collinearity of the composition designs is absorbed.
- `resid` (line 506) and `partial_r` (line 529): projection off a design
  and correlation of the two residuals; the ladder's arithmetic.
- `agg` (line 550): per-gene vector to (scalar, n_genes_used), median or
  mean, NaN-aware.
- `kfold_indices` (line 584) [L20]: section 1 (line 598) clamps k and
  shuffles; section 2 (line 602) cuts folds. One partition per section is
  shared by all nulls.
- `perm_plan` (line 616) [L19]: section 1 (line 663) guards; section 2
  (line 669) is the torus-shift path with sub-blocks 2a (line 694,
  cell-to-spot lookup) and 2b (line 711, wrapped shift); section 3
  (line 724) is the free fallback. `_to_grid` (line 731) embeds coordinates
  in an integer lattice.
- `boot_ci` (line 757) [L34]: refusals (line 791), replicate draws
  (line 796), percentile interval (line 814). `sha1_of` (line 828) [L51]
  fingerprints inputs for the run card.

### 3.3 `nulls.py` (620 lines): the zero-pixel competitors

| In | Processing | Out |
|---|---|---|
| `y`, `loglib` [L05], `comp` [L07], `coords`/`W` [L06][L28], folds [L20], `is_train` [L10] from `audit._section_block` | fit small linear or neighbour models, score with the shared rule | `(r_per_gene, pred, scored)` triples [L25][L26] back to `_section_block` |

- `linear_null_pred` (line 96): the shared engine behind three nulls.
  Sections: fit-mode gate (line 142), float64 coercion and empty outputs
  (line 152), `_note` sink (line 167), NaN/inf design refusal (line 177),
  in-sample branch (line 190), train-only branch (line 209, fit on
  `is_train`, score the rest), cross-fit branch (line 231, the k folds).
- `_score` (line 262): turns `(pred, scored)` into the `(r, pred, scored)`
  triple every null returns.
- `depth_null` (line 281): design block at line 304, `X = [1, loglib]`.
- `composition_null` (line 317): design block at line 339,
  `X = [1, comp]`, with 1-D labels one-hot upstream.
- `depth_composition_null` (line 360): design block at line 369, both
  stacked.
- `neighbour_null` (line 388): island mask (line 407), then `W @ Y` as the
  neighbour-mean prediction (line 420).
- `neighbour_matrix` (line 431) [L28]: grid path hashes integer cells to
  rook neighbours (line 458); physical path uses a KD-tree k-NN query
  (line 487); row-standardisation (line 510) makes `W @ Y` a mean.
  `_row_standardise` (line 519) maps islands to zero rows.
- `smooth_matrix` (line 537) [L29]: queen incidence plus identity, one
  row-standardisation (line 557); prices the truth-smoothing lever.
- `morans_I` (line 570): centring and S0 (line 595), numerator as
  residual-neighbour cross-product (line 610).

### 3.4 `drg.py` (426 lines): depth proxy and the attribution ladder

| In | Processing | Out |
|---|---|---|
| `y`, `yh`, `loglib`, `comp`, rng, folds, `pidx`/`pmask` from `audit._section_block` | cross-fitted depth proxy with a permutation guard; four residualised rungs; fixed-design floors and bootstraps | `lhat` [L21], rung vectors [L31], designs [L32], floors [L36], CIs [L34] |

- `fit_depth_proxy` (line 91): coercion (line 126), the fewer-than-10-spots
  refusal (line 134), kernel construction from standardised predictions
  (line 147), fold split (line 161), alpha grid with paired cross-fitting
  (line 169), non-convergence report (line 195), and the permutation
  selection guard (line 202) that re-runs the identical grid on permuted
  targets and rejects `lhat` when the real fit does not beat its own
  selection noise.
- `ladder_rungs` (line 259) [L31][L32]: designs `X1 = [1, lhat]`,
  `X2a = [1, comp]`, `X2 = [1, lhat, comp]` built at line 296; each rung is
  `colcorr(resid(y, Q), resid(yh, Q))` at line 322, with `Q` the
  orthonormal basis from `core.ortho_basis`.
- `_pr` (line 338): the project-then-correlate helper; `X=None` means the
  raw `r_full` rung (line 372).
- `perm_null_of_rung` (line 351) [L36]: residuals are computed once against
  the fixed design, then only the spot order permutes (line 381).
- `boot_of_rung` (line 403): rebuilds the basis on resampled rows
  (line 418) so the CI includes design refitting.

### 3.5 `ceiling.py` (371 lines): the measurement ceiling

| In | Processing | Out |
|---|---|---|
| raw `counts` [L08], `space` [L12], `lib_size_full` [L14] per section from `audit.audit()` | binomial split-half thinning, space transform, Spearman-Brown, same-rule floor | `(r_tech_per_gene, perm_null, method, note)` into the [L47] chain |

- `_dense` (line 94): sparse-to-dense entry.
- `apply_space` (line 106): one branch per `TargetSpace` member, panel
  CP10K (line 131), full-denominator CP10K (line 136), plain log1p
  (line 141), ST-Net median scaling (line 144), z-scored log (line 152);
  `CUSTOM` returns None, tested by the 2-row probe in `r_tech`.
- `_sb_sqrt` (line 168): the Spearman-Brown step (line 175) from half-depth
  to full-depth agreement.
- `r_tech` (line 186): the CUSTOM probe (line 232), the method split
  between panel-only and rest-of-transcriptome thinning (line 241),
  per-replicate dealing of every molecule (line 275), half library sizes
  (line 282), both halves into the declared space (line 293), the
  permutation floor scored with the same rule (line 306), per-gene
  average-then-transform (line 328), the per-draw collapse (line 338), and
  the sparsity caveat (line 355).

### 3.6 `levers.py` (218 lines): pricing the protocol levers

| In | Processing | Out |
|---|---|---|
| per-gene r vectors [L33], permuted r matrices, train-side keys [L24], `Y`, leakage design `Xl` [L46] | apply a questionable protocol choice identically to real and permuted data | lever values and nulls into `SelectionResult` |

- `topn_value` (line 48, block at line 68): rank finite per-gene r, keep
  the best `n_top`, aggregate.
- `topn_null` (line 85, block at line 92): the same rule once per
  permutation draw, giving the selection-noise distribution.
- `train_selected_value` (line 105, block at line 125): genes ranked by the
  train-side key instead, the legal counterpart.
- `null_transfer_leakage` (line 151): per held-out section, the two
  training sides that section-level and patient-level splits would allow
  (line 181), the same least-squares null fit on each (line 199); the
  difference is the split-granularity premium, measured on a zero-pixel
  null and therefore a lower bound.

### 3.7 `audit.py` (2242 lines): the orchestrator

| In | Processing | Out |
|---|---|---|
| all user inputs [L01] to [L14] | validate, coerce, loop sections, call every null / rung / lever / ceiling, aggregate, assemble | `AuditReport` [L38] to [L52]; wrappers [L53] to [L56] |

First half, validation and per-section machinery:

- `_mat` (line 126) and `_vec` (line 156): the only entry gates for
  matrices and vectors.
- `check_inputs` (line 190) [L52]: empty-report seeding (line 220), spots
  per section (line 256), the library-size panel-sum trap (line 278), the
  counts integerness spot-check (line 319), presence flags (line 366),
  grid plausibility (line 398).
- `_one_hot` (line 430) [L07]: labels to a 0/1 matrix.
- `_section_block` (line 455): the per-section engine. RNG split into
  three streams (line 501) [L18]; the shared permutation plan (line 513)
  [L19]; the three-route depth proxy (line 538) [L21]; the zero-pixel
  competitors (line 589) [L25] to [L27]; the neighbour null (line 657)
  [L30]; the held-out `ev` restriction (line 693) [L23]; the four ladder
  rungs (line 717) [L31], the observed-depth rung `Xo` (line 758) [L35];
  the selection levers (line 771) [L33][L45]; the spatial readouts
  (line 803) [L29][L37].
- `_perm_of_pred` (line 840) [L26] and `_perm_pergene` (line 880) [L33]:
  the two permutation re-scorers, scalar and per-gene.
- `_combine` (line 903) [L39]: the single funnel; nanmedian over sections,
  per-draw section medians for the floor, section bootstrap for the CI.

Second half, `audit()` (line 975) and assembly:

- strict pre-validation (line 1088); the coercion layer (line 1106) where
  `Y`, `P`, `sec`, `lib`, `loglib` are born [L01] to [L05]; configuration
  repairs (line 1142); coordinates and the lattice check (line 1169)
  [L06]; gene names (line 1203) [L13]; the composition matrix (line 1212)
  [L07]; the degradation ledger seeding (line 1248) [L48]; top-N trimming
  (line 1320); perm-kind resolution (line 1336); section order and
  duplicate coordinates (line 1350); the per-section loop (line 1376)
  [L38] with the `per_section` (line 1414) [L42] and `per_gene`
  (line 1427) [L43] rows; failed-proxy grouping (line 1458); the run-level
  floor label (line 1494); the ladder table (line 1523) [L44]; the
  headline (line 1559) and per-section win fractions (line 1598); the
  depth block (line 1630) with the DG share (line 1669) and the
  sd(log lib) table (line 1693); the composition block (line 1729); the
  spatial block (line 1756) with the block-vs-free pair (line 1788); the
  selection block (line 1816) with the train-selected counterpart
  (line 1838), the leakage lever (line 1851) [L46], and the lever price
  table (line 1881); the ceiling block (line 1939) [L47] with the ratio
  (line 1987) and its guards (lines 2065, 2077); the run card (line 2010)
  [L51]; the final container (line 2037).
- Wrappers: `nulls_only` (line 2131) [L53], `ladder` (line 2144) [L54],
  `perm_floor` (line 2159) [L55], `compare` (line 2205) [L56].

### 3.8 `report.py` (1357 lines): rendering, no computation

| In | Processing | Out |
|---|---|---|
| the frozen `AuditReport` and its `Stat` fields [L40] to [L50] | word claims, format tables, serialise, plot | console text, JSON, CSV dir, Markdown, HTML, matplotlib figures |

- Result dataclasses (lines 73 to 282): `RunMeta`, `Headline`,
  `DepthResult`, `CompositionResult`, `SpatialResult`, `SelectionResult`,
  `CeilingResult`, `Claim`, `AuditReport`. Containers only.
- `claims` (line 284): the evidence record helper (line 294), the
  finite-numbers guard (line 305), the DG wording (line 331), the ladder
  DRG claim (line 347), and the three forbidden sentences (line 376).
- `caveats` (line 415) [L48] to [L50]; `summary` (line 430) with the
  model-versus-strongest-null line (line 439).
- Serialisers: `to_json` (line 565, dict assembly at line 574),
  `to_csv_dir` (line 601, claims flattening at line 618), `to_markdown`
  (line 630), `to_html` (line 642), `plot` (line 654).
- Formatting helpers (lines 673 to 761): `_f`, `_ci`, `_fracstr`,
  `_default`, `_sec_json` (line 733, the field walker), `_df_json`.
- HTML machinery: `_stat_cells` (line 867, the value/floor/CI cells; the
  floor column cannot be switched off), `_stat_table` (line 888),
  `_df_table` (line 910), `_plot` (line 937, ladder lines at line 945,
  per-readout lines at line 975), `_figs` (line 995, in-memory PNGs),
  `_render_html` (line 1026: banners at line 1043, run card at line 1053,
  depth table at line 1105, composition at line 1128, spatial at
  line 1150, claims by verdict at line 1224, the ready-to-paste Methods
  paragraph at line 1252).
- `_render_md` (line 1263), `_md_table` (line 1303), `cite_text`
  (line 1317, protocol values pulled off `RunMeta` at line 1325).

### 3.9 `cli.py` (383 lines): flags to keywords, nothing else

| In | Processing | Out |
|---|---|---|
| file paths and `--*-col` flags [L15][L16] | load arrays, rename columns onto `audit()` keywords | an `audit()` call and `_emit` outputs; exit code |

- `_load_matrix` (line 64): extension dispatch (line 74), the `.npz`
  archive-or-sparse split (line 82).
- `_load_obs` (line 107): the per-spot table.
- `_common` (line 124): the shared flag set; matrices and `--space` at
  line 133, statistical settings forwarded verbatim at line 158.
- `_kwargs` (line 188): loads the three objects (line 197), always-present
  keywords (line 205), optional per-spot columns (line 220). Every flag is
  exactly one rename onto an `audit()` keyword; the mapping is row [L16].
- `_emit` (line 263): `--out` by extension (line 271), additive `--json`
  and `--csv-dir` (line 284), the always-printed console summary
  (line 292), the `--strict-exit` degradation gate (line 298).
- `main` (line 306): four subcommands sharing `_common` (line 312); the
  positional `audit(Y, P, **kw)` call at line 357.

### 3.10 `__init__.py` (81 lines): the public surface

| In | Processing | Out |
|---|---|---|
| the nine modules | re-export 26 names in four import groups | `import stnull` API |

Four commented import groups: the entry points (`audit`, `nulls_only`,
`ladder`, `perm_floor`, `compare`), the result containers, the Stat and
exception types, and the space vocabulary. No logic.

### 3.11 `examples/her2st_example.py` (239 lines): the worked proof

| In | Processing | Out |
|---|---|---|
| local HER2ST counts and frozen ridge predictions | rebuild `Y`, `P`, `counts`, `obs` on a 100-gene toy panel | one real `audit()` call [L17]; CLI-format dumps |

- BLAS thread cap before numpy import (line 48).
- `build` (line 71): load per-fold ridge predictions (line 80), intersect
  the differing fold panels (line 93), slice each section's sparse counts
  to the panel (line 110), truth as log1p CP10K on the panel (line 133),
  `obs` rebuilt in stacked row order (line 150).
- `dump_for_cli` (line 162): serialises the same five objects in the
  formats `cli._load_matrix` reads (line 165).
- `main` (line 189): the `audit()` call with the same keywords the CLI
  maps (line 194), then every renderer exercised on the one report
  (line 218).

---

## 4. Variable lineage table

One numbered row per variable or tensor that crosses a file or function
boundary. These ids are cited inside the source comments; run
`grep -rn "\[L07\]" src/stnull/` (any id) to list every birth and
consumption site with current line numbers. Anchors below are
`file:function`; a rename after a pure slice (such as `Y[m]` becoming
parameter `y`) is treated as the same object.

### A. User inputs into `audit()` (the coercion layer)

| ID | Name | Shape and units | Born at | Consumed by | Meaning |
|----|------|-----------------|---------|-------------|---------|
| [L01] | `y_true` -> `Y` | (n_all, g) float, declared space | `audit.audit` via `_mat`, densified | sliced `Y[m]` per section into `_section_block(y=...)`; from there every null (`nulls.py`), every rung (`drg.py`), every floor | the ground truth being audited; never transformed (decision D2) |
| [L02] | `y_pred` -> `P` | (n_all, g) float or None | `audit.audit` via `_mat` | `_section_block(yh=...)`; None switches on nulls-only mode | the model predictions under audit |
| [L03] | `section` -> `sec` | (n_all,) str | `audit.audit` via `_vec` | the per-section loop, the ceiling loop, `levers.null_transfer_leakage` | which tissue slice each spot belongs to; all r are within-section |
| [L04] | `lib_size` -> `lib` | (n_all,) UMI counts | `audit.audit` via `_vec`; panel-sum trap in `check_inputs` | `loglib` [L05]; ceiling fallback; sha1 in `RunMeta` | sequencing depth per spot; the biggest known confounder |
| [L05] | `loglib` | (n_all,) float | `np.log(np.maximum(lib, 1))` in `audit.audit`, once | depth null and joint null designs (`nulls.depth_null`, `depth_composition_null`), observed-DRG design `Xo`, `lhat` under `depth_proxy='observed'`, leakage design `Xl`, `sd_loglib`, target of `drg.fit_depth_proxy` | log depth: the zero-pixel predictor the depth null runs on |
| [L06] | `coords` -> `XY` | (n_all, 2) float | `audit.audit`; `_grid_plausibility` may flip `coord_kind` | `core.perm_plan` block permutation; `nulls.neighbour_matrix`; `nulls.smooth_matrix` | spot positions; everything spatial hangs on them |
| [L07] | `labels` / `composition` -> `Comp` | (n_all, K) one-hot or proportions | `audit.audit`: small classes merged to 'other', then `_one_hot` | composition and joint null designs (`nulls.py`), ladder designs X2a/X2 (`drg.ladder_rungs`), `Xo`, `Xl`; the one-hot-plus-intercept collinearity is absorbed by `core.ortho_basis` | what tissue class each spot is; the composition confounder |
| [L08] | `counts` | (n_all, g) raw integers, dense or sparse | user arg; integer spot-check in `check_inputs` | only `ceiling.r_tech`, per section | raw UMI counts: the only input that can price measurement noise |
| [L09] | `patient` -> `pat` | (n_all,) str | `audit.audit` via `_vec` | `levers.null_transfer_leakage`; `RunMeta.n_patients` | which spots share a patient; enables the split-granularity lever |
| [L10] | `is_train` -> `istr` | (n_all,) bool | `audit.audit` via `_vec` | fit mask of every null under `null_fit='train_only'` (`nulls.linear_null_pred`); the `ev` mask [L23]; the train-side key [L24] | the user's own train/test split |
| [L11] | `lib_pred` -> `lpred` | (n_all,) float | `audit.audit` via `_vec` | becomes `lhat` under `depth_proxy='given'` [L21] | a depth proxy the user computed themselves |
| [L12] | `space` -> `t_space` | `TargetSpace` member | `spaces.coerce_space` (REQUIRED, no default) | `ceiling.r_tech` via `apply_space`; `DEPTH_CARRYING` membership; `RunMeta.space`; the `compare()` comparability key | which normalisation `y_true` is in |
| [L13] | `genes` -> `gene_names` | list of g str | user arg, DataFrame columns, or synthesised `g%d` | `per_gene` rows [L43] | gene names, report cosmetics only |
| [L14] | `lib_size_full` | (n_all,) counts | user arg, finiteness-checked | ceiling only: enables rest-of-transcriptome thinning in `ceiling.r_tech` | full-transcriptome depth for honest half-depth denominators |

### B. Entry points (CLI and example)

| ID | Name | Shape and units | Born at | Consumed by | Meaning |
|----|------|-----------------|---------|-------------|---------|
| [L15] | `--true` / `--pred` files -> `Y`, `P` | on-disk .npy/.npz/.csv/.parquet | `cli._load_matrix` inside `cli._kwargs` | positional args of `audit(Y, P, **kw)`, i.e. [L01]/[L02] | the CLI's matrix loading; no statistics in the CLI |
| [L16] | `--obs` columns -> audit kwargs | one per-spot table | `cli._load_obs`, picked out in `cli._kwargs` | mapping: `--section-col` -> `section` [L03], `--lib-col` -> `lib_size` [L04], `--coord-cols` -> `coords` [L06], `--label-col`/`--comp-cols` -> [L07], `--patient-col` -> [L09], `--train-col` -> [L10], `--libpred-col` -> [L11], `--libfull-col` -> [L14], `--counts` -> [L08], `--genes` -> [L13], `--space` -> [L12] | the CLI-to-API contract: every flag is a rename, never a computation |
| [L17] | example arrays -> audit kwargs | (n_all, 100) toy panel | `her2st_example.build`: `Y` from raw counts re-normalised, `P` from frozen ridge predictions re-projected, `C` raw counts, `obs` meta | the `audit()` call in `her2st_example.main`; `dump_for_cli` writes the same objects as [L15]/[L16] files | proof-of-API on local HER2ST data; not part of the package |

### C. Per-section machinery (inside `audit._section_block`)

| ID | Name | Shape and units | Born at | Consumed by | Meaning |
|----|------|-----------------|---------|-------------|---------|
| [L18] | `rng_perm`, `rng`, `rng_boot` | 3 independent Generators | split at the top of `_section_block` from a per-section seed | `perm_plan`; null fitting and `fit_depth_proxy`; `boot_ci` | three streams so changing `n_perm` or `n_boot` cannot move point estimates |
| [L19] | `pidx`, `pmask` | (B, n) int, (B, n) bool or None | `core.perm_plan` (torus shift with coords, free otherwise); REBUILT on the held-out subset after the `ev` mask | `_perm_of_pred`, `_perm_pergene`, `drg.perm_null_of_rung`, the Moran permutation loop | the shared permutation plan: one coordinate-aware null spot order reused by every floor |
| [L20] | `folds` | 5 (train_idx, test_idx) pairs | `core.kfold_indices` once per section | every `nulls.linear_null_pred` cross-fit call | ONE partition shared by all nulls, so null differences measure the nulls, not the splits |
| [L21] | `lhat` | (n,) float | three routes in `_section_block`: 'observed' = `loglib`, 'given' = `lib_pred`, 'from_pred' = `drg.fit_depth_proxy` (guarded, all-NaN when rejected) | the depth_1d null; ladder design `X1 = [1, lhat]` | the depth axis DG/DRG residualise on; the route changes the meaning of the rung |
| [L22] | `rp`, `alpha` -> `depth_r_of_pred`, `depth_alpha` | scalars | returns of `fit_depth_proxy`, or direct `corrcoef(lhat, loglib)` under 'given' | `DepthResult.depth_r_of_pred`; a `per_section` column | how much depth the model's predictions carry (N/A under 'observed': tautology) |
| [L23] | `ev` | (n,) bool or None | `~is_train` when `null_fit='train_only'` | slices y, yh, lhat, comp, loglib, coords; ladder, levers and spatial readouts run on `ev` spots only | model and nulls must be scored on the SAME held-out spots |
| [L24] | `train_key` / `key` | (g,) train-side per-gene r | `colcorr` on the training spots | `levers.train_selected_value` and its permuted null | the legal gene ranking, computed before test spots are seen |

### D. Nulls: fit to Stat (the r_per_gene chain)

| ID | Name | Shape and units | Born at | Consumed by | Meaning |
|----|------|-----------------|---------|-------------|---------|
| [L25] | `r` (r_per_gene of a null) | (g,) Pearson r, NaN where unscorable | `nulls._score` via `core.colcorr`; first return of each null | `core.agg` to a section scalar, `col()` accessor, `audit._combine`, `core.make_stat`, into DepthResult / CompositionResult / SpatialResult and the headline candidates | the whole null pipeline in one line: per-gene r -> section median -> across-section median -> floored Stat |
| [L26] | `pred`, `scored` (of a null) | (n, g) float, (n,) bool | second and third returns of each null, born in `nulls.linear_null_pred` or `neighbour_null` | `audit._perm_of_pred` -> the null's own (B,) floor draws | the null's predictions, permuted to build the floor of the null's readout |
| [L27] | `out["depth_null_r"]` | (g,) float | stored in the section block | available to downstream tooling via `blocks`; the scalar travels via [L25] | per-gene depth-null r kept at full resolution |
| [L28] | `W` | (n, n) sparse CSR, row-standardised | `nulls.neighbour_matrix`, rebuilt on the `ev` subset | `neighbour_null(y, W)`; `morans_I(R, W)` and its permuted sub-blocks | who neighbours whom; makes `W @ Y` a neighbour mean |
| [L29] | `S`, `ys` | (n, n) smoother; (n, g) smoothed truth | `nulls.smooth_matrix`; `ys = S @ y` | `smooth_value`, `smooth_delta`, permuted twin -> `SpatialResult.smooth_lever` | prices the "smooth the truth before scoring" lever |
| [L30] | `r_nbr` | section scalar, then Stat | `agg` of the neighbour-null r | `SpatialResult.r_nbr`; duplicated into `CeilingResult.r_nbr` | how much r pure spatial smoothing of the TRUE y buys, no pixels |

### E. The attribution ladder (`drg.py`)

| ID | Name | Shape and units | Born at | Consumed by | Meaning |
|----|------|-----------------|---------|-------------|---------|
| [L31] | `rungs` = {r_full, dg, crg, drg} | each (g,) or None | `drg.ladder_rungs`; each rung `colcorr(resid(y,Q), resid(yh,Q))` via `_pr` | `agg` to section scalars, per-gene kept for `per_gene`; ladder table [L44]; headline Stats; claims wording | what survives when depth and/or composition are partialled out of BOTH sides |
| [L32] | `designs` = {dg: X1, crg: X2a, drg: X2} | (n,2), (n,1+K), (n,2+K) | inside `ladder_rungs`: X1=[1,lhat], X2a=[1,comp], X2=[1,lhat,comp] | `core.ortho_basis` inside `_pr`, `perm_null_of_rung`, `boot_of_rung` | the control subspaces; the design stays fixed while spot order permutes |
| [L33] | `rf`, `perm_r` | (g,) observed; (B, g) permuted | `rungs["r_full"]`; `audit._perm_pergene` | `levers.topn_value`, `levers.topn_null`, the train-selected null | the same top-N rule applied to real and to permuted predictions |
| [L34] | `out[k+"_ci"]`, note | (lo, hi) or None; str | `core.boot_ci` over `drg.boot_of_rung` | ladder CI columns; `drg_above_floor`; the single-section fallback CI of `_combine` | spot-bootstrap 95% interval of each rung, residualisation redone per draw |
| [L35] | `Xo` | (n, 2+K) = [1, loglib, comp] | built in `_section_block`, finiteness-gated | `out["drg_observed"]` via `drg._pr` plus its floor; a ladder table column | the conservative DRG that controls the MEASURED depth instead of `lhat` |
| [L36] | `out["r_full_perm_mean"]` | (B,) | `drg.perm_null_of_rung(..., how="mean")` | the floor of the mean-over-genes headline `r_mean` | the mean rule's own null; the median rule's null is a different distribution |
| [L37] | `out["r_full_perm_free"]` | (B',) free permutation | a second `perm_plan(kind="free")` capped at 100 draws | `SpatialResult.perm_block_vs_free`; HTML contrast rows | the same headline under the anticonservative floor, printed for contrast |

### F. Aggregation, Stat, and the report

| ID | Name | Shape and units | Born at | Consumed by | Meaning |
|----|------|-----------------|---------|-------------|---------|
| [L38] | `blocks` | {section -> per-section dict} | the per-section loop in `audit.audit` | the `col`/`pcol`/`ng` accessors; every `_combine` call; win fractions; ladder rows | the raw per-section results before any across-section aggregation |
| [L39] | `_combine(...)` -> Stat | vals: section scalars; perms: (B,) arrays | `audit._combine`: nanmedian value, per-draw section-median floor, section bootstrap CI | `core.make_stat` (floor = 95th percentile) | the single funnel: section scalars in, one floored Stat out |
| [L40] | `Stat.value` / `.floor` / `.ci` | float; float or None; (lo, hi) or None | frozen dataclass in `core.py`, built only by `make_stat` or `na_stat` | `report.py` renderers: `Stat.__str__`, `Stat.to_dict`, `_stat_cells`/`_stat_table` (the floor column cannot be switched off), claims | the triple: the package's contract that no number travels without its null |
| [L41] | `strongest` | (name, Stat) or None | `max` over the out-of-sample null candidates (oracle barred) | `Headline.strongest_null`; summary lines; claim 1; `compare()` columns | the best zero-pixel competitor the model must beat |
| [L42] | `per_section` | DataFrame, one row per section | rows in the per-section loop | `AuditReport.per_section`; CSV, JSON, HTML | the per-section readouts, flat, for downstream analysis |
| [L43] | `per_gene` | DataFrame, one row per (section, gene) | rows in the per-section loop | `AuditReport.per_gene`; `to_csv_dir` | per-gene rung values with gene names from [L13] |
| [L44] | `ladder_df` | one row per section, rungs + floors + CIs | assembly in `audit.audit` | DRG claim; `plot('ladder')`; console section [7] | the ladder table the paper's figure is drawn from |
| [L45] | `out["topn"][N]` = (v, null, k_used) | scalar, (B,), int | `levers.topn_value` / `topn_null` | `_combine` per N with `floor_kind='perm_selection'`; `SelectionResult.by_n`; lever price rows | the test-selected readout beside what pure selection noise buys |
| [L46] | `leak_df`, `leak` | DataFrame; Stat | `Xl = [1, loglib, Comp]`; `levers.null_transfer_leakage`; `make_stat` of the median delta | `SelectionResult.leakage_lever`; a lever price row | the section-vs-patient split premium on a zero-pixel null (lower bound) |
| [L47] | `r_tech` chain | (g,) per section -> Stat | `ceiling.r_tech` (thinning, `apply_space`, `_sb_sqrt`) | `agg`, then `_combine` -> `CeilingResult.r_tech`; the ratio `r_over_ceiling` | the measurement ceiling and the model's fraction of it |
| [L48] | `degr` | list of str | seeded early in `audit()`, appended at every fallback | `RunMeta.degradations`; `caveats()`; console section [9]; the CLI `--strict-exit` gate | the degradation ledger: every fallback with the direction it biases the headline |
| [L49] | `warns` | list of str | seeded in `audit()`, filled by `check_inputs` echoes and throughout | `AuditReport.warnings`; `caveats()`; JSON | non-ledger warnings (validation echoes, skipped sections) |
| [L50] | `na` | {section name -> reason} | seeded in `audit()` | `caveats()`; console N/A lines | why an entire report section is absent |
| [L51] | `inputs_sha1` | dict of 12-hex digests | `core.sha1_of` per input | `RunMeta.inputs_sha1`; the JSON run card | reproducibility fingerprints of the exact inputs |
| [L52] | `chk` (InputReport) | ok / errors / warnings | `audit.check_inputs` | raises `InsufficientData` under strict, else feeds `warns` [L49]; the whole `stnull check` subcommand | structural validation before any correlation exists |

### G. Convenience wrappers (thin re-routes, no new tensors)

| ID | Name | Returns | Route | Meaning |
|----|------|---------|-------|---------|
| [L53] | `nulls_only(...)` | AuditReport | forwards to `audit(y_true, None, ...)` | nulls with no model |
| [L54] | `ladder(...)` | DataFrame | runs full `audit`, returns `rep.ladder` [L44] | just the ladder table |
| [L55] | `perm_floor(...)` | Stat | re-implements the loop with `perm_plan` + `_perm_pergene` + `_combine` directly | one readout rule's floor, standalone |
| [L56] | `compare(*reps)` | DataFrame | comparability key from `RunMeta` [L12], raises `NotComparable` | side-by-side of audits, or a refusal |

### The ten most load-bearing flows

1. [L01]/[L02] `y_true`/`y_pred` -> `_mat` coercion -> per-section slices
   -> `_section_block(y, yh)`: the spine every other row hangs off.
2. [L05] `loglib`, born once in `audit()`, consumed by the depth null, the
   joint null, the observed-DRG design, the leakage design, and `lhat`
   under 'observed'.
3. [L21] `lhat` and its three birthplaces (observed / given / from_pred
   with guard): the one variable whose provenance changes the meaning of
   DG and DRG.
4. [L07]+[L32] labels -> merged classes -> `_one_hot` -> `Comp` -> null
   design and ladder designs -> `ortho_basis` SVD swallowing the intercept
   collinearity.
5. [L25] the r_per_gene chain: `_score`/`colcorr` -> `agg` -> `col()` ->
   `_combine` -> `make_stat` -> Stat: the only path a null value can take
   to the report.
6. [L19] `pidx`/`pmask` from `perm_plan`: coordinate-dependent, drawn once,
   rebuilt after the `ev` mask, reused by every floor.
7. [L39]+[L40] `_combine` -> Stat(value, floor, ci) -> the `report.py`
   renderers: the triple's whole life from computation to page.
8. [L16] the CLI contract: each `--*-col` flag is exactly one rename onto
   an `audit()` keyword; the CLI computes nothing.
9. [L17] the example's `build()`: raw HER2ST counts and frozen ridge
   predictions re-projected onto a 100-gene panel, handed to the same
   `audit()` keywords the CLI uses.
10. [L31]+[L33] `ladder_rungs` per-gene vectors: aggregated for the ladder
    Stats, kept per-gene for the `per_gene` table, re-ranked by
    `topn_value`/`topn_null` for the selection lever.

---

## 5. Reading paths

### Path 1: non-researcher, about 30 minutes

Goal: understand what the tool claims and why the floor column exists.

1. Section 1 of this file (the story), then the Glossary entries for spot,
   library size, Pearson r, and permutation floor.
2. `README.md`, the table at the top and the "What each number means"
   section.
3. `src/stnull/core.py` lines 137 to 275: the `Stat` class and its
   `__str__`. This is the whole output contract in one screen.
4. `src/stnull/core.py` `make_stat` (line 307), reading only the WHAT and
   WHY lines of its four numbered sections.
5. One rendered report: `examples/sample_report.html`, matching each table
   to the Glossary terms.

### Path 2: ST researcher, about 2 hours

Goal: be able to trust, or challenge, every number in a report.

1. Path 1, steps 1 to 4.
2. `src/stnull/nulls.py` top to bottom: `linear_null_pred` (line 96) and
   the three linear nulls (lines 281, 317, 360), then `neighbour_null`
   (line 388) and `neighbour_matrix` (line 431). Watch fit modes: cv,
   train_only, in-sample.
3. `src/stnull/core.py` `perm_plan` (line 616) and `kfold_indices`
   (line 584): why the floor is block-permuted and why all nulls share one
   fold partition.
4. `src/stnull/drg.py`: `fit_depth_proxy` (line 91) with its permutation
   guard, then `ladder_rungs` (line 259) and `perm_null_of_rung`
   (line 351).
5. `src/stnull/ceiling.py` `r_tech` (line 186): the thinning, the space
   transform, the Spearman-Brown step, the same-rule floor.
6. `src/stnull/levers.py` all four functions: the selection and leakage
   levers.
7. `src/stnull/audit.py` `_section_block` (line 455) once, slowly,
   following the [L##] tags; then skim the assembly half of `audit()`
   (from line 1376) to see how section blocks become the report.
8. Section 4 of this file as a reference while reading, plus
   `docs/INTERPRETATION.md` for wording.

### Path 3: contributor, the complete route

Goal: change code without moving a number you did not intend to move.

1. Path 2 in full.
2. `src/stnull/spaces.py` and `src/stnull/audit.py` `check_inputs`
   (line 190): every refusal and every trap, since new inputs enter here.
3. The full coercion and assembly half of `audit.py` (line 975 to the
   end), including `_combine` (line 903), the degradation ledger [L48],
   the run card [L51], and the four wrappers (from line 2131).
4. `src/stnull/report.py` end to end: the dataclasses, `claims`
   (line 284, including the three forbidden sentences at line 376), the
   serialisers, and `_render_html` (line 1026). Rule to preserve: the
   floor column cannot be switched off.
5. `src/stnull/cli.py`: confirm any new `audit()` keyword gets exactly one
   flag rename in `_kwargs` (line 188) and nothing more.
6. `examples/her2st_example.py` and the test suite under `tests/`: run
   `python -m pytest` before and after; the HER2ST example is the
   regression anchor.
7. Bookkeeping when you touch data flow: add or extend a row in the
   lineage table (Section 4; new rows take new numbers at the end of their
   group, numbers are never reused), and cite the id in a WHAT / WHY /
   WHERE comment at the birth and consumption sites.
