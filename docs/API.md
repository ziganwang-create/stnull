# stnull API reference

Version 0.3.0. Author: Zigan Wang. Everything public is re-exported from the
package root:

```python
from stnull import (audit, nulls_only, ladder, perm_floor, compare, check_inputs,
                    AuditReport, Stat, TargetSpace, RunMeta, Headline,
                    DepthResult, CompositionResult, SpatialResult,
                    SelectionResult, CeilingResult, Claim, InputReport,
                    cite_text, StnullError, MissingSpace, NotComparable,
                    LibSizeLooksLikePanelSum, InsufficientData, BadInput)
```

Contents: [audit](#audit) · [helper entry points](#helper-entry-points) ·
[AuditReport](#auditreport) · [Stat](#stat) · [result records](#result-records) ·
[TargetSpace](#targetspace) · [exceptions](#exceptions) · [CLI](#cli) ·
[design decisions](#design-decisions)

---

## `audit`

```python
audit(y_true, y_pred=None, *, space=None, section=None, genes=None,
      spot_ids=None, lib_size=None, coords=None, coord_kind="grid",
      labels=None, composition=None, counts=None, lib_size_full=None,
      patient=None, is_train=None, lib_pred=None, space_note="",
      depth_proxy="observed", null_fit="cv_within_section",
      top_n=None, n_perm=200, n_boot=500, perm_kind="auto",
      min_spots=50, min_genes=20, min_class_spots=20,
      min_sections_for_trend=5, n_rep_ceiling=20, seed=0, n_jobs=1,
      strict=True, verbose=True) -> AuditReport
```

Audit one set of spatial-expression predictions against zero-pixel null models.
All correlations are per-gene Pearson r computed **within a section**, then
aggregated over genes (median and mean) and over sections (median and mean).

### Required

| arg | type | meaning |
|---|---|---|
| `y_true` | `(n_spot, n_gene)` ndarray / DataFrame | already in the space declared by `space`; stnull never transforms it |
| `space` | `str` or `TargetSpace` | **no default**; see [TargetSpace](#targetspace) |
| `section` | `(n_spot,)` | section / slide id. The only grouping that cannot be omitted |

`y_pred` is optional. Omit it and the call runs **nulls-only**: sections 0, 2, 3,
4 and 6 of the report are produced, sections 1, 5 and 7 are marked `NO_MODEL`.
This is deliberate. See what zero pixels buy before you decide to train.

### Optional evidence (each omission switches off a report section)

| arg | shape | unlocks | omitting it |
|---|---|---|---|
| `lib_size` | `(n_spot,)` | section 2, the DG and DRG rungs | ladder degrades to r_full to CRG; banner: the largest known confounder was not audited |
| `coords` | `(n_spot, 2)` | section 4, block permutation | floors fall back to `perm_free`, which is **anticonservative**: floors too low, headline too flattering |
| `coord_kind` | `"grid"` / `"micron"` | how neighbours and block shifts are defined | since 0.2.0, micron data passed as `"grid"` fails a lattice check and is auto-switched to `"micron"` with a warning and a ledger entry (README defect #3, fixed) |
| `labels` | `(n_spot,)` | section 3, the CRG and DRG rungs | N/A. stnull will **not** invent labels by clustering, which would leak the model's own information into the null |
| `composition` | `(n_spot, K)` | same as `labels`, continuous | if both are given, `composition` wins |
| `counts` | `(n_spot, n_gene)` raw ints, may be `scipy.sparse` | `r_tech` in section 6 | `method="na"`; no Poisson stand-in (D8) |
| `lib_size_full` | `(n_spot,)` | honest full-library denominators in the split-half ceiling (`split_half_full`) | falls back to `lib_size`; without either, panel row sums (`split_half_panel`) |
| `patient` | `(n_spot,)` | the leakage lever | lever N/A; bootstrap stays at spot/section level, so CIs are **too narrow** |
| `is_train` | `(n_spot,)` bool | train-only null fitting and the legal train-selected top-N | nulls are cross-fitted within section, which is slightly optimistic |
| `lib_pred` | `(n_spot,)` | uses **your** predicted depth for l_hat | l_hat is cross-fitted from `y_pred`, which is defect #1 |
| `genes` | `(n_gene,)` | gene names in `per_gene` | auto-named `g0 … g{G-1}`; DataFrame columns win over this argument |
| `spot_ids` | `(n_spot,)` | nothing yet, accepted and unused | -- |

### Knobs

| arg | default | notes |
|---|---|---|
| `depth_proxy` | `"observed"` | `"observed"` (the default since 0.2.0) uses the measured log library size and matched the oracle ladder to +/-0.0011 in synthetic validation; `"from_pred"` cross-fits l_hat from `y_pred` (**biased low, README defect #1**, guarded by a permutation test on the alpha selection); `"given"` uses `lib_pred` and **requires** it (0.3.0: raises under `strict`, falls back to `"observed"` with a ledger entry otherwise); `"none"` skips the depth rungs. Any other value raises `BadInput` (0.3.0). Passing `lib_pred` together with `"from_pred"` still downgrades the request to `"given"`, which is now warned and booked. The `"given"` path carries no permutation guard. Under `"observed"`, `depth.depth_1d` and `depth.depth_r_of_pred` are reported as N/A with that reason: l_hat *is* the measured depth there, so both would be fixed by construction. `drg_observed` is always reported alongside |
| `null_fit` | `"cv_within_section"` | `"train_only"` needs `is_train`, and (0.3.0) restricts **every model-side readout** to the same held-out spots the nulls are scored on, and `per_section.n_eval` records how many; `"oracle"` fits on the whole section (upper bound, and the setting that reproduces published in-section numbers, README defect #2). Any other value raises `BadInput` (0.3.0). One K-fold partition is drawn per section and shared by every null, so a difference between two nulls is not two different random splits |
| `top_n` | `None` = `(10, 50, 100, 250)` trimmed to entries `<= n_gene` (warned) | with `strict=True`, an **explicit** entry above `n_gene` raises `BadInput` |
| `n_perm` | `200` | permutation draws. Point estimates do not move with it (there is a regression test) |
| `n_boot` | `500` | bootstrap draws for the CIs |
| `perm_kind` | `"auto"` | `"auto"` picks `block` when `coords` are given, else `free`; force with `"block"` / `"free"`. Any other value raises `BadInput` (0.3.0) |
| `min_spots` | `50` | sections below this raise under `strict`, else are marked `underpowered` and excluded from cross-section aggregates but kept in `per_section` |
| `min_genes` | `20` | fewer genes than this fails validation |
| `min_class_spots` | `20` | classes below this are merged into `"other"` and listed in `composition.small_classes_dropped` |
| `min_sections_for_trend` | `5` | fewer sections than this sets `depth.trend_sd_vs_r = None`. It does **not** switch off `frac_sections_*`, which stays meaningful at 3 sections (a deliberate deviation from the original spec) |
| `n_rep_ceiling` | `20` | resampling replicates for `r_tech`; roughly half the runtime of a small audit |
| `seed` | `0` | per-section RNG streams are derived from `sha1(section_name)`, not `hash()`, so results do not move between processes |
| `n_jobs` | `1` | **accepted and ignored**; runs are single-process |
| `strict` | `True` | see below |
| `verbose` | `True` | per-section progress on stdout, ASCII only |

### `strict=True` raises on

* `space` missing or unrecognised (`MissingSpace`), or `space="custom"` without
  `space_note=`
* `y_pred` containing NaN / Inf, or a shape mismatch with `y_true`
* any section with fewer than `min_spots` spots, or fewer than `min_genes` genes
* an explicit `max(top_n) > n_gene` (the `top_n=None` default trims itself)
* `lib_size` correlating > 0.999 with the row sum of `y_true`
  (`LibSizeLooksLikePanelSum`: you passed the panel row sum where the
  full-transcriptome depth was wanted)
* non-integer `counts`
* non-finite values in `lib_size`, `lib_size_full`, `coords`, `composition` or
  `lib_pred` (0.3.0). These feed `np.linalg.lstsq`, `cKDTree` and
  `rng.binomial`, none of which degrade gracefully; before 0.3.0 they passed
  validation and surfaced as `LinAlgError: SVD did not converge` naming neither
  the input nor the section. Non-finite `y_true` remains a warning, because it
  degrades per gene by design
* a wrong-length `lib_size_full` (0.3.0: it is validated like every other
  per-spot vector)
* an unrecognised `null_fit`, `depth_proxy` or `perm_kind` (`BadInput`, 0.3.0),
  and `depth_proxy="given"` without `lib_pred`

With `strict=False` these become entries in `rep.warnings`. Numbers that could
not be computed carry `status="na"` or `"degraded"`; note that stnull does *not*
currently mark every downstream number as degraded when a structural check was
waived.

### Returns

An [`AuditReport`](#auditreport).

### Example

```python
rep = audit(y_true=Y, y_pred=P, space="log1p_cp10k:panel",
            section=obs.section.values, lib_size=obs.lib_size.values,
            coords=obs[["arr_x", "arr_y"]].values, coord_kind="grid",
            labels=obs.label.values, patient=obs.patient.values,
            counts=C, lib_size_full=obs.lib_size.values,
            top_n=(10, 50), n_perm=200, n_boot=200, seed=0)

print(rep.headline.r_median)        # value, floor, CI, n -- always together
print(rep.headline.strongest_null)  # ('depth_null', Stat(...))
rep.to_html("audit.html")
rep.to_csv_dir("out/")
```

---

## Helper entry points

### `nulls_only`

```python
nulls_only(y_true, *, space=None, section=None, **kw) -> AuditReport
```

Alias for `audit(y_true, None, ...)`. Any `y_pred` in `**kw` is dropped.

```python
rep = nulls_only(y_true=Y, space="log1p_cp10k:panel",
                 section=sec, lib_size=lib, coords=xy, labels=lab)
print(rep.summary())     # first line: "STNULL 0.1.0 | NULLS-ONLY (no y_pred given)"
```

### `ladder`

```python
ladder(y_true, y_pred, *, space=None, section=None, lib_size=None,
       labels=None, **kw) -> pandas.DataFrame
```

Runs a full audit and returns only `rep.ladder`. Same cost as `audit()`. Use it
for convenience, not for speed.

Columns: `section`, `patient`, `n_spots`; then, for each of `r_full`, `dg`,
`crg`, `drg` and `drg_observed`, the value plus `_floor`, `_ci_lo` and `_ci_hi`;
then the boolean `drg_above_floor` (bootstrap CI lower bound above the
permutation floor) and `n_genes`.

### `perm_floor`

```python
perm_floor(y_true, y_pred, *, section, rule="all", top_n=None, kind="block",
           coords=None, coord_kind="grid", n_perm=200, seed=0) -> Stat
```

The permutation floor of one readout rule, standalone, without running an audit.
`rule` is `"all"` (median over all genes) or `"top_n"` (requires `top_n=<int>`,
and the floor is then `perm_selection`). `kind` is `"block"` or `"free"`; block
needs `coords`.

```python
blk = perm_floor(Y, P, section=sec, kind="block", coords=xy)
fre = perm_floor(Y, P, section=sec, kind="free")
print(blk, "\n", fre)   # on HER2ST: 0.049 vs 0.015 across 36 sections
```

### `compare`

```python
compare(*reps, on="headline") -> pandas.DataFrame
```

Puts several reports side by side, or **raises `NotComparable`** when their
`space`, gene aggregation, section aggregation, gene count or section count
differ. There is no approximate mode. Needs at least two reports
(`BadInput` otherwise).

### `check_inputs`

```python
check_inputs(y_true=None, y_pred=None, section=None, lib_size=None, coords=None,
             labels=None, composition=None, counts=None, patient=None,
             is_train=None, lib_pred=None, genes=None, lib_size_full=None,
             min_spots=50, min_genes=20, space=None, **ignored) -> InputReport
```

Structural validation only; never computes a correlation, so it returns in
seconds on a full dataset. `InputReport` fields: `ok`, `errors`, `warnings`,
`shapes`, `present`, `n_sections`, `small_sections`; `print()` renders it.

Since 0.3.0 it accepts `scipy.sparse` `y_true` / `y_pred` (it used to raise
`TypeError` from `np.isfinite` on the path `audit` itself supports), validates
`lib_size_full`, and treats a non-finite `lib_size`, `lib_size_full`, `coords`,
`composition` or `lib_pred` as an **error** rather than letting it reach
`lstsq` / `cKDTree` / `rng.binomial`.

### `cite_text`

```python
cite_text(rep=None) -> str
```

The Methods paragraph. With a report it is populated with that run's actual space,
aggregation, floor kind, permutation and bootstrap counts; without one it prints
the defaults. Same text as `stnull cite`.

---

## `AuditReport`

```python
@dataclass
class AuditReport:
    run:         RunMeta
    headline:    Headline | None          # None in nulls-only mode
    depth:       DepthResult | None
    composition: CompositionResult | None
    spatial:     SpatialResult | None
    selection:   SelectionResult | None
    ceiling:     CeilingResult | None
    ladder:      pd.DataFrame
    per_section: pd.DataFrame
    per_gene:    pd.DataFrame
    na:          dict[str, str]           # "composition" -> "labels not provided"
    warnings:    list[str]
```

### Methods

| method | returns | notes |
|---|---|---|
| `summary(lang="en")` | `str` | pure ASCII, safe on a Windows console. First line is always the headline next to the strongest null |
| `to_html(path, lang="en")` | `Path` | one self-contained file; figures inline as base64 when matplotlib is present, tables only when it is not |
| `to_markdown(path=None, lang="en")` | `str` | writes the file if `path` is given, always returns the text |
| `to_json(path=None)` | `dict` | every number plus provenance; machine-readable |
| `to_csv_dir(d)` | `list[Path]` | writes `per_section.csv`, `per_gene.csv`, `ladder.csv`, `claims.csv` |
| `claims()` | `list[Claim]` | fixed template sentences with a support flag; see [INTERPRETATION.md](INTERPRETATION.md) |
| `caveats()` | `list[str]` | degradations, N/A entries and warnings, flattened |
| `plot(which="ladder", ax=None)` | matplotlib axes | raises `RuntimeError` with a clear message when matplotlib is absent |

`lang="zh"` affects the HTML and Markdown files only. Console output is always
English ASCII.

### Table columns

`per_section` gives one row per section: `section`, `patient`, `n_spots`,
`n_eval` (spots the MODEL was scored on; equal to `n_spots` except under
`null_fit="train_only"`), `r_model`, `depth_null`, `depth_1d`, `cross_null`,
`oracle_null`,
`depth_comp_null`, `r_nbr`, `dg`, `crg`, `drg`, `drg_observed`, `moran`,
`smooth_delta`, `sd_loglib`, `depth_r_of_pred`.

`per_gene` gives one row per (section, gene): `section`, `gene`, `n_spot`,
`r_full`, `dg`, `crg`, `drg`.

`ladder`: see [`ladder`](#ladder) above.

---

## `Stat`

```python
@dataclass(frozen=True)
class Stat:
    value: float
    floor: float | None            # 95th percentile of the null of the SAME rule
    floor_kind: str                # perm_free | perm_block | perm_selection
                                   # | analytic_zero | none
    ci: tuple[float, float] | None # bootstrap 95%
    n_spots: int
    n_genes: int                   # genes that actually produced a finite r
    status: str                    # ok | na | underpowered | degraded | derived
    note: str = ""
```

The only number container in the package. It is frozen, and `__str__` /
`__repr__` always print value, floor, CI and n together:

```
0.1051  [perm_block floor 0.0319]  CI[0.0311,0.1086]  n=812/100  (ci_kind=section; perm med=-0.0014 max=0.0549 nperm=200)
```

Methods: `excess()` (`value - floor`, NaN when there is no floor),
`above_floor()` (CI lower bound above the floor when a CI exists, else the point
estimate; `None` when undecidable), `to_dict()`, and `float(stat)` for when you
genuinely need the bare number.

**Reading the floor.** `floor` is the one-sided 5% threshold (95th percentile) of
the null distribution. The median and max of that same distribution are in
`note`, because a threshold alone hides the shape of the null. The literature
more often quotes the null **median**, which is lower. On HER2ST the top-100
selection floor is 0.0818 as a free-permutation median and 0.1552 as a
block-permutation p95.

**No value without a floor.** If the floor could not be computed (`n_perm=0`, too
few genes, fewer than 5 finite null draws), `floor_kind` becomes `"none"`,
`status` becomes `"degraded"` or `"na"` and the note says so. There is no
"number now, floor later" path, and no renderer option to hide the floor column.

`n_genes` is the number of genes that produced a finite r in that readout, not
the panel size. Since 0.3.0 each null counts **its own** finite genes; before
that every null borrowed the model's count, which was 0 in nulls-only mode. Zero-variance genes score NaN, are excluded by `nanmedian`, and
are counted out here rather than silently changing the denominator.

---

## Result records

```python
@dataclass(frozen=True)
class RunMeta:
    stnull_version: str; seed: int; created_utc: str
    space: TargetSpace; space_note: str
    n_spots: int; n_genes: int; n_sections: int; n_patients: int | None
    inputs_present: dict[str, bool]      # lib_size / coords / labels / counts / ...
    inputs_sha1:    dict[str, str]       # 12-hex fingerprint of every input array
    null_fit: str; depth_proxy: str; perm_kind: str; n_perm: int; n_boot: int
    degradations: list[str]              # human-readable, each with its bias direction
    agg_gene: str = "median+mean"; agg_section: str = "median+mean"
```

```python
@dataclass
class Headline:
    r_median: Stat                       # per-gene r -> within-section median -> across-section median
    r_mean:   Stat
    strongest_null: tuple[str, Stat] | None      # rendered on the headline row, not separable
    frac_sections_model_beats_null: Stat         # the "0/36" readout
    aggregation_note: str
```

```python
@dataclass
class DepthResult:                       # needs lib_size
    depth_null: Stat                     # y ~ [1, log lib], zero pixels
    depth_1d: Stat | None                # image -> 1 predicted depth -> genes;
                                         # N/A under depth_proxy="observed"
    dg: Stat                             # partial r controlling depth only
    share_depth: Stat                    # 1 - dg/r_full, status="derived";
                                         # sections with |r_full| < 0.01 are
                                         # excluded from the ratio (0.3.0)
    depth_r_of_pred: Stat                # corr(l_hat, log lib); N/A under
                                         # depth_proxy="observed", where it is
                                         # 1 by construction
    sd_loglib_by_section: pd.DataFrame
    trend_sd_vs_r: Stat | None           # None below min_sections_for_trend
    lib_denominator_check: str           # panel row sum vs full transcriptome reminder
    target_carries_depth: bool           # True for space="log1p:raw"
```

```python
@dataclass
class CompositionResult:                 # needs labels or composition
    cross_null: Stat                     # class means, cross-fitted
    oracle_null: Stat                     # class means read off this section, UPPER_BOUND
    crg: Stat
    n_classes: int; class_counts: dict[str, int]
    small_classes_dropped: list[str]     # merged into "other"
    kind: str                            # "labels (one-hot)" | "continuous composition"
```

```python
@dataclass
class SpatialResult:                     # needs coords
    r_nbr: Stat                          # mean of the neighbours' TRUE y
    frac_sections_nbr_beats_model: Stat
    smooth_lever: Stat                   # delta from 3x3 smoothing the ground
                                         # truth. Since 0.3.0 the operator is
                                         # the unweighted box mean (1/9 on the
                                         # centre and on each neighbour); it
                                         # used to be standardised twice, which
                                         # left the centre at 1/2 and
                                         # understated the lever severalfold
    moran_I_resid: Stat                  # spatial autocorrelation of the model residual
    perm_block_vs_free: tuple[Stat, Stat] | None
```

```python
@dataclass
class SelectionResult:
    by_n: dict[int, Stat]                # every floor_kind is "perm_selection"
    train_selected: dict[int, Stat] | None       # needs is_train
    lever_price: pd.DataFrame            # lever, value_base, value_lever, delta,
                                         # floor_base, floor_lever, floor_kind, status, note
    leakage_lever: Stat | None           # zero-pixel lower bound, no floor
```

```python
@dataclass
class CeilingResult:
    r_tech: Stat | None
    r_nbr: Stat | None                   # same value as SpatialResult, for contrast
    r_over_ceiling: Stat | None
    method: str                          # split_half_full | split_half_panel | na
```

`r_tech` is a bound on *counting* noise only, and the Spearman-Brown step
assumes the two half-depth replicates are parallel tests on a linear scale.
log1p breaks that at ST-typical sparsity: below a mean of about 5 counts per
spot-gene the estimate runs roughly +0.06 to +0.11 high (Poisson simulation
against a Monte-Carlo Bayes-optimal bound). Since 0.3.0 the Stat's note carries
the measured mean count and that caveat whenever it applies, and the
permutation floor follows the same averaging rule as the value (it used to be
built from a single replicate, which left it about 38% too high).

```python
@dataclass(frozen=True)
class Claim:
    text_en: str; text_zh: str
    support: str                         # supported | not_supported | out_of_scope
    evidence: dict
```

---

## `TargetSpace`

```python
class TargetSpace(str, Enum):
    LOG1P_CP10K_PANEL = "log1p_cp10k:panel"   # denominator = panel row sum
    LOG1P_CP10K_FULL  = "log1p_cp10k:full"    # denominator = full-transcriptome lib
    LOG1P_RAW         = "log1p:raw"           # HEST convention -- target carries depth
    LOG10_MEDLIB      = "log10:median_lib"    # HisToGene / scprep convention
    ZSCORE            = "zscore:per_gene"
    CUSTOM            = "custom"              # requires space_note="..."
```

Pass the string or the enum member. `space="log1p:raw"` makes the report print a
`TARGET_CARRIES_DEPTH` banner above section 2: in that space `depth_null` is high
by construction, and only DG is a comparable number. `space="custom"` also
disables `r_tech`, because stnull cannot replicate a transform it does not know.

---

## Exceptions

All inherit from `StnullError`.

| exception | raised when |
|---|---|
| `MissingSpace` | `space` absent or unrecognised, or `custom` without `space_note` |
| `BadInput` | shape / dtype / argument problems, e.g. `max(top_n) > n_gene`, `coords` not `(n, 2)`, unknown `perm_floor(rule=...)` |
| `InsufficientData` | `strict=True` and validation failed (message embeds the whole `InputReport`) |
| `LibSizeLooksLikePanelSum` | `lib_size` correlates > 0.999 with the row sum of `y_true` |
| `NotComparable` | `compare()` on reports with different space / aggregation / panel / split |

---

## CLI

```
stnull audit  --true Y.npy --pred P.npy --obs obs.parquet --space log1p_cp10k:panel
              [--space-note TEXT] --section-col section
              [--lib-col lib_size] [--coord-cols arr_x,arr_y] [--coord-kind grid|micron]
              [--label-col label] [--comp-cols c1,c2,...] [--patient-col patient]
              [--train-col is_train] [--libpred-col lib_pred]
              [--counts counts.npz] [--libfull-col lib_full] [--genes genes.txt]
              [--top-n 10,50] [--perm 200] [--boot 500]
              [--perm-kind auto|block|free] [--depth-proxy observed|from_pred|given|none]
              [--null-fit cv_within_section|train_only|oracle] [--min-spots 50]
              [--seed 0] [--jobs 1] [--lang en|zh]
              [--out audit.html] [--csv-dir out/] [--json audit.json]
              [--no-strict] [--fail-on-degraded] [--quiet]
stnull nulls  <same, without --pred>
stnull levers <same; prints the protocol lever price list>
stnull check  <same; validates the inputs only>
stnull cite
```

`--true` / `--pred` / `--counts` accept `.npy`, `.npz` (one 2-D array, or a saved
sparse matrix), `.csv`, `.tsv`, `.parquet`. `--obs` is a per-spot table
(`.parquet` / `.csv` / `.tsv`) and every `*-col` flag names a column in it.
`--out` dispatches on the suffix (`.html` / `.md` / `.json`).

Exit codes: `0` ok · `2` strict validation failed · `3` degraded with
`--fail-on-degraded` · `4` inputs unreadable.

Invoke as `stnull ...` (console script) or `python -m stnull.cli ...`.

---

## Design decisions

**D1 `space` has no default.** The same zero-pixel composition null scores
0.007-0.083 r higher under a panel-row-sum denominator than under a
full-transcriptome one. That gap exceeds most single-step advances in this
literature, so a default would be stnull inventing comparability on your behalf.
This is why the five-line quickstart carries one extra keyword argument.

**D2 stnull never transforms your data.** There is no `normalize=`. A tool that
can move targets between spaces will be used to find the space where the number
looks best, and it cannot distinguish that from honest alignment. The one
internal transform is `ceiling.apply_space`, used solely to build the measurement
replicate; with `space="custom"` even that is refused.

**D3 `Stat` is the only number container, and there is no value without a
floor.** No `hide_nulls=`, no value-only export. Measured justification: a top-50
readout of 0.325 whose same-rule selection floor is 0.080 is not a small
correction, it is a fourfold overstatement.

**D4 Floors are named because they are not interchangeable.** `perm_free`
(shuffle spots), `perm_block` (torus shift on the lattice, preserves spatial
autocorrelation, the default when `coords` are given), `perm_selection` (the same
top-N rule applied to permuted r), `analytic_zero`. Residualised rungs permute
only the residual spot order with the design matrix held fixed, so the null
carries no depth or composition structure.

**D5 The DRG wording is hard-coded** and three readings are permanently
`out_of_scope`. See [INTERPRETATION.md](INTERPRETATION.md).

**D6 The strongest null is glued to the headline.** `headline.strongest_null` is
a tuple rendered on the same row, and `frac_sections_model_beats_null` is
reported next to it, because a median hides "the model won zero sections".

**D7 The oracle null is reported but labelled `UPPER_BOUND`** and is barred from
being the strongest null. It is the ceiling of any pure composition model, not a
competitor's score.

**D8 The ceiling is N/A rather than approximated.** A Poisson stand-in for
`r_tech` would bias *against* the model, which is the safe direction, but it
would still be a fabricated number.

**D9 Degradations are booked with a direction of bias**, and
`--fail-on-degraded` lets CI treat an under-specified audit as a failure. (The
0.1.0 gap, micron coordinates degrading without a ledger entry, was closed
in 0.2.0: the lattice check books it, as do free-permutation fallbacks and
duplicate coordinates.)

**D10 Dependencies and cost.** numpy / pandas / scipy only; matplotlib optional
and only for figures; no sklearn, since ridge and partial correlation are SVD-based
and rank-deficiency safe. Every per-gene computation is vectorised; there is no
per-gene Python loop. Measured: 812 spots x 100 genes x 3 sections with
`n_perm=200, n_boot=200, n_rep_ceiling=10` takes about 90 s, roughly half of it
in `r_tech`. The full 13,580 x 737 x 36 case has **not** been timed; extrapolation
suggests it exceeds the ten-minute design target, so start with `n_boot=100`.

**Deliberately not done:** model training, feature extraction, image reading, a
composite score, a built-in table of published numbers, and any API that
collapses the report into a single number.
