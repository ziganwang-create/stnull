# -*- coding: utf-8 -*-
"""stnull.report: result containers, claim wording and renderers.

Author: Zigan Wang.

WHAT THIS MODULE IS
    The output half of the package.  It holds the dataclasses that an audit
    returns (``RunMeta``, ``Headline``, ``DepthResult``, ``CompositionResult``,
    ``SpatialResult``, ``SelectionResult``, ``CeilingResult``, ``AuditReport``),
    the sentences the run is and is not entitled to (``AuditReport.claims``),
    and the console / Markdown / HTML / JSON / CSV renderers.

WHO CALLS IT
    ``audit`` builds these objects; the CLI and the user render them.  Nothing
    here computes a statistic, if a number is missing at this point, it is
    missing because the audit could not compute it, and the reason travels
    inside the Stat.

KEY ASSUMPTIONS
    * The renderers have no option to hide a floor column.  That is
      deliberate: a value without its null floor is the failure mode this
      package exists to stop.
    * ``claims()`` is hard-coded, including three readings permanently marked
      ``out_of_scope``.  Wording is part of the audit: a correct number under a
      wrong sentence is still a wrong result.
    * Console output is ASCII-only, so a report prints on a cp936 terminal;
      the Chinese wording lives in the HTML/Markdown renderers.

DATA LINEAGE (row numbers refer to docs/CODE_WALKTHROUGH.md (section 4, the variable lineage table))
    Inputs (all built inside audit.py's assembly half, audit.py:1105-1449)
        [L40] Stat objects (value / floor / ci triples from core.make_stat):
              every number a renderer prints arrives as one of these; the
              floor is the permutation floor, the score the same readout
              rule reaches on shuffled data, so anything at or below it is
              indistinguishable from noise.
        [L41] Headline.strongest_null, the best out-of-sample zero-pixel
              null (a "null" here = a predictor that never sees the image).
        [L42] per_section, [L43] per_gene, [L44] ladder: the three
              DataFrames framed at audit.py:1043-1100.
        [L45] SelectionResult.by_n and [L46] leakage_lever feed section 5.
        [L47] CeilingResult.r_tech feeds section 6.
        [L48] degradations, [L49] warnings, [L50] na: the three ledgers
              that caveats() merges into section 9.
        [L51] RunMeta.inputs_sha1, the reproducibility fingerprints of
              section 0 and the JSON run card.
    Outputs
        summary() -> console text (cli._emit prints it); to_json() ->
        dict/file via Stat.to_dict [L40]; to_markdown()/to_html() ->
        files the CLI --out flag routes to; to_csv_dir() -> the [L42]/
        [L43]/[L44] tables plus claims.csv.  Nothing flows back into
        audit.py: this module is a sink.
"""
from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .core import Stat, na_stat
from .spaces import DEPTH_CARRYING, TargetSpace

VERSION = "0.3.0"


# ------------------------------------------------------------------ records
@dataclass(frozen=True)
class RunMeta:
    """The run card: what was audited, under which protocol, with which seed.

    ``degradations`` is the ledger: one sentence per optional input that was
    missing or per fallback that fired, each stating the DIRECTION in which it
    biases the headline (decision D9).

    WHERE: built once at audit.py:1430-1445; ``degradations`` is the list
    seeded at audit.py:779 [L48], ``inputs_sha1`` the fingerprints of
    audit.py:1440-1443 [L51].  Rendered in section 0 of every format.
    """

    stnull_version: str
    seed: int
    created_utc: str
    space: TargetSpace
    space_note: str
    n_spots: int
    n_genes: int
    n_sections: int
    n_patients: Optional[int]
    inputs_present: Dict[str, bool]
    inputs_sha1: Dict[str, str]
    null_fit: str
    depth_proxy: str
    perm_kind: str
    n_perm: int
    n_boot: int
    degradations: List[str] = field(default_factory=list)
    agg_gene: str = "median+mean"
    agg_section: str = "median+mean"


@dataclass
class Headline:
    """Section 1: the model r, both aggregations, and the strongest null.

    ``strongest_null`` is (name, Stat) for the highest-scoring OUT-OF-SAMPLE
    zero-pixel null; in-sample (oracle) nulls are barred from this slot
    (decision D7).

    WHERE: assembled at audit.py:1105-1153; ``r_median``/``r_mean`` come
    off the r_per_gene chain [L25] through _combine [L39], and
    ``strongest_null`` is [L41], picked at audit.py:1127-1129.
    """

    r_median: Stat
    r_mean: Stat
    strongest_null: Optional[Tuple[str, Stat]]
    frac_sections_model_beats_null: Stat
    aggregation_note: str = (
        "per-gene Pearson r -> within-section median AND mean -> "
        "across-section median AND mean; the literature mixes the two, so both "
        "are printed")


@dataclass
class DepthResult:
    """Section 2: everything about sequencing depth.

    ``depth_1d`` and ``depth_r_of_pred`` are statements about the IMAGE
    pipeline and exist only when l_hat came from the model
    (``depth_proxy='from_pred'`` or ``'given'``); under the default
    ``'observed'`` they are N/A with that reason attached, because l_hat is
    then the measured log library size and both readouts would be fixed by
    construction.

    WHERE: assembled at audit.py:1160-1232.  ``depth_null`` rides the
    [L25] chain from nulls.depth_null; ``dg`` is the ladder's DG rung
    [L31]; ``depth_r_of_pred`` is [L22]; ``sd_loglib_by_section`` derives
    from loglib [L05] at audit.py:998.
    """

    depth_null: Stat
    depth_1d: Optional[Stat]
    dg: Stat
    share_depth: Stat
    depth_r_of_pred: Stat
    sd_loglib_by_section: pd.DataFrame
    trend_sd_vs_r: Optional[Stat]
    lib_denominator_check: str
    target_carries_depth: bool = False


@dataclass
class CompositionResult:
    """Section 3: what a class-mean lookup table already predicts.

    ``cross_null`` is cross-fitted (honest); ``oracle_null`` reads the class
    means off the section being scored and is an UPPER_BOUND, barred from the
    headline.  Classes below ``min_class_spots`` are merged into 'other' and
    listed in ``small_classes_dropped``.

    WHERE: assembled at audit.py:1233-1245 from the composition-null arm
    of the [L25] chain (design built from Comp [L07]); ``crg`` is the
    ladder's CRG rung [L31].
    """

    cross_null: Stat
    oracle_null: Stat
    crg: Stat
    n_classes: int
    class_counts: Dict[str, int]
    small_classes_dropped: List[str]
    kind: str = "labels"


@dataclass
class SpatialResult:
    """Section 4: what spatial autocorrelation alone is worth.

    ``smooth_lever`` is the delta in the headline when the GROUND TRUTH is
    replaced by its unweighted 3x3 box mean and the model is not retrained.
    ``perm_block_vs_free`` shows the same value under both floors, because the
    free-permutation floor common in this literature is anticonservative here.

    WHERE: assembled at audit.py:1246-1275; ``r_nbr`` is [L30],
    ``smooth_lever`` comes from the smoother chain [L29], and
    ``perm_block_vs_free`` pairs the block floor with the free floor of
    [L37].
    """

    r_nbr: Stat
    frac_sections_nbr_beats_model: Stat
    smooth_lever: Stat
    moran_I_resid: Stat
    perm_block_vs_free: Optional[Tuple[Stat, Stat]]


@dataclass
class SelectionResult:
    """Section 5: gene selection, smoothing and split granularity, priced.

    ``by_n`` is the test-selected readout, whose floor is the SAME selection
    rule applied to permuted predictions; ``train_selected`` is the legal
    version and exists only with ``is_train``.  ``leakage_lever`` is a lower
    bound measured on a zero-pixel null and carries no permutation floor.

    WHERE: assembled at audit.py:1280-1375; ``by_n`` is [L45] (built from
    the [L33] per-gene vectors via levers.topn_value/topn_null),
    ``leakage_lever`` is [L46], and ``lever_price`` is the price table of
    audit.py:1329-1353.
    """

    by_n: Dict[int, Stat]
    train_selected: Optional[Dict[int, Stat]]
    lever_price: pd.DataFrame
    leakage_lever: Optional[Stat]


@dataclass
class CeilingResult:
    """Section 6: the measurement ceiling and the model's fraction of it.

    ``method`` is ``split_half_full`` / ``split_half_panel`` / ``na``.  At
    ST-typical sparsity the estimate is an approximate bound that runs high;
    the Stat's own note carries the measured direction and size.

    WHERE: assembled at audit.py:1380-1426 from the r_tech chain [L47]
    (ceiling.r_tech on the raw counts [L08]); ``r_nbr`` duplicates [L30]
    here as a spatial reference.
    """

    r_tech: Optional[Stat]
    r_nbr: Optional[Stat]
    r_over_ceiling: Optional[Stat]
    method: str


@dataclass(frozen=True)
class Claim:
    """One sentence, in English and Chinese, with its support verdict.

    ``support`` is ``supported`` | ``not_supported`` | ``out_of_scope``, and
    ``evidence`` carries the Stats (or the reason) behind the verdict.
    """

    text_en: str
    text_zh: str
    support: str          # supported | not_supported | out_of_scope
    evidence: dict


# --------------------------------------------------------------- the report
@dataclass
class AuditReport:
    """Everything one audit produced: Stats, tables, claims and the ledger.

    ``na`` maps a report section to the reason it is absent; ``warnings`` and
    ``run.degradations`` are joined by :meth:`caveats`, which is what every
    renderer prints in section 9.

    WHERE: the single return value of audit(), built at audit.py:1449-1450.
    ``ladder`` is [L44], ``per_section`` [L42], ``per_gene`` [L43],
    ``na`` [L50], ``warnings`` [L49].  Consumers: cli._emit and the user.
    """

    run: RunMeta
    headline: Optional[Headline]
    depth: Optional[DepthResult]
    composition: Optional[CompositionResult]
    spatial: Optional[SpatialResult]
    selection: Optional[SelectionResult]
    ceiling: Optional[CeilingResult]
    ladder: pd.DataFrame
    per_section: pd.DataFrame
    per_gene: pd.DataFrame
    na: Dict[str, str] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    # ------------------------------------------------------------- claims
    def claims(self) -> List[Claim]:
        """The sentences this run supports, and the ones it never will.

        Returns a list of :class:`Claim`, each tagged ``supported``,
        ``not_supported`` or ``out_of_scope``.  The out-of-scope entries are
        constant: they are readings of DRG that no run can license, and they
        are printed whether or not the current numbers look good.
        """
        out: List[Claim] = []

        # WHAT: helper turning a Stat into a JSON-safe evidence record.
        # WHERE: Stat.to_dict is core.py:184 (flow [L40]); the "stat" key
        # names which report field the number came from.
        def ev(path, s: Optional[Stat]):
            if s is None:
                return {"stat": path, "value": None}
            d = s.to_dict()
            d["stat"] = path
            return d

        # --- claim 1: model vs the strongest zero-pixel null -----------
        # WHAT: only emitted when both numbers are finite; the sentence
        # states which side won instead of leaving it to the reader.
        # WHERE: h.r_median rides the [L25] chain; sn is strongest_null
        # [L41]; the beats fraction is the win-rate Stat from
        # audit.py:1132-1141.
        h = self.headline
        if h is not None and self.run.n_sections:
            sn = h.strongest_null
            if sn is not None and np.isfinite(sn[1].value) and np.isfinite(h.r_median.value):
                beats = h.r_median.value > sn[1].value  # True: model above null
                out.append(Claim(
                    text_en=("Across %d sections the model's median per-gene r is "
                             "%.4f, %s the strongest zero-pixel null tried here "
                             "(%s = %.4f). Model beats that null in %s of sections."
                             % (self.run.n_sections, h.r_median.value,
                                "above" if beats else "BELOW", sn[0], sn[1].value,
                                _fracstr(h.frac_sections_model_beats_null))),
                    text_zh=("%d 张切片上,模型逐基因中位 r = %.4f,%s本次跑出的最强零像素零模型"
                             "(%s = %.4f);模型在 %s 的切片上赢过该零模型。"
                             % (self.run.n_sections, h.r_median.value,
                                "高于" if beats else "低于", sn[0], sn[1].value,
                                _fracstr(h.frac_sections_model_beats_null))),
                    support="supported",
                    evidence={"headline": ev("headline.r_median", h.r_median),
                              "null": ev("strongest_null", sn[1])}))
        # --- claim 2: the depth-controlled r (DG) ----------------------
        # WHAT: what remains after sequencing depth is partialled out of
        # both truth and prediction.
        # WHERE: d.dg is the DG rung of the ladder [L31], aggregated
        # through _combine [L39] at audit.py:1185; its floor and CI are
        # the rung's own permutation floor [L32] and bootstrap CI [L34].
        d = self.depth
        if d is not None and np.isfinite(d.dg.value):
            out.append(Claim(
                text_en=("After controlling for sequencing depth, the predictions "
                         "of this model retain r = %.4f (%s floor %s, 95%% CI %s)."
                         % (d.dg.value, d.dg.floor_kind, _f(d.dg.floor), _ci(d.dg))),
                text_zh=("在控制测序深度之后,**该模型的预测**还剩下 r = %.4f"
                         "(置换底线 %s,95%% CI %s)。"
                         % (d.dg.value, _f(d.dg.floor), _ci(d.dg))),
                support="supported", evidence=ev("depth.dg", d.dg)))
        # --- claim 3: the fully controlled r (DRG) ---------------------
        # WHAT: median DRG across sections, read straight off the ladder
        # table, plus how many sections clear their own permutation floor.
        # WHERE: lad is the ladder DataFrame [L44] (rows built at
        # audit.py:1081-1099); drg_above_floor is the per-section CI vs
        # floor comparison of audit.py:1095-1097 ([L34]).
        lad = self.ladder
        drg = None  # across-section median DRG, or None when unscorable
        if lad is not None and len(lad) and "drg" in lad.columns:
            v = lad["drg"].astype(float)
            if np.isfinite(v).any():
                drg = float(np.nanmedian(v))
        if drg is not None:
            ctrl = "depth and composition"
            # n_pass: sections whose bootstrap CI lower bound clears the floor
            n_pass = int(lad.get("drg_above_floor", pd.Series(dtype=bool)).fillna(False).sum()) \
                if "drg_above_floor" in lad.columns else 0
            out.append(Claim(
                text_en=("After controlling for %s, THE PREDICTIONS OF THIS MODEL "
                         "retain a median r = %.4f (%d/%d sections have a bootstrap "
                         "CI lower bound above their permutation floor). This is a "
                         "property of these predictions, not an upper bound on the "
                         "information content of H&E."
                         % (ctrl, drg, n_pass, int(lad["drg"].notna().sum()))),
                text_zh=("在控制深度与成分后,**该模型的预测**还剩下中位 r = %.4f"
                         "(%d/%d 张切片的 bootstrap CI 下界高于其置换底线)。"
                         "这是**这组预测**的性质,不是 H&E 信息量的上界。"
                         % (drg, n_pass, int(lad["drg"].notna().sum()))),
                support="supported", evidence={"drg_median": drg}))
        # --- claims 4-6: permanently forbidden readings ----------------
        # WHAT: three sentences a user might be tempted to write and never
        # may, emitted unconditionally with the reason as evidence.
        # WHY: printing them only when the numbers look bad would make the
        # report read as advocacy; printing them always makes it an audit.
        out.append(Claim(
            text_en=("H&E carries no information beyond depth and composition."),
            text_zh="H&E 在深度与成分之外没有信息。",
            support="out_of_scope",
            evidence={"why": "DRG measures a model, not the data. In the "
                             "reference dataset DRG rose with backbone quality "
                             "(8/8 sections), which cannot happen if DRG were a "
                             "property of the data."}))
        out.append(Claim(
            text_en="DRG close to zero proves the task is impossible.",
            text_zh="DRG 接近 0 证明这个任务做不成。",
            support="out_of_scope",
            evidence={"why": "A null result for one predictor is not a null "
                             "result for the hypothesis class."}))
        out.append(Claim(
            text_en=("DRG above its floor proves the model learned genuine "
                     "morphology."),
            text_zh="DRG 高于底线证明模型学到了真正的形态学信号。",
            support="out_of_scope",
            evidence={"why": "Above-floor residual information is statistically "
                             "distinguishable, not mechanistically morphological; "
                             "that needs a separate experiment."}))
        # --- claim 7: cross-paper comparison, marked not_supported -----
        # WHERE: the honest route is stnull.compare() (audit.py:1556,
        # flow [L56]), which refuses to compare runs whose RunMeta
        # protocol keys differ instead of quietly aligning them.
        out.append(Claim(
            text_en=("Our r is higher than the number reported in paper X."),
            text_zh="我们的 r 比某论文高。",
            support="not_supported",
            evidence={"why": "Cross-paper comparison requires identical space, "
                             "panel, split and aggregation. Use stnull.compare(), "
                             "which raises NotComparable when they differ."}))
        return out

    def caveats(self) -> List[str]:
        """The degradation ledger, the N/A reasons and the warnings, in order.

        WHERE: merges the three ledgers audit() filled: degradations [L48]
        (seeded audit.py:779), na [L50] (audit.py:773) and warnings [L49]
        (audit.py:772).  Consumed by section 9 of every renderer and by
        cli._emit's --fail-on-degraded exit-code check.
        """
        c = list(self.run.degradations)         # copy: callers may extend
        for k, v in sorted(self.na.items()):    # sorted: deterministic output
            c.append("section '%s' is N/A: %s" % (k, v))
        c.extend(self.warnings)
        return c

    # ------------------------------------------------------------ renderers
    def summary(self, lang="en") -> str:
        """The console report: ten numbered sections, ASCII only.

        ``lang`` is accepted for symmetry with the file renderers but the
        console text is always English, a cp936 terminal cannot be trusted
        with anything else.
        """
        L = []  # output lines, joined once at the end
        # --- banner: the one comparison a reader must not miss ----------
        # WHAT: model median r beside the strongest zero-pixel null [L41],
        # before any detail; a nulls-only run says so instead.
        h = self.headline
        if h is None:
            L.append("STNULL %s | NULLS-ONLY (no y_pred given)" % VERSION)
        else:
            sn = h.strongest_null
            L.append("model median r = %s" % _f(h.r_median.value))
            L.append("  vs strongest zero-pixel null (%s) = %s"
                     % (sn[0] if sn else "none", _f(sn[1].value) if sn else "n/a"))
            L.append("  model beats that null in %s of sections"
                     % _fracstr(h.frac_sections_model_beats_null))
        # --- run card: protocol before numbers --------------------------
        # WHY: r values are not interpretable without space, permutation
        # kind and null-fit mode; printing them first makes any copy-paste
        # of the report carry its own protocol.
        L.append("-" * 72)
        L.append("stnull %s | space=%s | %d spots x %d genes x %d sections | seed=%d"
                 % (VERSION, self.run.space.value, self.run.n_spots,
                    self.run.n_genes, self.run.n_sections, self.run.seed))
        L.append("perm=%s x%d | boot x%d | null_fit=%s | depth_proxy=%s"
                 % (self.run.perm_kind, self.run.n_perm, self.run.n_boot,
                    self.run.null_fit, self.run.depth_proxy))
        L.append("")
        # --- [1] headline: both aggregations, as full Stats -------------
        # WHY both: the literature mixes median-of-genes and mean-of-genes,
        # and the two can differ by 0.05 or more on the same predictions.
        # WHERE: each Stat prints via Stat.__str__ (core.py:166), which
        # always shows value, floor and CI together [L40].
        if h is not None:
            L.append("[1] HEADLINE")
            L.append("    median-of-genes : %s" % h.r_median)
            L.append("    mean-of-genes   : %s" % h.r_mean)
        # --- [2] depth: the sequencing-depth audit ----------------------
        # WHERE: DepthResult, assembled audit.py:1160-1232; each absent
        # section prints its N/A reason from the na ledger [L50] instead
        # of silently vanishing.
        if self.depth is not None:
            L.append("[2] DEPTH")
            if self.depth.target_carries_depth:
                L.append("    !! TARGET_CARRIES_DEPTH: space=log1p:raw, a high "
                         "depth null is the definition of the space, not a bug")
            L.append("    depth_null      : %s" % self.depth.depth_null)
            if self.depth.depth_1d is not None:
                L.append("    depth_1d(img)   : %s" % self.depth.depth_1d)
            L.append("    DG (ctrl depth) : %s" % self.depth.dg)
            L.append("    share of r from depth : %s" % _f(self.depth.share_depth.value))
            # printed as a full Stat, not just a value: under the default depth
            # proxy this row is N/A and the reason is the informative part
            L.append("    corr(l_hat, log lib)  : %s" % self.depth.depth_r_of_pred)
            if self.depth.trend_sd_vs_r is not None:
                L.append("    corr(sd(log lib), r)  : %s" % _f(self.depth.trend_sd_vs_r.value))
        else:
            L.append("[2] DEPTH : N/A -- %s" % self.na.get("depth", "lib_size not provided"))
        # --- [3] composition: what a class-mean lookup already predicts -
        # WHERE: CompositionResult (audit.py:1233-1245); the oracle null
        # is tagged UPPER_BOUND because it reads class means off the very
        # section being scored.
        if self.composition is not None:
            L.append("[3] COMPOSITION (%d classes)" % self.composition.n_classes)
            L.append("    cross null      : %s" % self.composition.cross_null)
            L.append("    oracle null     : %s <UPPER_BOUND>" % self.composition.oracle_null)
            L.append("    CRG (ctrl comp) : %s" % self.composition.crg)
        else:
            L.append("[3] COMPOSITION : N/A -- %s"
                     % self.na.get("composition", "labels/composition not provided"))
        # --- [4] spatial: what smoothing alone is worth -----------------
        # WHERE: SpatialResult (audit.py:1246-1275); r_nbr is the
        # neighbour null [L30], the smoothing lever comes from [L29].
        if self.spatial is not None:
            L.append("[4] SPATIAL")
            L.append("    r_nbr (true-y smoothing, no pixels) : %s" % self.spatial.r_nbr)
            L.append("    nbr beats model in %s of sections"
                     % _fracstr(self.spatial.frac_sections_nbr_beats_model))
            L.append("    lever: smoothing the ground truth   : %s"
                     % _f(self.spatial.smooth_lever.value))
            L.append("    Moran's I of model residual         : %s"
                     % _f(self.spatial.moran_I_resid.value))
        else:
            L.append("[4] SPATIAL : N/A -- %s" % self.na.get("spatial", "coords not provided"))
        # --- [5] selection: test-selected vs train-selected top-N -------
        # WHERE: SelectionResult.by_n [L45] and train_selected; the floor
        # of a test-selected row is the same selection rule applied to
        # permuted predictions, which is why it sits well above zero.
        if self.selection is not None:
            L.append("[5] SELECTION AND LEVERS")
            for n, s in sorted(self.selection.by_n.items()):
                L.append("    test-selected top-%-4d : %s" % (n, s))
            if self.selection.train_selected:
                for n, s in sorted(self.selection.train_selected.items()):
                    L.append("    train-selected top-%-3d : %s" % (n, s))
            if self.selection.leakage_lever is not None:
                L.append("    leakage lever (null-model) : %s" % self.selection.leakage_lever)
        # --- [6] ceiling: the measurement-noise bound -------------------
        # WHERE: CeilingResult from the r_tech chain [L47].
        if self.ceiling is not None:
            L.append("[6] CEILING (%s)" % self.ceiling.method)
            if self.ceiling.r_tech is not None:
                L.append("    r_tech          : %s" % self.ceiling.r_tech)
            if self.ceiling.r_over_ceiling is not None:
                L.append("    r / r_tech      : %s" % _f(self.ceiling.r_over_ceiling.value))
        # --- [7] ladder: r_full -> DG -> CRG -> DRG ---------------------
        # WHERE: the ladder DataFrame [L44]; the console shows only the
        # across-section median per rung, the full table is in the files.
        # (Section [8], the claims, is a file-renderer section; the
        # console keeps its numbering and skips it.)
        if self.ladder is not None and len(self.ladder):
            L.append("[7] ATTRIBUTION LADDER (across-section median)")
            for c in ("r_full", "dg", "crg", "drg"):
                if c in self.ladder.columns and self.ladder[c].notna().any():
                    L.append("    %-7s : %s" % (c, _f(float(np.nanmedian(
                        self.ladder[c].astype(float))))))
        # --- [9] the ledger, always printed, even when empty ------------
        # WHERE: caveats() merges [L48]/[L50]/[L49]; _ascii guards against
        # a non-ASCII degradation message breaking a cp936 console.
        L.append("[9] DEGRADATION LEDGER")
        cav = self.caveats()
        if not cav:
            L.append("    none -- all optional inputs present")
        for c in cav:
            L.append("    - %s" % _ascii(c))
        L.append("")
        L.append("This report evaluates THESE PREDICTIONS, not the information "
                 "content of H&E.")
        return "\n".join(L)

    def to_json(self, path=None) -> dict:
        """Machine-readable report; every Stat keeps floor, CI, n and status.

        WHERE: Stats serialise through _sec_json -> Stat.to_dict
        (core.py:184, flow [L40]); DataFrames through _df_json.  The
        per_gene table is deliberately absent (it can be huge; use
        to_csv_dir).  Consumers: --json / --out x.json in cli._emit, and
        any downstream meta-analysis script.
        """
        # WHAT: assemble one plain dict; the space enum is replaced by its
        # string value so the JSON needs no knowledge of TargetSpace.
        d = {
            "stnull_version": VERSION,
            "run": {**{k: v for k, v in self.run.__dict__.items()
                       if k not in ("space",)},
                    "space": self.run.space.value},
            "headline": _sec_json(self.headline),
            "depth": _sec_json(self.depth),
            "composition": _sec_json(self.composition),
            "spatial": _sec_json(self.spatial),
            "selection": _sec_json(self.selection),
            "ceiling": _sec_json(self.ceiling),
            "ladder": _df_json(self.ladder),
            "per_section": _df_json(self.per_section),
            "na": self.na,
            "warnings": self.warnings,
            "claims": [c.__dict__ for c in self.claims()],
        }
        if path:
            # WHY default=_default: numpy scalars, arrays, enums and Paths
            # are not JSON-native; _default converts them, mapping any
            # non-finite float to null (JSON has no NaN).
            Path(path).write_text(json.dumps(d, indent=2, ensure_ascii=False,
                                             default=_default), encoding="utf-8")
        return d

    def to_csv_dir(self, d) -> List[Path]:
        """Write per_section / per_gene / ladder / claims as CSV into ``d``.

        WHERE: the three DataFrames are [L42], [L43] and [L44]; this is
        the only renderer that writes per_gene in full.  Called by the
        CLI --csv-dir flag (cli._emit) and by the example script.
        """
        d = Path(d)
        d.mkdir(parents=True, exist_ok=True)
        out = []  # paths written, returned so the caller can echo them
        for name, df in (("per_section", self.per_section),
                         ("per_gene", self.per_gene),
                         ("ladder", self.ladder)):
            if df is not None and len(df):
                p = d / ("%s.csv" % name)
                df.to_csv(p, index=False, encoding="utf-8")
                out.append(p)
        # WHAT: claims become one CSV row each, evidence flattened to a
        # JSON string so the verdicts survive alongside the numbers.
        cl = pd.DataFrame([{"support": c.support, "text_en": c.text_en,
                            "text_zh": c.text_zh,
                            "evidence": json.dumps(c.evidence, ensure_ascii=False,
                                                   default=_default)}
                           for c in self.claims()])
        p = d / "claims.csv"
        cl.to_csv(p, index=False, encoding="utf-8")
        out.append(p)
        return out

    def to_markdown(self, path=None, lang="en") -> str:
        """Markdown report; ``lang='zh'`` switches the claim wording only.

        WHERE: delegates to _render_md below; the CLI routes --out x.md
        here.  The Markdown embeds the console summary verbatim, so the
        two formats can never diverge on a number.
        """
        md = _render_md(self, lang)
        if path:
            Path(path).write_text(md, encoding="utf-8")
        return md

    def to_html(self, path, lang="en") -> Path:
        """One self-contained HTML file (inline CSS, embedded figures).

        WHERE: delegates to _render_html below; the CLI routes --out
        x.html here and the example writes the README's sample report
        through it.
        """
        html = _render_html(self, lang)
        p = Path(path)
        p.write_text(html, encoding="utf-8")
        return p

    def plot(self, which="ladder", ax=None):
        """``which='ladder'`` or ``'nulls'``; needs the optional matplotlib.

        WHERE: 'ladder' draws the [L44] table, 'nulls' the [L42]
        per-section null columns; both delegate to _plot below.  The Agg
        backend is forced so plotting works on a headless machine.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "plot() needs matplotlib; it is an optional dependency. "
                "to_html() degrades to plain tables without it. (%s)" % e)
        return _plot(self, which, ax, plt)


# ------------------------------------------------------------------ helpers
def _ascii(s):
    """Force any string to ASCII (lossy '?'), for the console renderer only."""
    return str(s).encode("ascii", "replace").decode("ascii")


def _f(x, nd=4):
    """Format one number to ``nd`` decimals; None/NaN/inf print as 'n/a'.

    WHY: every renderer funnels numbers through here, so a missing value
    can never print as 'nan' in one format and '' in another.
    """
    if x is None:
        return "n/a"
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)  # non-numeric cell (e.g. a note string): pass through
    return "n/a" if not np.isfinite(x) else ("%." + str(nd) + "f") % x


def _ci(s: Optional[Stat]):
    """Render a Stat's 95% confidence interval [L40] as '[lo,hi]' or 'n/a'."""
    if s is None or s.ci is None:
        return "n/a"
    return "[%s,%s]" % (_f(s.ci[0]), _f(s.ci[1]))


def _fracstr(s: Optional[Stat]):
    """Render a win-fraction Stat as 'k/n (pct)'.

    WHERE: the fraction Stats (audit.py:1132-1141, 1251-1254) store the
    fraction in ``value`` and the section count in ``n_spots``, so
    value * n_spots recovers the numerator.
    """
    if s is None or not np.isfinite(s.value):
        return "n/a"
    return "%d/%d (%.0f%%)" % (round(s.value * s.n_spots), s.n_spots,
                               100 * s.value)


def _default(o):
    """json.dumps fallback: numpy scalars/arrays, TargetSpace and Path.

    WHY None for non-finite floats: JSON has no NaN/inf; null is the only
    faithful encoding a downstream parser will read back correctly.
    """
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        v = float(o)
        return v if np.isfinite(v) else None
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (TargetSpace,)):
        return o.value
    if isinstance(o, Path):
        return str(o)
    return str(o)


def _sec_json(sec):
    """Serialise one result dataclass for to_json.

    WHAT: walks the dataclass fields and converts, by type: a bare Stat
    via Stat.to_dict (core.py:184, flow [L40]); a (name, Stat) tuple such
    as Headline.strongest_null [L41]; a {key -> Stat} dict such as
    SelectionResult.by_n [L45]; a DataFrame via _df_json; anything else
    passes through untouched.
    WHY by type rather than by field name: a new field with one of these
    types is serialised correctly without touching this function.
    """
    if sec is None:
        return None
    out = {}
    for k, v in sec.__dict__.items():
        if isinstance(v, Stat):
            out[k] = v.to_dict()
        elif isinstance(v, tuple) and len(v) == 2 and isinstance(v[1], Stat):
            out[k] = {"name": v[0], **v[1].to_dict()}
        elif isinstance(v, dict) and v and all(isinstance(x, Stat) for x in v.values()):
            out[k] = {str(kk): vv.to_dict() for kk, vv in v.items()}
        elif isinstance(v, pd.DataFrame):
            out[k] = _df_json(v)
        else:
            out[k] = v
    return out


def _df_json(df):
    """DataFrame -> list of one dict per row (empty list when absent).

    WHY the to_json round-trip: pandas' own serialiser already maps NaN
    to null and numpy types to plain ones, so the result is JSON-clean.
    """
    if df is None or not len(df):
        return []
    return json.loads(df.to_json(orient="records"))


# --------------------------------------------------------------- rendering
# T: the bilingual string table for the file renderers.  Keys "0".."10" are
# the section headings; "banner"/"depthbanner"/"nomodel" are the three
# warning banners.  The console renderer does not use T (ASCII only).
T = {
    "en": {
        "0": "0. Run card and declared protocol",
        "1": "1. Headline",
        "2": "2. Depth confounding",
        "3": "3. Composition confounding",
        "4": "4. Spatial",
        "5": "5. Selection bias and protocol levers",
        "6": "6. Measurement ceiling",
        "7": "7. Attribution ladder",
        "8": "8. What you may and may not say",
        "9": "9. Degradation and N/A ledger",
        "10": "10. Reproducibility appendix",
        "banner": ("This report evaluates THIS SET OF PREDICTIONS. It says nothing "
                   "about the information content of H&E in general."),
        "depthbanner": ("TARGET_CARRIES_DEPTH: space=log1p:raw means the target "
                        "itself is monotone in sequencing depth. A large depth null "
                        "here is a definition, not a defect; DG is the comparable "
                        "number."),
        "nomodel": "NO_MODEL: no y_pred was supplied, so only the nulls are reported.",
    },
    "zh": {
        "0": "0. 运行卡与口径声明",
        "1": "1. 主指标",
        "2": "2. 深度混杂",
        "3": "3. 成分混杂",
        "4": "4. 空间",
        "5": "5. 选择偏倚与协议旋钮",
        "6": "6. 天花板",
        "7": "7. 归因阶梯",
        "8": "8. 可以说 / 不可以说",
        "9": "9. 降级与 N/A 记账",
        "10": "10. 复现附录",
        "banner": "本报告评的是**这组预测**,不是 H&E 的信息量。",
        "depthbanner": ("TARGET_CARRIES_DEPTH:space=log1p:raw 时目标本身随测序深度单调,"
                        "深度零模型高是口径的定义而非模型的问题,DG 才是可比数。"),
        "nomodel": "NO_MODEL:未提供 y_pred,本报告只出零模型部分。",
    },
}

# CSS: inlined into every HTML report so the file is fully self-contained
# (no network requests, opens identically from a mail attachment or an
# air-gapped machine).  The .tag.ub style renders the UPPER_BOUND badge of
# the oracle composition null.
CSS = """
:root{--fg:#16181d;--bg:#fff;--mut:#5b6270;--line:#dfe3ea;--warn:#8a4b00;
--warnbg:#fff6e5;--bad:#a10e25;--ok:#0a6b3d;--accent:#1b3a6b;}
*{box-sizing:border-box}
body{margin:0;padding:0 0 4rem;background:var(--bg);color:var(--fg);
font:15px/1.55 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:1080px;margin:0 auto;padding:1.5rem}
h1{font-size:1.5rem;margin:.2rem 0 .1rem}
h2{font-size:1.08rem;margin:2rem 0 .5rem;padding-bottom:.25rem;
border-bottom:2px solid var(--line);color:var(--accent)}
h3{font-size:.95rem;margin:1.1rem 0 .3rem;color:var(--mut)}
.sub{color:var(--mut);font-size:.86rem}
.banner{border-left:4px solid var(--accent);background:#f2f5fa;padding:.6rem .8rem;
margin:.8rem 0;font-size:.9rem}
.banner.warn{border-left-color:var(--warn);background:var(--warnbg);color:var(--warn)}
.big{font-size:1.9rem;font-weight:650;letter-spacing:-.02em}
.grid{display:flex;flex-wrap:wrap;gap:1rem;margin:.6rem 0}
.card{flex:1 1 240px;border:1px solid var(--line);border-radius:8px;padding:.7rem .85rem}
.card .lab{font-size:.76rem;text-transform:uppercase;letter-spacing:.06em;color:var(--mut)}
.tw{overflow-x:auto;-webkit-overflow-scrolling:touch}
table{border-collapse:collapse;width:100%;font-size:.85rem;margin:.3rem 0 .6rem}
th,td{border-bottom:1px solid var(--line);padding:.32rem .5rem;text-align:right;
white-space:nowrap}
th:first-child,td:first-child{text-align:left}
thead th{background:#f6f8fb;font-weight:600;color:var(--mut);position:sticky;top:0}
td.na{color:var(--mut)}
.tag{display:inline-block;font-size:.72rem;padding:.05rem .4rem;border-radius:4px;
border:1px solid var(--line);color:var(--mut)}
.tag.ub{border-color:#b58900;color:#8a6100;background:#fffbe8}
.tag.bad{border-color:var(--bad);color:var(--bad);background:#fdeef1}
.tag.ok{border-color:var(--ok);color:var(--ok);background:#eaf7f0}
ul{margin:.3rem 0 .6rem 1.1rem;padding:0}
li{margin:.2rem 0}
code,pre{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:.82rem}
pre{background:#f6f8fb;border:1px solid var(--line);border-radius:6px;padding:.7rem;
overflow-x:auto}
img.fig{max-width:100%;height:auto;border:1px solid var(--line);border-radius:6px}
.foot{margin-top:2.5rem;color:var(--mut);font-size:.8rem;border-top:1px solid var(--line);
padding-top:.7rem}
"""


def _esc(s):
    """Minimal HTML escaping; every user-influenced string passes through it."""
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def _stat_cells(s: Optional[Stat]):
    """One Stat [L40] -> the five fixed table cells.

    WHAT: value, floor, floor_kind, 95% CI, n spots/genes; a None Stat
    becomes five 'n/a' cells so table columns never shift.
    WHY the greyed class on a non-ok status: a degraded number stays
    visible but visibly second-class.
    WHERE: consumed only by _stat_table, which appends the status/note
    cell; the cells mirror Stat.__str__ (core.py:166), so HTML and console
    always show the same triple.
    """
    if s is None:
        return ['<td class="na">n/a</td>'] * 5
    cls = "" if s.status == "ok" else ' class="na"'
    return ["<td%s>%s</td>" % (cls, _f(s.value)),
            "<td%s>%s</td>" % (cls, _f(s.floor)),
            "<td%s>%s</td>" % (cls, _esc(s.floor_kind)),
            "<td%s>%s</td>" % (cls, _ci(s)),
            "<td%s>%d/%d</td>" % (cls, s.n_spots, s.n_genes)]


def _stat_table(rows, lang="en"):
    """Render (name, Stat, tag) rows.  The floor column cannot be switched off.

    This is the [L40] contract on the page: there is no code path that
    renders a Stat's value without its null_floor column beside it.  The
    ``tag`` slot exists for the UPPER_BOUND badge on the oracle null.
    """
    hdr = (["metric", "value", "null_floor", "floor_kind", "95% CI", "n spot/gene",
            "status / note"])
    h = "".join("<th>%s</th>" % _esc(x) for x in hdr)
    body = []
    for name, s, tag in rows:
        cells = _stat_cells(s)
        note = "" if s is None else _esc(s.note)
        st = "n/a" if s is None else _esc(s.status)
        tg = ' <span class="tag ub">%s</span>' % tag if tag else ""
        body.append("<tr><td>%s%s</td>%s<td class='na'>%s %s</td></tr>"
                    % (_esc(name), tg, "".join(cells), st, note))
    return ("<div class='tw'><table><thead><tr>%s</tr></thead><tbody>%s</tbody>"
            "</table></div>" % (h, "".join(body)))


def _df_table(df, maxrows=200):
    """Any DataFrame -> a plain HTML table, capped at ``maxrows`` rows.

    WHY the cap: per_gene [L43] can run to thousands of rows; the HTML
    stays readable and points to --csv-dir for the full table.  Floats go
    through _f so 'n/a' formatting matches the Stat tables.
    """
    if df is None or not len(df):
        return "<p class='sub'>(empty)</p>"
    d = df.head(maxrows)
    h = "".join("<th>%s</th>" % _esc(c) for c in d.columns)
    rows = []
    for _, r in d.iterrows():
        tds = []
        for c in d.columns:
            v = r[c]
            if isinstance(v, float):
                tds.append("<td>%s</td>" % _f(v))
            else:
                tds.append("<td>%s</td>" % _esc(v))
        rows.append("<tr>%s</tr>" % "".join(tds))
    extra = ("<p class='sub'>showing %d of %d rows; full table in --csv-dir</p>"
             % (len(d), len(df))) if len(df) > maxrows else ""
    return ("<div class='tw'><table><thead><tr>%s</tr></thead><tbody>%s</tbody>"
            "</table></div>%s" % (h, "".join(rows), extra))


def _plot(rep, which, ax, plt):
    """Shared drawing code for plot() and _figs; plt is injected by the caller.

    WHY inject plt: matplotlib is optional, so this module never imports
    it at the top level; both callers do the guarded import themselves.
    """
    if which == "ladder":
        # --- the attribution ladder figure -----------------------------
        # WHAT: one faint line per section over the rung columns of the
        # ladder table [L44], the across-section median in red on top,
        # and the median DRG permutation floor as a dashed reference.
        # WHY only columns with data: a nulls-only run has no rungs and a
        # run without labels has no crg/drg; the x axis shrinks to match.
        lad = rep.ladder
        cols = [c for c in ("r_full", "dg", "crg", "drg") if c in lad.columns
                and lad[c].notna().any()]
        if ax is None:
            fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=130)
        for _, row in lad.iterrows():
            ys = [row[c] for c in cols]   # this section's rung values
            ax.plot(range(len(cols)), ys, "-o", lw=1, ms=3, alpha=.55,
                    color="#1b3a6b")
        med = [float(np.nanmedian(lad[c].astype(float))) for c in cols]
        ax.plot(range(len(cols)), med, "-o", lw=2.4, ms=6, color="#a10e25",
                label="across-section median")
        if "drg_floor" in lad.columns and lad["drg_floor"].notna().any():
            ax.axhline(float(np.nanmedian(lad["drg_floor"].astype(float))),
                       ls="--", lw=1, color="#5b6270",
                       label="median permutation floor (DRG)")
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels([c.upper() for c in cols])
        ax.set_ylabel("per-gene Pearson r (section median)")
        ax.axhline(0, lw=.8, color="#999")
        ax.legend(fontsize=7, frameon=False)
        ax.set_title("Attribution ladder, one line per section", fontsize=9)
        return ax
    if which == "nulls":
        # --- model vs nulls, per section -------------------------------
        # WHAT: one line per readout over the sections, from the
        # per_section table [L42]: the model beside each zero-pixel null,
        # so a section where a null wins is visible at a glance.
        if ax is None:
            fig, ax = plt.subplots(figsize=(6.4, 3.2), dpi=130)
        ps = rep.per_section
        cols = [c for c in ("r_model", "depth_null", "cross_null", "r_nbr")
                if c in ps.columns and ps[c].notna().any()]
        x = np.arange(len(ps))
        for i, c in enumerate(cols):
            ax.plot(x, ps[c].astype(float), "-o", ms=3, lw=1.2, label=c)
        ax.set_xticks(x)
        ax.set_xticklabels(ps["section"].astype(str), rotation=90, fontsize=6)
        ax.set_ylabel("section median r")
        ax.legend(fontsize=7, frameon=False)
        ax.set_title("Model vs zero-pixel nulls, per section", fontsize=9)
        return ax
    raise ValueError("unknown plot: %r (try 'ladder' or 'nulls')" % which)


def _figs(rep):
    """Return list of (caption, base64 png), empty if matplotlib is absent.

    WHAT: renders the two standard figures to in-memory PNGs and base64-
    encodes them for data: URIs, keeping the HTML single-file.
    WHY the blanket except around each figure: a figure is decoration;
    a plotting failure must degrade to tables, never kill the report.
    WHERE: consumed only by _render_html, which places 'nulls' in section
    1 and 'ladder' in section 7.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []
    out = []
    for which in ("nulls", "ladder"):
        try:
            fig, ax = plt.subplots(figsize=(6.6, 3.4), dpi=130)
            _plot(rep, which, ax, plt)
            fig.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            plt.close(fig)
            out.append((which, base64.b64encode(buf.getvalue()).decode("ascii")))
        except Exception:
            continue
    return out


def _render_html(rep: AuditReport, lang="en") -> str:
    """One self-contained HTML file: inline CSS, base64 figures, no requests.

    Section order matches summary(): 0 run card, 1 headline, 2 depth,
    3 composition, 4 spatial, 5 selection, 6 ceiling, 7 ladder, 8 claims,
    9 ledger, 10 reproducibility.  Every Stat cell comes through
    _stat_table, so the floor column [L40] is present in each section.
    """
    t = T.get(lang, T["en"])  # bilingual heading/banner table defined above
    P = []
    A = P.append  # A(x) appends one HTML fragment; joined once at the end
    A("<title>stnull audit</title><style>%s</style>" % CSS)
    A("<div class='wrap'>")
    A("<h1>stnull audit report</h1>")
    A("<p class='sub'>stnull %s &middot; %s &middot; space=<code>%s</code> &middot; "
      "seed=%d</p>" % (VERSION, rep.run.created_utc, _esc(rep.run.space.value),
                       rep.run.seed))
    # WHAT: the scope banner always shows; the two warning banners fire
    # only in a nulls-only run and when the declared space [L12] itself
    # rises with sequencing depth.
    A("<div class='banner'>%s</div>" % _esc(t["banner"]))
    if rep.headline is None:
        A("<div class='banner warn'>%s</div>" % _esc(t["nomodel"]))
    if rep.depth is not None and rep.depth.target_carries_depth:
        A("<div class='banner warn'>%s</div>" % _esc(t["depthbanner"]))

    # 0 run card
    # WHAT: sizes as cards, then the input-presence table with the sha1
    # fingerprints [L51], then the protocol line.
    # WHY: a reader must be able to tell WHICH data and protocol produced
    # the numbers before seeing any of them.
    A("<h2>%s</h2>" % _esc(t["0"]))
    A("<div class='grid'>")
    for lab, val in (("spots", rep.run.n_spots), ("genes", rep.run.n_genes),
                     ("sections", rep.run.n_sections),
                     ("patients", rep.run.n_patients if rep.run.n_patients else "n/a")):
        A("<div class='card'><div class='lab'>%s</div><div class='big'>%s</div></div>"
          % (lab, val))
    A("</div>")
    pres = pd.DataFrame([{"input": k, "present": v,
                          "sha1": rep.run.inputs_sha1.get(k, "")}
                         for k, v in rep.run.inputs_present.items()])
    A(_df_table(pres))
    A("<p class='sub'>null_fit=<code>%s</code>, depth_proxy=<code>%s</code>, "
      "perm=<code>%s</code> x%d, bootstrap x%d, aggregation=%s over genes then %s "
      "over sections.</p>"
      % (_esc(rep.run.null_fit), _esc(rep.run.depth_proxy), _esc(rep.run.perm_kind),
         rep.run.n_perm, rep.run.n_boot, rep.run.agg_gene, rep.run.agg_section))
    if rep.run.space_note:
        A("<p class='sub'>space_note: %s</p>" % _esc(rep.run.space_note))

    # 1 headline
    # WHERE: r_median/r_mean off the [L25] chain, strongest_null [L41],
    # and the win fraction; the 'nulls' figure is placed right under the
    # table so the per-section picture sits beside the aggregate.
    A("<h2>%s</h2>" % _esc(t["1"]))
    h = rep.headline
    if h is None:
        A("<p class='sub'>NO_MODEL.</p>")
    else:
        rows = [("model, median over genes", h.r_median, None),
                ("model, mean over genes", h.r_mean, None)]
        if h.strongest_null:
            rows.append(("strongest zero-pixel null: %s" % h.strongest_null[0],
                         h.strongest_null[1], None))
        rows.append(("fraction of sections where model beats that null",
                     h.frac_sections_model_beats_null, None))
        A(_stat_table(rows, lang))
        A("<p class='sub'>%s</p>" % _esc(h.aggregation_note))
    figs = _figs(rep)
    for name, b64 in figs:
        if name == "nulls":
            A("<img class='fig' alt='model vs nulls per section' "
              "src='data:image/png;base64,%s'>" % b64)
    if not figs:
        A("<p class='sub'>(matplotlib not installed: figures omitted, all numbers "
          "are in the tables)</p>")

    # 2 depth
    # WHAT: the six depth readouts of DepthResult as one Stat table.
    # WHY the loud N/A banner: a missing lib_size [L04] means the field's
    # main confounder went unaudited, which the reader must not miss.
    A("<h2>%s</h2>" % _esc(t["2"]))
    d = rep.depth
    if d is None:
        A("<div class='banner warn'>N/A &mdash; %s. The largest known confounder "
          "in this field was not audited.</div>"
          % _esc(rep.na.get("depth", "lib_size not provided")))
    else:
        A(_stat_table([("depth_null (zero pixels, log lib only)", d.depth_null, None),
                       ("depth_1d (image -> 1 number -> genes)", d.depth_1d, None),
                       ("DG (r after controlling depth)", d.dg, None),
                       ("share of r attributable to depth", d.share_depth, None),
                       ("corr(l_hat, log lib) -- depth carried by the model",
                        d.depth_r_of_pred, None),
                       ("corr(sd(log lib), r) across sections", d.trend_sd_vs_r, None)],
                      lang))
        A("<p class='sub'>%s</p>" % _esc(d.lib_denominator_check))
        A("<h3>per-section spread of log library size</h3>")
        A(_df_table(d.sd_loglib_by_section))

    # 3 composition
    # WHAT: cross-fitted null, oracle null (tagged UPPER_BOUND) and CRG.
    # WHY no clustering fallback: inventing labels from the expression
    # would let the model's own signal leak into its null (see banner).
    A("<h2>%s</h2>" % _esc(t["3"]))
    c = rep.composition
    if c is None:
        A("<div class='banner warn'>N/A &mdash; %s. stnull will NOT invent labels "
          "by clustering: that would leak the model's own information into the "
          "null.</div>" % _esc(rep.na.get("composition", "labels not provided")))
    else:
        A(_stat_table([("cross null (class means, cross-fitted)", c.cross_null, None),
                       ("oracle null (class means read off this section)",
                        c.oracle_null, "UPPER_BOUND"),
                       ("CRG (r after controlling composition)", c.crg, None)], lang))
        A("<p class='sub'>%d classes (%s): %s</p>"
          % (c.n_classes, _esc(c.kind),
             _esc(", ".join("%s=%d" % (k, v) for k, v in sorted(c.class_counts.items())))))
        if c.small_classes_dropped:
            A("<p class='sub'>merged into 'other' (too few spots): %s</p>"
              % _esc(", ".join(c.small_classes_dropped)))

    # 4 spatial
    # WHAT: neighbour null [L30], smoothing lever [L29], residual Moran's
    # I, and the block-vs-free floor pair [L37] when it exists.
    # WHY the N/A banner mentions floors: without coords [L06] every
    # permutation floor in the report fell back to free permutation.
    A("<h2>%s</h2>" % _esc(t["4"]))
    s = rep.spatial
    if s is None:
        A("<div class='banner warn'>N/A &mdash; %s. Permutation floors fall back to "
          "free permutation, which is ANTICONSERVATIVE on spatially autocorrelated "
          "data: the floors below are too low and the headline therefore looks too "
          "good.</div>" % _esc(rep.na.get("spatial", "coords not provided")))
    else:
        rows = [("r_nbr (mean of neighbours' TRUE y, zero pixels)", s.r_nbr, None),
                ("fraction of sections where r_nbr beats the model",
                 s.frac_sections_nbr_beats_model, None),
                ("lever: 3x3 smoothing of the ground truth (delta)",
                 s.smooth_lever, None),
                ("Moran's I of the model residual", s.moran_I_resid, None)]
        if s.perm_block_vs_free:
            rows.append(("floor under block permutation", s.perm_block_vs_free[0], None))
            rows.append(("floor under free permutation", s.perm_block_vs_free[1], None))
        A(_stat_table(rows, lang))

    # 5 selection
    # WHERE: by_n [L45], the optional train-selected rows, the leakage
    # lever [L46], and the lever price list (audit.py:1329-1353).
    A("<h2>%s</h2>" % _esc(t["5"]))
    sel = rep.selection
    if sel is None:
        A("<p class='sub'>NO_MODEL.</p>")
    else:
        rows = [("test-selected top-%d" % n, st, None)
                for n, st in sorted(sel.by_n.items())]
        if sel.train_selected:
            rows += [("train-selected top-%d (legal readout)" % n, st, None)
                     for n, st in sorted(sel.train_selected.items())]
        if sel.leakage_lever is not None:
            rows.append(("leakage lever, measured on a zero-pixel null",
                         sel.leakage_lever, None))
        A(_stat_table(rows, lang))
        A("<h3>protocol lever price list</h3>")
        A(_df_table(sel.lever_price))

    # 6 ceiling
    # WHERE: the r_tech chain [L47]; without raw counts [L08] the section
    # states its N/A reason and substitutes nothing.
    A("<h2>%s</h2>" % _esc(t["6"]))
    ce = rep.ceiling
    if ce is None or ce.method == "na":
        A("<div class='banner warn'>r_tech N/A &mdash; %s. No approximation is "
          "substituted.</div>" % _esc(rep.na.get("ceiling", "counts not provided")))
        if ce is not None and ce.r_nbr is not None:
            A(_stat_table([("r_nbr (spatial smoothing reference)", ce.r_nbr, None)],
                          lang))
    else:
        A(_stat_table([("r_tech (%s)" % ce.method, ce.r_tech, None),
                       ("r_nbr (spatial smoothing reference)", ce.r_nbr, None),
                       ("r / r_tech, relative to this section's measurement "
                        "noise ceiling", ce.r_over_ceiling, None)], lang))

    # 7 ladder
    # WHERE: the full [L44] table plus the ladder figure from _figs; the
    # paper's ladder plot is this figure.
    A("<h2>%s</h2>" % _esc(t["7"]))
    A("<p class='sub'>r_full &rarr; DG (control depth) &rarr; CRG (control "
      "composition) &rarr; DRG (control both). Each cell carries the permutation "
      "floor of its own rung and a spot-bootstrap 95% CI.</p>")
    A(_df_table(rep.ladder))
    for name, b64 in figs:
        if name == "ladder":
            A("<img class='fig' alt='attribution ladder' "
              "src='data:image/png;base64,%s'>" % b64)

    # 8 claims
    # WHAT: claims() output grouped by verdict; the out-of-scope group
    # prints last and always, with each claim's why-string beside it.
    A("<h2>%s</h2>" % _esc(t["8"]))
    for grp, title in (("supported", "Supported by this run"),
                       ("not_supported", "NOT supported by this run"),
                       ("out_of_scope", "Out of scope -- do not write these")):
        cl = [c for c in rep.claims() if c.support == grp]
        if not cl:
            continue
        A("<h3>%s</h3><ul>" % _esc(title))
        for c in cl:
            txt = c.text_zh if lang == "zh" else c.text_en
            why = c.evidence.get("why", "") if isinstance(c.evidence, dict) else ""
            A("<li>%s%s</li>" % (_esc(txt),
                                 (" <span class='sub'>%s</span>" % _esc(why)) if why else ""))
        A("</ul>")

    # 9 ledger
    # WHERE: caveats() merges the [L48]/[L50]/[L49] ledgers; an empty
    # ledger is stated explicitly rather than omitted.
    A("<h2>%s</h2>" % _esc(t["9"]))
    cav = rep.caveats()
    if not cav:
        A("<p class='sub'>None: every optional input was supplied.</p>")
    else:
        A("<ul>%s</ul>" % "".join("<li>%s</li>" % _esc(x) for x in cav))

    # 10 repro
    # WHAT: the ready-to-paste Methods paragraph, populated with this
    # run's actual parameters (cite_text below).
    A("<h2>%s</h2>" % _esc(t["10"]))
    A("<pre>%s</pre>" % _esc(cite_text(rep)))
    A("<div class='foot'>Generated by stnull %s. Every value in this document is "
      "printed next to the permutation floor of its own readout rule; there is no "
      "option to hide that column.</div>" % VERSION)
    A("</div>")
    return "\n".join(P)


def _render_md(rep: AuditReport, lang="en") -> str:
    """Markdown renderer: the console summary in a code fence, plus the
    ladder [L44] and per_section [L42] tables, the claims and the Methods
    paragraph.

    WHY embed summary() verbatim: one source of truth for the numbers; the
    Markdown adds tables around it instead of re-deriving anything.
    """
    t = T.get(lang, T["en"])
    L = ["# stnull audit report", "",
         "stnull %s | %s | space=`%s` | seed=%d" % (VERSION, rep.run.created_utc,
                                                    rep.run.space.value, rep.run.seed),
         "", "> %s" % t["banner"], ""]
    L.append("```")
    L.append(rep.summary(lang="en"))
    L.append("```")
    L.append("")
    L.append("## %s" % t["7"])
    L.append("")
    L.append(_md_table(rep.ladder))
    L.append("")
    L.append("## %s" % t["1"])
    L.append("")
    L.append(_md_table(rep.per_section))
    L.append("")
    L.append("## %s" % t["8"])
    L.append("")
    for c in rep.claims():
        mark = {"supported": "[OK]", "not_supported": "[NO]",
                "out_of_scope": "[OUT OF SCOPE]"}[c.support]
        L.append("- %s %s" % (mark, c.text_zh if lang == "zh" else c.text_en))
    L.append("")
    L.append("## %s" % t["10"])
    L.append("")
    L.append("```")
    L.append(cite_text(rep))
    L.append("```")
    return "\n".join(L)


def _md_table(df, maxrows=100):
    """DataFrame -> a GitHub-flavoured Markdown table, capped at maxrows."""
    if df is None or not len(df):
        return "_(empty)_"
    d = df.head(maxrows)
    L = ["| " + " | ".join(str(c) for c in d.columns) + " |",
         "|" + "|".join("---" for _ in d.columns) + "|"]
    for _, r in d.iterrows():
        L.append("| " + " | ".join(
            _f(r[c]) if isinstance(r[c], float) else str(r[c]) for c in d.columns)
            + " |")
    return "\n".join(L)


def cite_text(rep: Optional[AuditReport] = None) -> str:
    """A Methods paragraph populated with the parameters of an actual run.

    Called with no report it prints the package defaults, which is what
    ``stnull cite`` does.  The last sentence (DRG is a property of the
    audited predictions) is not optional: it is the reading the rest of the
    package exists to protect.
    """
    # WHAT: pull the four protocol values off RunMeta when a report is
    # given; otherwise fall back to the package defaults.
    # WHERE: rep.run was built at audit.py:1430-1445; consumers are the
    # `stnull cite` subcommand (cli.py:main) and section 10 of the HTML
    # and Markdown renderers.
    run = rep.run if rep is not None else None
    space = run.space.value if run else "<declare your target space>"  # [L12]
    nperm = run.n_perm if run else 200      # permutation draws behind floors
    nboot = run.n_boot if run else 500      # bootstrap draws behind the CIs
    perm = run.perm_kind if run else "block"
    return (
        "METHODS PARAGRAPH (paste and edit)\n"
        "Predictions were audited with stnull %s. Expression targets were in the\n"
        "%s space; all correlations are per-gene Pearson r computed within a\n"
        "section and aggregated first over genes (median and mean are both\n"
        "reported) and then over sections. Every reported value is accompanied by\n"
        "the null floor of the same readout rule, obtained by %s permutation of\n"
        "the spot order within each section (%d draws, floor = 95th percentile of\n"
        "the null distribution), and by a %d-draw bootstrap 95%% confidence\n"
        "interval. Three zero-pixel null models were fitted alongside the model: a\n"
        "depth null (per-gene least squares on log library size only), a\n"
        "composition null (class-mean lookup, cross-fitted within section, with an\n"
        "oracle variant that reads class means off the scored section and is\n"
        "reported as an upper bound), and a neighbour null (the mean of the true\n"
        "expression of a spot's lattice neighbours). The attribution ladder\n"
        "residualises both y and y_hat on X = [1], [1, l_hat], [1, pi] and\n"
        "[1, l_hat, pi] via an SVD orthonormal basis (rank-deficiency safe),\n"
        "giving r_full, DG, CRG and DRG respectively; l_hat is the sequencing\n"
        "depth carried by the model's own predictions, obtained by cross-fitted\n"
        "kernel ridge regression of log library size on y_hat within each section.\n"
        "DRG is a property of the audited predictions and is not an upper bound on\n"
        "the information content of H&E.\n"
        % (VERSION, space, perm, nperm, nboot))
