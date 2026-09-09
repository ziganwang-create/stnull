# -*- coding: utf-8 -*-
"""stnull.audit, the orchestrator: :func:`audit` and its entry points.

Author: Zigan Wang.

WHAT THIS MODULE IS
    The only file that knows about sections, options and report assembly.  It
    validates the inputs, loops over sections computing every readout with the
    permutation floor of that readout's own rule, aggregates the sections, and
    packs the result into the containers in ``stnull.report``.

WHO CALLS IT
    Users (``stnull.audit``), the CLI (``stnull.cli``), and the convenience
    wrappers at the bottom of this file (``nulls_only``, ``ladder``,
    ``perm_floor``, ``compare``).

THE PIPELINE, IN ORDER
    1. ``check_inputs``, structural validation, no correlations computed.
    2. option validation, closed sets for null_fit / depth_proxy / perm_kind,
       so a typo can never mislabel the protocol in an archived report.
    3. per-section ``_section_block``, nulls, ladder rungs, levers, floors.
    4. ``_combine``, across-section aggregation with a section bootstrap.
    5. report assembly + the degradation ledger.

KEY ASSUMPTIONS
    * All correlations are computed WITHIN a section.  Pooling spots across
      sections would let between-section depth and composition differences
      inflate r, which is the confounder under audit.
    * The user's data is never transformed (decision D2) and no model is ever
      trained; every null here sees zero pixels.
    * Point estimates must not move when the user changes ``n_perm`` or
      ``n_boot``, so each section draws three independent RNG streams.

DATA LINEAGE (numbers cite docs/CODE_WALKTHROUGH.md (section 4, the variable lineage table); glossary in CODE_WALKTHROUGH.md)
    Inputs.  User arrays enter through the coercion layer at the top of
    :func:`audit`: ``y_true``/``y_pred`` become dense float matrices ``Y``/``P``
    [L01]/[L02]; ``section`` becomes ``sec`` [L03]; ``lib_size`` becomes ``lib``
    [L04] and its natural log ``loglib`` [L05], born once and reused by every
    depth control in the package; ``coords`` become ``XY`` [L06]; ``labels`` or
    ``composition`` become the class matrix ``Comp`` [L07] via :func:`_one_hot`;
    ``counts`` [L08], ``patient`` [L09], ``is_train`` [L10], ``lib_pred``
    [L11], ``space`` [L12], ``genes`` [L13] and ``lib_size_full`` [L14] follow
    the same route.  The CLI (``cli.py``) and the example script only rename
    files/columns onto these keywords [L15]-[L17]; they compute nothing.
    Per section.  ``audit`` slices each input by the section mask and hands the
    slices to :func:`_section_block`, which computes every zero-pixel null (in
    ``nulls.py``), every attribution-ladder rung (in ``drg.py``), every
    protocol lever (in ``levers.py``) and every permutation floor for that one
    section, all under one shared permutation plan [L19] and one shared
    cross-fitting partition [L20].
    Outputs.  The per-section dicts are collected into ``blocks`` [L38],
    funnelled through :func:`_combine` [L39] into ``Stat`` triples
    (value, floor, ci) [L40] built by ``core.make_stat``, and packed into the
    containers of ``report.py``, which renders console, JSON, CSV and HTML.
"""
from __future__ import annotations

import datetime
import hashlib
import sys
import warnings as _warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp

from . import ceiling as _ceil
from . import drg as _drg
from . import levers as _lev
from . import nulls as _nulls
from .core import (BadInput, InsufficientData, LibSizeLooksLikePanelSum, Stat,
                   agg, boot_ci, colcorr, kfold_indices, make_stat, na_stat,
                   ortho_basis, perm_plan, resid, sha1_of)
from .report import (VERSION, AuditReport, CeilingResult, CompositionResult,
                     DepthResult, Headline, RunMeta, SelectionResult,
                     SpatialResult, cite_text)
from .spaces import DEPTH_CARRYING, TargetSpace, coerce_space

__all__ = ["audit", "nulls_only", "ladder", "perm_floor", "compare",
           "check_inputs", "InputReport"]

#: Closed option sets.  They are validated at the top of audit() so that a
#: typo raises instead of silently selecting a different estimator while the
#: run card prints the typo as the protocol that was used.
_NULL_FITS = ("cv_within_section", "train_only", "oracle")
_DEPTH_PROXIES = ("observed", "given", "from_pred", "none")
_PERM_KINDS = ("auto", "block", "free")


# ------------------------------------------------------------ input checking
@dataclass
class InputReport:
    """What :func:`check_inputs` found, before any statistic is computed.

    ``ok`` is False when at least one ERROR was raised; ``audit(strict=True)``
    refuses to run in that case, and ``strict=False`` turns the errors into
    warnings on the report.  ``small_sections`` lists sections below
    ``min_spots``.
    """

    ok: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    shapes: Dict[str, str] = field(default_factory=dict)
    present: Dict[str, bool] = field(default_factory=dict)
    n_sections: int = 0
    small_sections: List[str] = field(default_factory=list)

    def __str__(self):
        L = ["InputReport: %s" % ("OK" if self.ok else "FAILED")]
        L.append("  shapes: " + ", ".join("%s=%s" % (k, v)
                                          for k, v in self.shapes.items()))
        L.append("  present: " + ", ".join(
            "%s=%s" % (k, "yes" if v else "NO") for k, v in self.present.items()))
        for e in self.errors:
            L.append("  ERROR   %s" % e)
        for w in self.warnings:
            L.append("  warning %s" % w)
        return "\n".join(L)

    __repr__ = __str__


def _mat(x, name):
    """Coerce one user matrix into a float64 2-D array (sparse passes through).

    WHAT: the single entry gate for matrix inputs.  A DataFrame is stripped to
    its values, a scipy sparse matrix is returned as-is for now (audit()
    densifies after validation, so check_inputs can stay cheap on sparse
    data), and anything else must already be 2-D (n_spots, n_genes).  A spot
    is one measured tissue location on the slide; a gene column holds that
    spot's expression readout for one gene.
    WHY: every routine downstream assumes a plain (n_spots, n_genes) float
    array; failing here produces an error that names the offending argument
    instead of a numpy shape error several call frames deep.
    WHERE: called on ``y_true`` and ``y_pred`` from check_inputs() and again
    from the top of audit(), where the results become ``Y`` [L01] and ``P``
    [L02]; those are sliced per section and become the ``y``/``yh`` arguments
    of _section_block().
    """
    if x is None:
        return None
    if isinstance(x, pd.DataFrame):
        return np.asarray(x.values, dtype=np.float64)
    if sp.issparse(x):
        return x
    a = np.asarray(x)
    if a.ndim != 2:
        raise BadInput("%s must be 2-D (n_spot, n_gene), got shape %s"
                       % (name, a.shape))
    return a.astype(np.float64)


def _vec(x, name, n=None):
    """Coerce one per-spot vector to a flat 1-D array, length-checked against n.

    WHAT: flattens Series/Index/array-likes to shape (n_spots,) and, when
    ``n`` is given, refuses a length that differs from the row count of
    ``y_true``.
    WHY: per-spot metadata (section labels, library sizes, patient ids) must
    align row-for-row with the expression matrix; a silent misalignment would
    scramble every within-section correlation without raising anywhere.
    WHERE: called on ``section`` [L03], ``lib_size`` [L04], ``patient``
    [L09], ``lib_pred`` [L11] and ``is_train`` [L10] from both check_inputs()
    and the coercion block of audit().
    """
    if x is None:
        return None
    if isinstance(x, (pd.Series, pd.Index)):
        x = x.values
    a = np.asarray(x).ravel()
    if n is not None and a.size != n:
        raise BadInput("%s has length %d but y_true has %d rows"
                       % (name, a.size, n))
    return a


def _nonfinite_count(a):
    """Number of non-finite entries, sparse-safe (sparse zeros are finite)."""
    if a is None:
        return 0
    if sp.issparse(a):
        return int((~np.isfinite(a.data)).sum())
    arr = np.asarray(a, dtype=np.float64)
    return int((~np.isfinite(arr)).sum())


def check_inputs(y_true=None, y_pred=None, section=None, lib_size=None,
                 coords=None, labels=None, composition=None, counts=None,
                 patient=None, is_train=None, lib_pred=None, genes=None,
                 lib_size_full=None, min_spots=50, min_genes=20, space=None,
                 **_ignored):
    """Cheap structural validation; never computes a correlation.

    Parameters
    ----------
    Same names and units as :func:`audit`; every argument is optional except
    ``y_true`` and ``section``.
    min_spots, min_genes : int
        Thresholds below which a section / panel is judged underpowered.
    space : str or None
        Only checked for presence here; ``audit`` is where it is coerced.

    Returns
    -------
    InputReport with ``ok``, ``errors``, ``warnings``, shapes and presence
    flags.  ``audit(strict=True)`` refuses to run when ``ok`` is False.

    What counts as an ERROR (not a warning)
        Anything that would otherwise surface as an exception from deep inside
        numpy or scipy naming neither the input nor the section: shape
        mismatches, and non-finite values in ``lib_size``, ``lib_size_full``,
        ``coords``, ``composition`` or ``lib_pred``.  Those feed
        ``np.linalg.lstsq``, ``cKDTree`` and ``rng.binomial``, none of which
        degrade gracefully.  Non-finite values in ``y_true`` are only a
        warning, because they degrade per gene by design.
    """
    # WHAT: start an empty report [L52], then fill errors/warnings as each
    # input is inspected.  WHY: collecting everything into one object lets
    # `stnull check` (cli.py) print the full list in one pass instead of
    # failing on the first problem.  WHERE: the returned InputReport gates
    # audit() under strict=True and otherwise feeds its `warns` ledger [L49].
    rep = InputReport(ok=True)

    # ---- y_true / y_pred: shapes and finiteness ------------------------ [L01][L02]
    # y_true is the anchor: every other input is validated against its row
    # count n (spots) and column count G (genes).
    Y = _mat(y_true, "y_true")
    if Y is None:
        rep.ok = False
        rep.errors.append("y_true is required")
        return rep
    n, G = Y.shape  # n = spots over the whole run (n_all), G = genes in the panel
    rep.shapes["y_true"] = "%dx%d" % (n, G)
    # y_true / y_pred may still be sparse here (audit densifies later), so the
    # finiteness tests below go through _nonfinite_count rather than
    # np.isfinite, which raises a TypeError on a sparse matrix.
    P = _mat(y_pred, "y_pred")
    if P is not None:
        rep.shapes["y_pred"] = "%dx%d" % P.shape
        if P.shape != Y.shape:
            rep.ok = False
            rep.errors.append("y_pred shape %s != y_true shape %s"
                              % (P.shape, Y.shape))
        elif _nonfinite_count(P):
            rep.ok = False
            rep.errors.append("y_pred contains %d non-finite values"
                              % _nonfinite_count(P))
    if _nonfinite_count(Y):
        rep.warnings.append(
            "y_true contains %d non-finite values; affected genes will score NaN"
            % _nonfinite_count(Y))
    # ---- section labels: required, and sections must be big enough ----- [L03]
    # WHAT: count spots per section (a section = one tissue slice) and flag
    # any below min_spots.  WHY: every r in this package is computed within a
    # section; a tiny section gives a correlation too noisy to floor, so it is
    # an error rather than a quiet NaN later.  WHERE: the same labels drive
    # the per-section loop in audit() and the ceiling loop.
    sec = _vec(section, "section", n)
    if sec is None:
        rep.ok = False
        rep.errors.append("section is required (all r are computed within a section)")
        return rep
    uniq, cnts = np.unique(sec.astype(str), return_counts=True)
    rep.n_sections = len(uniq)
    rep.shapes["sections"] = "%d" % len(uniq)
    rep.small_sections = [str(u) for u, c in zip(uniq, cnts) if c < min_spots]
    if rep.small_sections:
        rep.ok = False
        rep.errors.append("sections with < min_spots=%d spots: %s"
                          % (min_spots, ", ".join(rep.small_sections)))
    if G < min_genes:
        rep.ok = False
        rep.errors.append("only %d genes (< min_genes=%d)" % (G, min_genes))
    # ---- lib_size: finiteness plus the panel-sum trap ------------------ [L04]
    # WHAT: validate the per-spot library size (total UMI count of a spot; a
    # UMI is a deduplicated tag for one captured RNA molecule, so library
    # size = sequencing depth).  WHY: depth is the biggest known confounder
    # this package controls for, so a broken lib_size silently breaks the
    # audit's main null.  WHERE: audit() turns this vector into loglib [L05],
    # which feeds the depth null, the joint null, the observed-DRG design and
    # the leakage design.
    lib = _vec(lib_size, "lib_size", n)
    if lib is not None:
        # NaN is invisible to np.nanmin, and np.maximum(nan, 1.0) is nan, so a
        # single missing library size used to reach np.linalg.lstsq and come
        # back as LinAlgError('SVD did not converge').  Refuse it here.
        _nf = _nonfinite_count(lib)
        if _nf:
            rep.ok = False
            rep.errors.append("lib_size contains %d non-finite value(s); the "
                              "depth null cannot be fitted" % _nf)
        elif np.nanmin(lib) <= 0:
            rep.warnings.append("lib_size has non-positive entries; clamped to 1")
        # The panel-sum trap: if the user passed the row sum of the evaluated
        # gene panel instead of the full-transcriptome depth, undoing a
        # log1p-CP10K transform of y_true reconstructs that row sum almost
        # exactly (r > 0.999).  A panel sum is partly made of the very genes
        # being predicted, so "controlling for depth" with it would leak the
        # target into the control.  Raised as LibSizeLooksLikePanelSum.
        try:
            rs = np.expm1(np.clip(np.asarray(
                Y.todense() if sp.issparse(Y) else Y, dtype=np.float64),
                0, 50)).sum(1)
            good = np.isfinite(rs) & np.isfinite(lib)
            if good.sum() > 10 and np.std(rs[good]) > 0 and np.std(lib[good]) > 0:
                cc = float(np.corrcoef(rs[good], lib[good])[0, 1])
                if cc > 0.999:
                    rep.ok = False
                    rep.errors.append(
                        "lib_size correlates %.5f with the row sum of y_true: it "
                        "looks like a PANEL row sum, not a full-transcriptome "
                        "library size (LibSizeLooksLikePanelSum)" % cc)
        except Exception:
            pass
    # ---- counts: must be raw integers -------------------------------- [L08]
    # WHAT: spot-check the first 50 rows for integerness.  WHY: the
    # measurement ceiling (ceiling.r_tech) thins counts binomially, which is
    # only meaningful on raw UMI counts; normalised values would give a
    # ceiling that prices the normalisation, not the measurement noise.
    # WHERE: counts is consumed only by ceiling.r_tech, row-sliced per
    # section in the assembly half of audit().
    if counts is not None:
        C = counts
        sub = (np.asarray(C[:50].todense()) if sp.issparse(C)
               else np.asarray(C)[:50])
        if not np.allclose(sub, np.rint(sub)):
            rep.ok = False
            rep.errors.append("counts must be raw integer counts")
        if (sp.issparse(C) and C.shape != Y.shape) or (
                not sp.issparse(C) and np.asarray(C).shape != Y.shape):
            rep.ok = False
            rep.errors.append("counts shape != y_true shape")
    # Row-count check for every per-spot input, plus a finiteness check for the
    # numeric ones.  coords feed cKDTree, composition and lib_pred feed lstsq
    # and the SVD basis, lib_size_full feeds rng.binomial, none of the three
    # can degrade gracefully on a NaN, so a non-finite entry is an error here
    # rather than an exception several call frames down.
    _numeric = {"coords", "composition", "lib_pred", "lib_size_full"}
    for nm, v in (("coords", coords), ("labels", labels),
                  ("patient", patient), ("is_train", is_train),
                  ("lib_pred", lib_pred), ("composition", composition),
                  ("lib_size_full", lib_size_full)):
        if v is None:
            continue
        a = np.asarray(v.values if isinstance(v, (pd.Series, pd.DataFrame)) else v)
        if a.shape[0] != n:
            rep.ok = False
            rep.errors.append("%s has %d rows, y_true has %d" % (nm, a.shape[0], n))
            continue
        if nm in _numeric:
            try:
                _nf = _nonfinite_count(a)
            except (TypeError, ValueError):
                _nf = 0
            if _nf:
                rep.ok = False
                rep.errors.append("%s contains %d non-finite value(s)" % (nm, _nf))
    if genes is not None and len(genes) != G:
        rep.ok = False
        rep.errors.append("genes has length %d but y_true has %d columns"
                          % (len(genes), G))
    # ---- presence flags ------------------------------------------------
    # WHAT: record which optional inputs were supplied at all.  WHY: absence
    # is information: each missing input switches off a whole report section
    # (no lib_size = no depth null, no counts = no ceiling), and the report
    # prints these flags so a reader can see what the audit could not test.
    rep.present = {"y_pred": y_pred is not None, "lib_size": lib_size is not None,
                   "coords": coords is not None, "labels": labels is not None,
                   "composition": composition is not None,
                   "counts": counts is not None, "patient": patient is not None,
                   "is_train": is_train is not None, "lib_pred": lib_pred is not None}
    if space is None:
        rep.warnings.append("space= not supplied to check_inputs (audit will raise)")
    return rep


def _seed_of(name) -> int:
    """Deterministic per-section seed. Python's hash() is salted per process,
    so using it here would make point estimates differ between runs.

    WHERE: audit() seeds each section's parent Generator as
    ``default_rng([seed, _seed_of(section_name)])``, the root of the three
    per-section RNG streams [L18].  Consequence: results for one section do
    not change when other sections are added or removed from the run.
    """
    return int(hashlib.sha1(str(name).encode("utf-8")).hexdigest()[:8], 16)


def _grid_plausibility(XY, sec):
    """(median NN spacing, rook-neighbour hit rate) of rint(coords), by section.

    On a genuine integer lattice the spacing is ~1 and almost every spot has a
    lattice neighbour at Manhattan distance 1.  Micron coordinates fail both.

    WHAT: for each section, (a) the median nearest-neighbour distance and
    (b) the fraction of spots that have a rook neighbour (up/down/left/right)
    after rounding coordinates to integers; the medians across sections are
    returned.
    WHY: the user declares coord_kind='grid' or 'micron', but a wrong 'grid'
    declaration on micron positions would make the rook neighbour graph empty
    and silently kill the spatial nulls.  audit() uses these two numbers to
    auto-switch to 'micron' and books the switch in the degradation ledger.
    WHERE: input is XY [L06] and sec [L03] from the coercion layer; the
    verdict changes only ``coord_kind``, never the coordinates themselves.
    """
    from scipy.spatial import cKDTree
    meds, hits = [], []
    for s in np.unique(sec):
        C = XY[sec == s]
        n = len(C)
        if n < 5:
            continue
        d, _ = cKDTree(C).query(C, k=2)
        meds.append(float(np.median(np.atleast_2d(d)[:, -1])))
        g = np.rint(C).astype(np.int64)
        cells = {(int(a), int(b)) for a, b in g}
        hit = np.mean([any((int(a) + dx, int(b) + dy) in cells
                           for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)))
                       for a, b in g])
        hits.append(float(hit))
    if not meds:
        return float("nan"), float("nan")
    return float(np.median(meds)), float(np.median(hits))


# ------------------------------------------------------------- one section
def _one_hot(lbl, classes):
    """(n, n_class) indicator matrix; unknown labels give an all-zero row.

    WHAT: turns categorical tissue labels into a 0/1 matrix, one column per
    class, so a label can act as a regression design column.
    WHY: the composition null asks how much r one gets from knowing only the
    tissue class of each spot; that question needs the classes as numeric
    columns.  The one-hot block plus an intercept is collinear by
    construction (the columns of a full one-hot sum to the intercept);
    ``core.ortho_basis`` absorbs that redundancy via its SVD threshold, so no
    column is dropped here.
    WHERE: audit() calls this after merging small classes into 'other'; the
    result is ``Comp`` [L07], sliced per section into ``comp`` of
    _section_block(), where it enters the composition-null design
    (nulls.py) and the ladder designs X2a/X2 (drg.py) [L32].
    """
    M = np.zeros((len(lbl), len(classes)))
    idx = {c: i for i, c in enumerate(classes)}
    for i, v in enumerate(lbl):
        j = idx.get(v)
        if j is not None:
            M[i, j] = 1.0
    return M


def _section_block(y, yh, loglib, comp, coords, coord_kind, is_train,
                   lib_pred, depth_proxy, null_fit, n_perm, n_boot, perm_kind,
                   top_n, rng, want_smooth=True):
    """All per-section numbers. Returns a plain dict of scalars/arrays.

    Parameters
    ----------
    y, yh : (n, g) ndarray, truth and predictions for ONE section; ``yh`` may
        be None (nulls-only mode).
    loglib : (n,) or None, natural log of the library size.
    comp : (n, K) or None, one-hot labels or continuous composition.
    coords : (n, 2) or None; coord_kind : {'grid', 'micron'}.
    is_train : (n,) bool or None, the user's own train/test flag.
    lib_pred : (n,) or None, a user-supplied depth proxy.
    depth_proxy : {'observed', 'given', 'from_pred', 'none'}.
    null_fit : {'cv_within_section', 'train_only', 'oracle'}.
    n_perm, n_boot : int; perm_kind : {'block', 'free'}; top_n : sequence.
    rng : numpy Generator, the section's own seeded stream.

    Returns
    -------
    dict of raw per-section quantities; ``audit`` turns them into Stats.

    Statistical premise
        Under ``null_fit='train_only'`` the nulls are scored on held-out spots
        only, so THE MODEL IS SCORED ON THE SAME SPOTS: the model side is
        restricted to ``~is_train`` before any rung, lever or floor is
        computed.  Comparing an in-sample model r with an out-of-sample null r
        would bias the headline in exactly the direction this package exists
        to catch.

    Lineage
        Every argument is a per-section slice made in audit()'s section loop:
        ``y``/``yh`` are ``Y[m]``/``P[m]`` [L01]/[L02], ``loglib`` is
        ``loglib[m]`` [L05], ``comp`` is ``Comp[m]`` [L07], ``coords`` is
        ``XY[m]`` [L06], ``is_train`` is ``istr[m]`` [L10] and ``lib_pred``
        is ``lpred[m]`` [L11].  The returned dict is stored as
        ``blocks[section]`` [L38]; audit()'s assembly half reads it through
        the ``col``/``pcol``/``ng`` accessors and funnels the scalars into
        Stats via _combine() [L39].
    """
    out = {}                 # the per-section result dict, returned as blocks[s] [L38]
    n, G = y.shape           # n = spots in THIS section, G = genes in the panel
    out["n_spots"] = n
    out["null_notes"] = {}   # {null name -> first fitting note}, for the report
    # ---- three independent RNG streams --------------------------------- [L18]
    # WHAT: split the section's parent Generator into three children: one for
    # permutations, one for null fitting, one for bootstraps.
    # WHY: independent streams mean point estimates do NOT move when the user
    # changes n_perm / n_boot (reproducibility, not cosmetics): drawing more
    # permutations must not re-deal the random numbers the fits consume.
    # WHERE: the parent rng comes from audit()'s section loop, seeded
    # default_rng([seed, _seed_of(section_name)]).
    _s = rng.integers(0, 2 ** 31 - 1, 3)             # (3,) child seeds
    rng_perm = np.random.default_rng(int(_s[0]))     # permutation stream
    rng = np.random.default_rng(int(_s[1]))          # fitting stream
    rng_boot = np.random.default_rng(int(_s[2]))     # bootstrap stream
    # ---- the shared permutation plan ------------------------------------ [L19]
    # WHAT: draw n_perm null spot orders once.  pidx is (B, n) permuted
    # indices; pmask is (B, n) bools (or None) marking which spots each draw
    # may score.  A "permutation floor" is the value a readout rule produces
    # when spot order is shuffled, i.e. what the rule yields with no signal.
    # WHY: perm_plan (core.py) uses a torus-shift block permutation when
    # coordinates exist, preserving spatial autocorrelation so the floor is
    # not anticonservative; without coordinates it falls back to free
    # shuffling.  ONE plan is drawn per section and reused by every floor
    # below, so floors of different readouts are comparable draw by draw.
    # WHERE: consumed by _perm_of_pred, _perm_pergene, drg.perm_null_of_rung
    # and the Moran loop; rebuilt further down if the ev mask [L23] shrinks n.
    pidx, pmask = perm_plan(n, n_perm, perm_kind, coords, coord_kind, rng_perm)
    out["perm_kind_used"] = ("perm_block" if (perm_kind == "block" and pmask is not None)
                             else "perm_free")
    # ---- the shared cross-fitting partition ----------------------------- [L20]
    # ONE cross-fitting partition per section, drawn before anything else and
    # shared by every null.  Two consequences, both wanted: the difference
    # between two nulls measures the nulls rather than two different random
    # splits, and adding or skipping a null (or a depth proxy) cannot move the
    # value of the others.
    # WHERE: passed as folds= into every nulls.*_null call below; consumed by
    # the cross-fit branch of nulls.linear_null_pred.
    folds = kfold_indices(n, 5, np.random.default_rng(int(_s[1])))  # 5 (train, test) index pairs

    # ---- depth proxy ---------------------------------------------------- [L21]
    # WHAT: pick l_hat, the (n,) depth axis the DG/DRG ladder rungs
    # residualise on.  Three routes: 'observed' copies the measured log
    # library size [L05]; 'given' takes the user's own lib_pred [L11];
    # 'from_pred' cross-fits a kernel-ridge estimate of depth FROM the
    # model's predictions (drg.fit_depth_proxy), with a permutation guard
    # that returns all-NaN when the fit is indistinguishable from chance.
    # WHY: which route was taken changes the meaning of the rung: 'observed'
    # asks "what survives after removing measured depth" (conservative),
    # 'from_pred' asks "how much depth do the predictions themselves carry".
    # WHERE: l_hat feeds the depth_1d null below and the ladder design
    # X1 = [1, lhat] built inside drg.ladder_rungs [L32].
    lhat = None       # (n,) float depth proxy, or None when unavailable
    lhat_note = ""    # provenance string, printed in the report
    if loglib is not None:
        if depth_proxy == "observed":
            lhat = np.asarray(loglib, float)
            lhat_note = "observed log lib (conservative)"
        elif depth_proxy == "given" and lib_pred is not None:
            lhat = np.asarray(lib_pred, float)
            lhat_note = "user-supplied lib_pred"
        elif depth_proxy == "from_pred" and yh is not None:
            # rp = cross-fitted corr(l_hat, log lib), alpha = the ridge
            # penalty the fit selected; both scalars travel to the report as
            # depth_r_of_pred / depth_alpha [L22].
            lhat, rp, alpha, lhat_note = _drg.fit_depth_proxy(yh, loglib, rng)
            if lhat is not None and np.isfinite(lhat).all():
                out["depth_r_of_pred"] = rp
                out["depth_alpha"] = alpha
            else:
                # The selection guard rejected this proxy (or no alpha
                # converged).  Publishing its r against an analytic-zero floor
                # would assert exactly what the guard just refused, so the
                # readout goes N/A and the reason travels in lhat_note.
                out["depth_r_of_pred"] = np.nan
                out["depth_alpha"] = np.nan
        if lhat is not None and not np.isfinite(lhat).all():
            lhat = None
            lhat_note += " (failed)"
    # corr(l_hat, log lib) is a statement ABOUT THE MODEL only when l_hat came
    # from the model.  Under depth_proxy='observed', l_hat IS the observed log
    # library size, so the correlation is 1 by construction and measures
    # nothing; it stays absent rather than printing a tautology as evidence.
    if (lhat is not None and "depth_r_of_pred" not in out
            and depth_proxy != "observed"):
        ok = np.isfinite(lhat) & np.isfinite(loglib)
        out["depth_r_of_pred"] = (float(np.corrcoef(lhat[ok], loglib[ok])[0, 1])
                                  if ok.sum() > 3 and np.std(lhat[ok]) > 0 else np.nan)
    out["lhat_note"] = lhat_note
    out["lhat"] = lhat

    # ---- nulls (no pixels) ---------------------------------------- [L25][L26]
    # WHAT: fit the zero-pixel competitors.  Each nulls.* call returns the
    # triple (r, pred, scored): r is the (g,) per-gene Pearson r of the null
    # (Pearson r = linear correlation, -1 to 1, computed per gene across
    # spots), pred is the null's own (n, g) predictions, scored is the (n,)
    # bool mask of spots the null was allowed to score [L26].
    # WHY: the audit's central question is whether the model beats predictors
    # that never saw an image; every null here uses only per-spot metadata.
    # WHERE: r collapses via agg(r, 'median') to one scalar per section
    # [L25], which audit()'s assembly half reads through col() and combines
    # across sections into a Stat; pred/scored feed _perm_of_pred, whose (B,)
    # output becomes the null's own permutation floor.
    def _null_note(name, notes):
        if notes:
            out["null_notes"][name] = notes[0]

    if loglib is not None:
        # Depth null: predict every gene from log library size alone,
        # design X = [1, loglib] fitted in nulls.depth_null [L05].
        nt = []
        r, pred, scored = _nulls.depth_null(y, loglib, null_fit, is_train, rng,
                                            notes=nt, folds=folds)
        out["depth_null"] = agg(r, "median")[0]        # section scalar [L25]
        out["depth_null_ngene"] = agg(r, "median")[1]  # genes that scored
        out["depth_null_perm"] = _perm_of_pred(y, pred, scored, pidx, pmask)  # (B,) floor draws
        out["depth_null_r"] = r                        # (g,) kept at full resolution [L27]
        _null_note("depth_null", nt)
        # depth_1d asks "what if the image only ever predicted one number?".
        # That question exists only when l_hat comes from the image; under
        # depth_proxy='observed' it would just refit depth_null on other folds.
        if lhat is not None and depth_proxy != "observed":
            nt = []
            r1, pred1, sc1 = _nulls.depth_null(y, lhat, null_fit, is_train, rng,
                                               notes=nt, folds=folds)
            out["depth_1d"] = agg(r1, "median")[0]
            out["depth_1d_ngene"] = agg(r1, "median")[1]
            out["depth_1d_perm"] = _perm_of_pred(y, pred1, sc1, pidx, pmask)
            _null_note("depth_1d", nt)
    if comp is not None:
        # Composition null: predict every gene from tissue class alone,
        # design X = [1, comp] in nulls.composition_null [L07].  Fitted twice:
        # once under the user's null_fit (honest, 'cross_null') and once
        # 'oracle' (in-sample upper bound, barred from the headline).
        nt = []
        r, pred, scored = _nulls.composition_null(y, comp, null_fit, is_train,
                                                  rng, notes=nt, folds=folds)
        out["cross_null"] = agg(r, "median")[0]
        out["cross_null_ngene"] = agg(r, "median")[1]
        out["cross_null_perm"] = _perm_of_pred(y, pred, scored, pidx, pmask)
        _null_note("cross_null", nt)
        nt = []
        r, pred, scored = _nulls.composition_null(y, comp, "oracle", None, rng,
                                                  notes=nt, folds=folds)
        out["oracle_null"] = agg(r, "median")[0]
        out["oracle_null_ngene"] = agg(r, "median")[1]
        out["oracle_null_perm"] = _perm_of_pred(y, pred, scored, pidx, pmask)
        _null_note("oracle_null", nt)
        if loglib is not None:
            # Joint null: depth AND composition together, design
            # X = [1, loglib, comp] in nulls.depth_composition_null; the
            # strongest honest zero-pixel competitor.
            nt = []
            r, pred, scored = _nulls.depth_composition_null(
                y, loglib, comp, null_fit, is_train, rng, notes=nt, folds=folds)
            out["depth_comp_null"] = agg(r, "median")[0]
            out["depth_comp_null_ngene"] = agg(r, "median")[1]
            out["depth_comp_null_perm"] = _perm_of_pred(y, pred, scored, pidx, pmask)
            _null_note("depth_comp_null", nt)
    # ---- neighbour null -------------------------------------------- [L28][L30]
    # WHAT: predict each spot's truth as the mean of its spatial neighbours'
    # TRUE values (W @ y, no fitting at all).  W is the (n, n) sparse
    # row-standardised rook/kNN weight matrix from nulls.neighbour_matrix.
    # WHY: prices pure spatial smoothing: how much r one gets just because
    # neighbouring spots look alike, no pixels, no model.
    # WHERE: the scalar r_nbr [L30] lands in SpatialResult and is duplicated
    # into CeilingResult; W is reused by the Moran readout further down (and
    # rebuilt if the ev mask [L23] drops spots).
    W = S = None
    if coords is not None:
        W = _nulls.neighbour_matrix(coords, coord_kind, mode="rook")  # (n, n) sparse CSR [L28]
        r, pred, has = _nulls.neighbour_null(y, W)
        out["r_nbr"] = agg(r, "median")[0]
        out["r_nbr_ngene"] = agg(r, "median")[1]
        out["r_nbr_perm"] = _perm_of_pred(y, pred, has, pidx, pmask)

    if yh is None:
        return out  # nulls-only mode [L53]: no predictions, no model side

    # ---- evaluation mask: model and nulls must be scored on the SAME spots [L23]
    # Under train_only the nulls above were scored on ~is_train only.  Keeping
    # the model on all spots would put an in-sample model r beside an
    # out-of-sample null r, which flatters the model by exactly the amount the
    # model memorised.  Everything below therefore runs on the held-out spots.
    ev = None  # (n,) bool held-out evaluation mask, or None (= score all spots)
    if null_fit == "train_only" and is_train is not None:
        _tr = np.asarray(is_train, dtype=bool)
        if (~_tr).sum() >= 3:
            ev = ~_tr
    train_key = None  # (g,) per-gene train-side r, the legal gene-ranking key [L24]
    if ev is not None:
        if (~ev).sum() > 5:
            # the train-side ranking key has to be taken before the train
            # spots are dropped; it is the legal (unbiased) selection readout
            # [L24], consumed by levers.train_selected_value further down
            train_key = colcorr(y[~ev], yh[~ev])
        # WHAT: restrict every per-section array to the held-out spots.
        # WHY: the nulls above were scored out-of-sample under train_only;
        # everything below (ladder, levers, spatial) must run on the same
        # spots or the comparison is rigged in the model's favour.
        # WHERE: this is the rebirth point of [L23]: n shrinks, so the
        # permutation plan [L19] and the neighbour matrix [L28] are redrawn
        # on the subset; out['n_eval'] records the new n for the report.
        y = y[ev]
        yh = yh[ev]
        lhat = None if lhat is None else np.asarray(lhat, float)[ev]
        comp = None if comp is None else np.asarray(comp, float)[ev]
        loglib = None if loglib is None else np.asarray(loglib, float)[ev]
        coords = None if coords is None else np.asarray(coords, float)[ev]
        n = int(ev.sum())
        out["n_eval"] = n
        pidx, pmask = perm_plan(n, n_perm, perm_kind, coords, coord_kind, rng_perm)
        kind_model = ("perm_block" if (perm_kind == "block" and pmask is not None)
                      else "perm_free")
        if kind_model != out["perm_kind_used"]:
            out["perm_kind_used"] = "perm_mixed"
        if coords is not None:
            W = _nulls.neighbour_matrix(coords, coord_kind, mode="rook")

    # ---- the model: the attribution ladder -------------------------- [L31][L32]
    # WHAT: score the model at four rungs.  r_full = plain per-gene r of
    # prediction vs truth; dg = r after partialling the depth axis l_hat out
    # of BOTH sides (design X1 = [1, lhat]); crg = same for composition
    # (X2a = [1, comp]); drg = both at once (X2 = [1, lhat, comp]).  Each
    # rung is a (g,) vector from drg.ladder_rungs [L31]; the designs travel
    # alongside in rungs['_designs'] [L32].
    # WHY: the ladder separates "how much r" from "r from what": a rung that
    # collapses when depth is removed was mostly depth.  Both median and
    # mean over genes are kept because the literature mixes the two rules.
    # WHERE: out[k] scalars feed the ladder table and the headline Stats in
    # audit()'s assembly half; out[k+'_pergene'] feeds the per_gene table;
    # the (B,) out[k+'_perm'] arrays are each rung's own permutation floor
    # (drg.perm_null_of_rung keeps the design fixed and permutes spot order).
    rungs = _drg.ladder_rungs(y, yh, lhat, comp)
    designs = rungs.pop("_designs")  # {dg: X1, crg: X2a, drg: X2} control designs [L32]
    for k in ("r_full", "dg", "crg", "drg"):
        v = rungs.get(k)  # (g,) per-gene r of this rung, or None if its control is absent
        if v is None:
            out[k] = np.nan
            continue
        out[k] = agg(v, "median")[0]
        out[k + "_mean"] = agg(v, "mean")[0]
        out[k + "_ngene"] = agg(v, "median")[1]
        out[k + "_pergene"] = v
        X = None if k == "r_full" else designs[k]
        out[k + "_perm"] = _drg.perm_null_of_rung(y, yh, X, pidx, pmask)
        if k == "r_full":
            # the mean-over-genes headline needs the null of the MEAN rule;
            # the median rule's null is a different distribution (measured 28%
            # apart on a skewed per-gene r spread)
            out["r_full_perm_mean"] = _drg.perm_null_of_rung(
                y, yh, X, pidx, pmask, how="mean")  # (B,) floor of the MEAN rule [L36]
        # Spot-bootstrap 95% CI of the rung [L34]: resample spots with
        # replacement (rng_boot stream [L18]) and redo the residualisation
        # per draw; consumed by the ladder table and, for single-section
        # runs, as the fallback CI of _combine.
        ci, note = boot_ci(lambda idx, X=X: _drg.boot_of_rung(y, yh, X, idx),
                           n, n_boot, rng_boot)
        out[k + "_ci"] = ci
        out[k + "_ci_note"] = note
    # ---- conservative DRG using the OBSERVED depth ---------------------- [L35]
    # WHAT: one extra rung with design Xo = [1, loglib, comp], i.e. the
    # MEASURED log library size [L05] instead of the estimated l_hat.
    # WHY: under depth_proxy='from_pred' the main drg rung controls an
    # estimate; this variant cannot be fooled by a bad estimate, so it is
    # the conservative companion printed beside it in the ladder table.
    if loglib is not None and comp is not None:
        Xo = np.hstack([np.ones((n, 1)), np.asarray(loglib).reshape(-1, 1),
                        np.asarray(comp, float)])  # (n, 2+K) design [L35]
        if np.isfinite(Xo).all():
            out["drg_observed"] = agg(_drg._pr(y, yh, Xo), "median")[0]
            out["drg_observed_perm"] = _drg.perm_null_of_rung(y, yh, Xo, pidx, pmask)

    # ---- selection levers ------------------------------------------ [L33][L45]
    # WHAT: price the "report only the best genes" lever.  For each N in
    # top_n, topn_value averages the N highest per-gene r; topn_null applies
    # THE SAME pick-the-best rule to each permuted draw.
    # WHY: selecting the top N genes on the test data inflates r by pure
    # order statistics even when nothing is predictable; the fair floor is
    # what the identical selection rule earns on shuffled predictions
    # (floor_kind='perm_selection' in the assembly half).
    # WHERE: rf comes from the ladder above [L31]; perm_r [L33] is the
    # (B, g) permuted per-gene r matrix from _perm_pergene; out['topn']
    # lands in SelectionResult.by_n [L45].
    rf = rungs["r_full"]                          # (g,) observed per-gene r [L33]
    perm_r = _perm_pergene(y, yh, pidx, pmask)    # (B, g) permuted per-gene r [L33]
    out["topn"] = {}
    for N in top_n:
        v, k_used = _lev.topn_value(rf, N)        # test-selected value + genes used
        out["topn"][N] = (v, _lev.topn_null(perm_r, N), k_used)
    # The legal variant: rank genes on TRAIN spots (key [L24]), score them on
    # test spots.  key exists either from the train_only path above or, on
    # the cv path, is computed here from the user's own split [L10].
    key = train_key
    if key is None and is_train is not None and ev is None \
            and is_train.sum() > 5 and (~is_train).sum() > 5:
        key = colcorr(y[is_train], yh[is_train])
    if key is not None:
        out["topn_train"] = {}
        for N in top_n:
            v, k_used = _lev.train_selected_value(rf, key, N)
            null = np.array([_lev.train_selected_value(pr, key, N)[0]
                             for pr in perm_r])
            out["topn_train"][N] = (v, null, k_used)

    # ---- spatial levers -------------------------------------------- [L29][L37]
    # WHAT: two spatial readouts.  (1) The smoothing lever [L29]: smooth the
    # TRUE y with the kernel S before scoring (ys = S @ y) and record how
    # much r that buys; some papers smooth the ground truth as protocol, and
    # this prices it.  (2) Moran's I of the residuals R = y - yh: whether
    # what the model gets wrong is spatially clustered.
    # WHY: both get their own permutation floors from the shared plan [L19];
    # the Moran loop permutes R and subsets W to the spots each draw scores.
    # WHERE: smooth_* feeds SpatialResult.smooth_lever; moran feeds
    # SpatialResult in the assembly half.
    if coords is not None:
        S = _nulls.smooth_matrix(coords, coord_kind)  # (n, n) sparse smoother [L29]
        ys = np.asarray(S @ y)                        # (n, g) smoothed truth [L29]
        out["smooth_value"] = agg(colcorr(ys, yh), "median")[0]
        out["smooth_delta"] = out["smooth_value"] - out["r_full"]
        sp_perm = _perm_pergene(ys, yh, pidx, pmask)
        out["smooth_perm"] = np.array([np.nanmedian(r) for r in sp_perm])
        out["smooth_delta_perm"] = out["smooth_perm"] - out["r_full_perm"]
        R = y - yh                                    # (n, g) residuals for Moran's I
        out["moran"] = agg(_nulls.morans_I(R, W), "median")[0]
        mp = np.empty(len(pidx))
        for b, pm in enumerate(pidx):
            m = pmask[b] if pmask is not None else slice(None)
            Rp = R[pm] if pmask is None else R[pm[pmask[b]]]
            mp[b] = np.nanmedian(_nulls.morans_I(
                Rp, W if pmask is None else W[np.ix_(pmask[b], pmask[b])]))
        out["moran_perm"] = mp
        # Free-permutation floor of the headline, for contrast [L37]: the
        # same r_full readout floored WITHOUT block structure.  Free
        # shuffling destroys spatial autocorrelation, so this floor is
        # anticonservative; the report prints it beside the block floor
        # (perm_block_vs_free) to show how much the choice matters.
        fidx, _ = perm_plan(n, min(n_perm, 100), "free", None, coord_kind,
                            rng_perm)
        out["r_full_perm_free"] = _drg.perm_null_of_rung(y, yh, None, fidx, None)
    return out


def _perm_of_pred(y, pred, scored, pidx, pmask):
    """Permutation null of a null-model readout: permute the null's predictions.

    WHAT: for each of the B draws of the shared plan, re-scores the null's
    own predictions against truth under a shuffled spot order and takes the
    median over genes; returns a (B,) array of floor draws (or None when
    fewer than 3 spots were scored).
    WHY: even a zero-pixel null needs its own floor: its median r must be
    read against what the same readout gives when spot order is random.  The
    index gymnastics below (remap) exist because the null may have scored
    only a subset of spots [L26], while the permutation plan [L19] was drawn
    on all n spots of the section; a permuted partner outside the scored set
    is dropped from that draw.
    WHERE: y/pred/scored come from the nulls.* triples inside
    _section_block; pidx/pmask are the shared plan from core.perm_plan; the
    result is stored as out['*_perm'] and later folded into make_stat by
    _combine (95th percentile of the draws = the printed floor).
    """
    scored = np.asarray(scored, dtype=bool)
    if scored.sum() < 3:
        return None
    ys = y[scored]
    ps = pred[scored]
    pos = np.where(scored)[0]
    remap = -np.ones(len(y), dtype=np.int64)
    remap[pos] = np.arange(len(pos))
    out = np.empty(len(pidx))
    for b, pm in enumerate(pidx):
        tgt = remap[pm[pos]]
        ok = tgt >= 0
        if pmask is not None:
            ok &= pmask[b][pos]
        if ok.sum() < 3:
            out[b] = np.nan
            continue
        r = colcorr(ys[ok], ps[tgt[ok]])
        out[b] = np.nanmedian(r) if np.isfinite(r).any() else np.nan
    return out


def _perm_pergene(y, yh, pidx, pmask):
    """(n_perm, n_gene) matrix of permuted per-gene r, feeds selection floors.

    WHAT: unlike _perm_of_pred, this keeps the full per-gene resolution of
    each permuted draw instead of collapsing to a median.
    WHY: the selection levers need it: applying "pick the top N genes" to a
    pre-collapsed scalar would be meaningless; the rule must see the whole
    permuted gene vector to reproduce its own selection bias [L33].
    WHERE: called in _section_block for the top-N lever (perm_r), for the
    smoothing lever's floor, and by the standalone perm_floor() wrapper; the
    rows go into levers.topn_null and levers.train_selected_value.
    """
    out = np.empty((len(pidx), y.shape[1]))
    for b, pm in enumerate(pidx):
        if pmask is not None:
            m = pmask[b]
            out[b] = colcorr(y[m], yh[pm[m]])
        else:
            out[b] = colcorr(y, yh[pm])
    return out


# ------------------------------------------------------------- aggregation
def _combine(vals, perms, n_spots, n_genes, floor_kind, rng, note="",
             how="median", fallback_ci=None):
    """Across-section aggregation: section scalars in, one floored Stat out [L39].

    WHAT: the single funnel every aggregated number passes through.  Takes
    one scalar per section (``vals``) plus each section's (B,) permutation
    draws (``perms``), and returns a Stat carrying value, floor and CI [L40].
    WHY: aggregating value and null with the SAME rule (median across
    sections, draw by draw) keeps the floor comparable to the value; a floor
    aggregated differently from its value would not floor anything.
    WHERE: vals/perms are read from blocks[s] [L38] by the col()/pcol()
    accessors in audit()'s assembly half; the Stat is built by core.make_stat
    (floor = 95th percentile of the finite draws) and rendered by report.py.
    """
    # (a) coerce and refuse the all-NaN case (na_stat records the reason)
    v = np.asarray([np.nan if x is None else x for x in vals], dtype=float)  # (n_sections,)
    if not np.isfinite(v).any():
        return na_stat(note or "no section produced a finite value",
                       int(np.sum(n_spots)), 0)
    # (b) the point estimate: nan-median (or mean) across sections
    val = float(np.nanmedian(v)) if how == "median" else float(np.nanmean(v))
    # (c) the permutation floor, aggregated PER DRAW: draw b's floor value is
    #    the median across sections of draw b, mirroring how the value was
    #    formed.  All sections must have contributed (len(good) == len(perms)),
    #    otherwise the floor would cover fewer sections than the value.
    perm = None
    good = [p for p in perms if p is not None]
    if good and len(good) == len(perms):
        L = min(len(p) for p in good)                       # common draw count
        M = np.vstack([np.asarray(p, float)[:L] for p in good])  # (n_sections, L)
        with _np_quiet():
            perm = (np.nanmedian(M, axis=0) if how == "median"
                    else np.nanmean(M, axis=0))             # (L,) aggregated draws
    # (d) the CI: a section-level bootstrap (resample sections, 500 draws)
    #    when there are at least 3 sections; a single-section run falls back
    #    to the spot bootstrap the caller computed (fallback_ci [L34]), with
    #    the kind recorded in the note so the report can say which it is.
    ci = None
    ci_note = ""
    fin = v[np.isfinite(v)]
    if fin.size >= 3:
        draws = np.array([np.nanmedian(fin[rng.integers(0, fin.size, fin.size)])
                          for _ in range(500)])
        ci = (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)))
        ci_note = "ci_kind=section"
    elif fallback_ci is not None:
        ci = fallback_ci
        ci_note = "ci_kind=spot (single section)"
    note = "; ".join(x for x in (note, ci_note) if x)
    return make_stat(val, perm, ci, int(np.sum(n_spots)),
                     int(np.nanmedian(np.asarray(n_genes, float))) if len(n_genes) else 0,
                     floor_kind=floor_kind, note=note)


class _np_quiet:
    """Context manager that silences numpy warnings (all-NaN slices raise
    RuntimeWarning inside nanmedian even when NaN is the intended signal).
    Used only around aggregations that handle NaN deliberately."""

    def __enter__(self):
        self._c = _warnings.catch_warnings()
        self._c.__enter__()
        _warnings.simplefilter("ignore")
        self._e = np.errstate(all="ignore")
        self._e.__enter__()

    def __exit__(self, *a):
        self._e.__exit__(*a)
        self._c.__exit__(*a)


# ------------------------------------------------------------------- audit
def audit(y_true, y_pred=None, *, space=None, section=None, genes=None,
          spot_ids=None, lib_size=None, coords=None, coord_kind="grid",
          labels=None, composition=None, counts=None, lib_size_full=None,
          patient=None, is_train=None, lib_pred=None, space_note="",
          depth_proxy="observed", null_fit="cv_within_section",
          top_n=None, n_perm=200, n_boot=500, perm_kind="auto",
          min_spots=50, min_genes=20, min_class_spots=20,
          min_sections_for_trend=5, n_rep_ceiling=20, seed=0, n_jobs=1,
          strict=True, verbose=True) -> AuditReport:
    """Audit a set of spatial-expression predictions against zero-pixel nulls.

    Parameters
    ----------
    y_true : (n_spot, n_gene) array / DataFrame / sparse
        The evaluation target exactly as the model was scored against; stnull
        never transforms it (decision D2).
    y_pred : same shape or None
        Model predictions.  None runs the nulls only.
    space : str or TargetSpace
        REQUIRED, no default (decision D1): which normalisation ``y_true`` is
        in.  See ``stnull.spaces.SPACE_HELP``.
    section : (n_spot,) labels
        REQUIRED.  Every correlation is computed within a section.
    genes, spot_ids : sequences or None, names only, for the report.
    lib_size : (n_spot,) counts
        FULL-TRANSCRIPTOME total counts per spot, not the panel row sum.  The
        difference moves the composition null by 0.007-0.083 r.
    coords : (n_spot, 2); coord_kind : {'grid', 'micron'}
        Array indices or physical positions.  A 'grid' declaration that fails
        a lattice check is auto-switched to 'micron' and booked in the ledger.
    labels : (n_spot,) categorical  /  composition : (n_spot, K) proportions
        The composition control.  Labels are one-hot encoded; classes with
        fewer than ``min_class_spots`` spots are merged into 'other'.
    counts : (n_spot, n_gene) raw integer counts
        Needed for the measurement ceiling ``r_tech``; there is no fallback.
    lib_size_full : (n_spot,) counts
        Full-transcriptome depth used for the ceiling's rest-of-transcriptome
        thinning; defaults to ``lib_size``.
    patient : (n_spot,) labels
        Enables the split-granularity leakage lever.
    is_train : (n_spot,) bool
        The user's own split; required by ``null_fit='train_only'``.
    lib_pred : (n_spot,) float
        A depth proxy you computed yourself; used by ``depth_proxy='given'``.
    space_note : str, required with ``space='custom'``.
    depth_proxy : {'observed', 'given', 'from_pred', 'none'}
        Where l_hat comes from.  'observed' (default, validated) uses the
        measured log library size; 'from_pred' cross-fits it from y_pred and
        carries a permutation guard; 'none' turns the depth rungs off.
    null_fit : {'cv_within_section', 'train_only', 'oracle'}
        Where the zero-pixel nulls get their parameters.  'oracle' is an
        in-sample UPPER BOUND and is barred from the headline comparison.
    top_n : tuple or None, gene counts for the selection lever; None means
        (10, 50, 100, 250) trimmed to the panel.
    n_perm, n_boot : int, permutation draws for floors, bootstrap draws for
        intervals.  Neither moves any point estimate.
    perm_kind : {'auto', 'block', 'free'}
        'auto' = block whenever coordinates exist.  Free permutation destroys
        spatial autocorrelation and gives ANTICONSERVATIVE floors.
    min_spots, min_genes, min_class_spots, min_sections_for_trend : int
        Power thresholds; sections below ``min_spots`` are refused under
        strict and excluded from aggregates otherwise.
    n_rep_ceiling : int, thinning replicates for ``r_tech``.
    seed : int; n_jobs : int (accepted, single-process only).
    strict : bool
        True (default) refuses to run on inputs that did not validate.
    verbose : bool, per-section progress on stdout.

    Returns
    -------
    AuditReport: Stats, tables, claims and a degradation ledger.  Every
    exposed value carries the permutation floor of its own readout rule.

    Statistical premises
        Per-gene Pearson r, aggregated over genes (median AND mean, because
        the literature mixes them) and then over sections.  Floors are the
        95th percentile of the permutation null of the SAME rule; intervals
        are percentile bootstraps.  Nothing is pooled across sections.

    Not applicable when
        The audit cannot invent what it was not given: no lib_size means the
        depth rungs are N/A, no labels/composition means the composition
        section is N/A (stnull will not cluster to manufacture labels), no
        counts means no ceiling.  Each omission is booked in the ledger with
        the direction in which it biases the headline.
    """
    t_space = coerce_space(space, space_note)  # TargetSpace enum member [L12]; raises if space is missing
    # Closed sets, checked before anything is computed: an unrecognised value
    # used to fall through to a default while RunMeta still printed the string
    # the user passed, i.e. an archived report that names a protocol the run
    # did not use.
    if null_fit not in _NULL_FITS:
        raise BadInput("unknown null_fit=%r; use one of %s"
                       % (null_fit, "|".join(_NULL_FITS)))
    if depth_proxy not in _DEPTH_PROXIES:
        raise BadInput("unknown depth_proxy=%r; use one of %s"
                       % (depth_proxy, "|".join(_DEPTH_PROXIES)))
    if perm_kind not in _PERM_KINDS:
        raise BadInput("unknown perm_kind=%r; use one of %s"
                       % (perm_kind, "|".join(_PERM_KINDS)))
    rng = np.random.default_rng(seed)  # run-level Generator; sections re-seed from it [L18]
    warns: List[str] = []              # non-ledger warnings, -> AuditReport.warnings [L49]
    na: Dict[str, str] = {}            # {report section -> why it is absent} [L50]
    # top_n=None means "the package default": that default must adapt to the
    # panel instead of crashing a 50-gene (HEST-style) run under strict=True.
    _topn_default = top_n is None
    if top_n is None:
        top_n = (10, 50, 100, 250)
    degr: List[str] = []  # the degradation ledger [L48]: every fallback taken
    #                       during the run, each with the direction it biases
    #                       the headline; printed under caveats in report.py

    # ---- structural validation before any statistic exists ------------- [L52]
    # WHAT: run check_inputs on the raw user objects.  WHY: under strict=True
    # (the default) a shape mismatch or a poisoned lib_size refuses to run at
    # all; under strict=False the errors are demoted to warnings on the
    # report, so a degraded run is still labelled as degraded.
    chk = check_inputs(y_true=y_true, y_pred=y_pred, section=section,
                       lib_size=lib_size, coords=coords, labels=labels,
                       composition=composition, counts=counts, patient=patient,
                       is_train=is_train, lib_pred=lib_pred, genes=genes,
                       lib_size_full=lib_size_full, min_spots=min_spots,
                       min_genes=min_genes, space=space)
    if not chk.ok:
        if strict:
            raise InsufficientData(
                "strict=True and the inputs did not validate:\n" + str(chk))
        warns.extend("degraded input: " + e for e in chk.errors)
    warns.extend(chk.warnings)

    # ------------------------------------------------- input coercion layer
    # WHAT: turn every user input into a plain, dense, row-aligned numpy
    # object exactly once.
    # WHY: one coercion point fixes shapes and dtypes in one place, so every
    # statistic below can slice with boolean masks and never see pandas or
    # sparse types again.
    # WHERE: the raw objects are the audit() keywords, fed either directly,
    # via cli.py:_kwargs (one rename per --*-col flag, [L16]) or via
    # examples/her2st_example.py:main [L17]; the coerced arrays are sliced
    # per section in the loop below and become the parameters of
    # _section_block (this file).
    Y = _mat(y_true, "y_true")    # [L01] (n_all, g) ground truth in the declared space
    if sp.issparse(Y):
        Y = np.asarray(Y.todense(), dtype=np.float64)
    n, G = Y.shape                # n = spots in the whole run, G = genes on the panel
    # (a "spot" is one measured tissue location on the slide; the "panel" is
    # the fixed list of genes being scored)
    P = _mat(y_pred, "y_pred")    # [L02] (n_all, g) model predictions, or None (nulls-only mode)
    if P is not None and sp.issparse(P):
        P = np.asarray(P.todense(), dtype=np.float64)
    sec = _vec(section, "section", n).astype(str)    # [L03] (n_all,) tissue-slice label per spot
    pat = _vec(patient, "patient", n)                # [L09] (n_all,) patient label per spot, or None
    pat = pat.astype(str) if pat is not None else None
    lib = _vec(lib_size, "lib_size", n)  # [L04] (n_all,) library size = total UMI count per spot
    # (UMI = one uniquely tagged captured molecule; library size measures how
    # deeply each spot was sequenced, the biggest known confounder here)
    # [L05] loglib: (n_all,) natural log of depth, clipped at 1 so an empty
    # spot gives log(1)=0 instead of -inf.  Born HERE and only here; consumed
    # by the depth null and joint null (nulls.py:depth_null /
    # depth_composition_null via _section_block), the observed-DRG design Xo,
    # the leakage design Xl below [L46], lhat under depth_proxy='observed'
    # [L21], and the per-section sd_loglib.
    loglib = np.log(np.maximum(lib.astype(float), 1.0)) if lib is not None else None
    lpred = _vec(lib_pred, "lib_pred", n)  # [L11] (n_all,) user-supplied depth proxy, or None
    istr = _vec(is_train, "is_train", n)   # [L10] (n_all,) user's own train/test split, or None
    istr = istr.astype(bool) if istr is not None else None
    # --------------------------------------------- configuration repairs
    # WHAT: repair two impossible configurations before any number exists.
    # WHY: an impossible mode must either raise (strict=True) or fall back
    # loudly; a silent fallback would change what every null means without
    # any record of it.
    # WHERE: null_fit and depth_proxy travel into _section_block and decide
    # how nulls are fitted [L10] and where lhat comes from [L21]; the
    # fallback messages land in warns [L49] and degr [L48].
    if null_fit == "train_only" and istr is None:
        msg = "null_fit='train_only' requires is_train"
        if strict:
            raise BadInput(msg + "; pass is_train= or use "
                                 "null_fit='cv_within_section'")
        warns.append(msg + " -> falling back to null_fit='cv_within_section'")
        null_fit = "cv_within_section"
    if depth_proxy == "given" and lpred is None:
        # 'given' with nothing given used to leave l_hat=None, silently
        # switching off DG and DRG with no entry anywhere saying why.
        msg = ("depth_proxy='given' requires lib_pred=; without it there is no "
               "depth proxy at all")
        if strict:
            raise BadInput(msg + "; pass lib_pred= or use "
                                 "depth_proxy='observed'")
        warns.append(msg + " -> falling back to depth_proxy='observed'")
        degr.append(msg + " -> fell back to depth_proxy='observed' (the "
                          "measured log library size)")
        depth_proxy = "observed"
    # ------------------------------------------------------ spot coordinates
    # WHAT: coerce coords to an (n_all, 2) float array XY [L06] and check the
    # declared coord_kind against the actual geometry.
    # WHY: every spatial device (block permutation floors, neighbour nulls,
    # the smoothing lever, Moran's I) hangs on these two columns; micron
    # coordinates mislabelled as 'grid' used to switch all of that off with
    # no visible trace.
    # WHERE: XY[m] goes into _section_block, which feeds core.perm_plan
    # [L19], nulls.neighbour_matrix [L28] and nulls.smooth_matrix [L29].
    XY = None
    if coords is not None:
        XY = np.asarray(coords.values if isinstance(coords, pd.DataFrame)
                        else coords, dtype=float)   # [L06] (n_all, 2) spot positions
        if XY.ndim != 2 or XY.shape[1] < 2:
            raise BadInput("coords must be (n_spot, 2)")
        XY = XY[:, :2]
    if XY is not None and coord_kind == "grid":
        # micron coordinates passed as 'grid' used to degrade SILENTLY:
        # r_nbr/Moran went N/A and the floor fell back to free permutation with
        # empty warnings.  Detect the mismatch and switch to micron handling.
        nn_med, rook_hit = _grid_plausibility(XY, sec)
        if np.isfinite(nn_med) and (nn_med > 2.0 or rook_hit < 0.2):
            msg = ("coord_kind='grid' but the coordinates do not look like an "
                   "integer lattice (median nearest-neighbour spacing %.3g, "
                   "rook-neighbour hit rate %.0f%%) -> switched to "
                   "coord_kind='micron': kNN neighbourhoods, pitch-estimated "
                   "block permutation. Pass coord_kind='micron' yourself to "
                   "silence this warning." % (nn_med, 100.0 * rook_hit))
            warns.append(msg)
            degr.append("coords declared 'grid' failed the lattice check "
                        "(spacing %.3g, rook hit %.0f%%) -> auto-switched to "
                        "'micron' so the spatial section stays on instead of "
                        "silently going N/A with free-permutation floors"
                        % (nn_med, 100.0 * rook_hit))
            coord_kind = "micron"
    # WHAT: resolve gene names [L13]: explicit genes= wins, then DataFrame
    # columns, then synthetic g0..g{G-1}.
    # WHY: names are report cosmetics; no statistic reads them.
    # WHERE: consumed only by the per_gene table rows below [L43].
    gene_names = (list(genes) if genes is not None else
                  (list(y_true.columns) if isinstance(y_true, pd.DataFrame)
                   else ["g%d" % i for i in range(G)]))

    # ---------------------------------------------------- composition design
    # WHAT: build Comp [L07], the (n_all, K) composition matrix, from either
    # continuous proportions (composition=) or categorical labels (labels=),
    # merging classes with fewer than min_class_spots spots into 'other'.
    # WHY: a class-mean null fitted on a 3-spot class is noise; merging keeps
    # the design estimable.  The one-hot block plus an intercept is exactly
    # collinear by construction; core.ortho_basis absorbs that later through
    # its SVD threshold [L32], so no column needs dropping here.
    # WHERE: labels/composition arrive via [L16]/[L17]; Comp[m] goes into
    # _section_block for the composition null and joint null (nulls.py:
    # composition_null / depth_composition_null) and the ladder designs
    # X2a/X2 (drg.py:ladder_rungs) [L32]; the unsliced Comp also enters the
    # leakage design Xl below [L46].
    Comp, comp_kind, class_counts, dropped, n_classes = None, "none", {}, [], 0
    if composition is not None:
        Comp = np.asarray(composition.values if isinstance(composition, pd.DataFrame)
                          else composition, dtype=float)
        if Comp.ndim == 1:
            Comp = Comp[:, None]
        comp_kind = "continuous composition"
        n_classes = Comp.shape[1]
        class_counts = {"comp_%d" % i: int(n) for i in range(n_classes)}
    elif labels is not None:
        lab = _vec(labels, "labels", n).astype(str)
        vals, cnts = np.unique(lab, return_counts=True)
        keep = [v for v, c in zip(vals, cnts) if c >= min_class_spots]
        dropped = [str(v) for v, c in zip(vals, cnts) if c < min_class_spots]
        lab2 = np.where(np.isin(lab, keep), lab, "other")  # small classes renamed 'other'
        classes = sorted(set(lab2.tolist()))               # K class names, sorted for determinism
        Comp = _one_hot(lab2, classes)   # [L07] (n_all, K) 0/1 indicator matrix (_one_hot above)
        comp_kind = "labels (one-hot)"
        n_classes = len(classes)
        class_counts = {str(c): int((lab2 == c).sum()) for c in classes}
        if n_classes < 2:
            Comp = None
            na["composition"] = "fewer than 2 usable classes after merging"
    # ------------------------------------------------ the degradation ledger
    # WHAT: record, before any computation, every missing input and every
    # mode that weakens the audit, each with the DIRECTION it biases the
    # result.
    # WHY: a missing confounder check does not make the model look worse, it
    # makes the audit blind in that direction; the ledger is how a degraded
    # run is prevented from reading like a clean one.
    # WHERE: degr [L48] ends in RunMeta.degradations and report.caveats();
    # na [L50] marks whole report sections absent with the reason; warns
    # [L49] becomes AuditReport.warnings.
    if null_fit == "oracle":
        degr.append("null_fit='oracle' -> every zero-pixel null is fit IN-SAMPLE "
                    "on the scored spots: null values are UPPER BOUNDS, the "
                    "model-vs-null comparison is conservative, and none of these "
                    "numbers may be quoted as an out-of-sample null")
    if null_fit == "train_only":
        degr.append("null_fit='train_only' -> the nulls are fitted on is_train "
                    "and scored on the held-out spots, so EVERY model-side "
                    "readout (r_full, the ladder rungs, top-N, Moran, the "
                    "smoothing lever and their floors) is restricted to those "
                    "same held-out spots; per_section.n_spots is the whole "
                    "section, n_eval is what the model was scored on")
    if Comp is None and "composition" not in na:
        na["composition"] = ("labels/composition not provided; stnull does not "
                             "invent labels by clustering (that would leak the "
                             "model's own information into the null)")
        degr.append("no labels/composition -> section 3 and the CRG/DRG rungs are "
                    "off; the strongest-null column is therefore a LOWER bound")
    if loglib is None:
        na["depth"] = "lib_size not provided"
        degr.append("no lib_size -> section 2 and the DG/DRG rungs are off; the "
                    "largest known confounder in this field was NOT audited, so "
                    "the headline is unaudited in its most vulnerable direction")
    if XY is None:
        na["spatial"] = "coords not provided"
        degr.append("no coords -> permutation floors fall back to free permutation, "
                    "which is anticonservative on spatially autocorrelated data: "
                    "floors are too low and the headline looks too good")
    if counts is None:
        na["ceiling"] = "counts not provided"
        degr.append("no counts -> r_tech is N/A; no approximation is substituted")
    if pat is None:
        degr.append("no patient -> bootstrap stays at spot/section level (CIs are "
                    "narrower than a patient-level bootstrap would give) and the "
                    "leakage lever is N/A")
    if istr is None:
        if null_fit == "oracle":
            degr.append("no is_train and null_fit='oracle' -> nulls are fit "
                        "in-sample on the scored spots (UPPER BOUNDS); only "
                        "test-selected top-N readouts are shown")
        else:
            degr.append("no is_train -> nulls are cross-fitted within section "
                        "(null_fit=cv_within_section), which can be slightly "
                        "optimistic relative to a train-only fit; only "
                        "test-selected top-N readouts are shown")
    if lpred is None and depth_proxy == "from_pred":
        degr.append("no lib_pred -> l_hat is cross-fitted from y_pred inside each "
                    "section; the observed-depth variant is reported alongside as "
                    "the conservative bound")
    if lpred is not None and depth_proxy == "from_pred":
        # A supplied lib_pred wins over cross-fitting, but the user asked for
        # the guarded path and is getting the unguarded one, so say so: the
        # 'given' proxy is used as handed over, with no permutation test that
        # it tracks depth at all.
        depth_proxy = "given"
        msg = ("lib_pred was supplied together with depth_proxy='from_pred' -> "
               "the request was downgraded to depth_proxy='given': l_hat is "
               "your vector as given, and the permutation guard that protects "
               "'from_pred' against a proxy indistinguishable from noise does "
               "NOT run on this path")
        warns.append(msg)
        degr.append(msg)

    # WHAT: trim top-N requests wider than the panel (raise under strict when
    # the user asked for them explicitly).
    # WHY: a "top 200 of 100 genes" readout is the all-gene readout wearing a
    # selection label; the floor_kind would then misdescribe it.
    # WHERE: top_n travels into _section_block, where levers.topn_value /
    # topn_null price each N [L45].
    if P is not None and top_n and max(top_n) > G:
        msg = ("top_n=%s exceeds the number of genes (%d): a top-N readout wider "
               "than the panel is just the all-gene readout under another name"
               % (list(top_n), G))
        if strict and not _topn_default:
            # the user explicitly asked for a readout wider than the panel
            raise BadInput(msg)
        warns.append(msg + (" -> trimmed to fit the panel" if _topn_default else ""))
        top_n = tuple(n for n in top_n if n <= G) or (G,)

    # WHAT: resolve perm_kind='auto': block permutation when coordinates
    # exist, free permutation otherwise.
    # WHY: a "permutation floor" is the r the same readout gives once the
    # spot order is destroyed; any value below it is indistinguishable from
    # noise.  Spots close in space are correlated, so shuffling them freely
    # gives a floor that is too low (anticonservative); the torus-shift block
    # permutation of core.perm_plan [L19] preserves the spatial structure.
    # WHERE: pk goes into _section_block and from there into perm_plan; the
    # kind actually used per section comes back as perm_kind_used and is
    # folded into floor_kind below.
    pk = perm_kind
    if pk == "auto":
        pk = "block" if XY is not None else "free"

    # WHAT: fix the section order once, then check for duplicate coordinates
    # (several spots on one lattice cell).
    # WHY: duplicated cells make lattice neighbour links ill-defined and a
    # torus shift is then not a permutation, so those sections must fall back
    # to free floors and say so.
    # WHERE: sections drives the per-section loop, the ceiling loop and
    # levers.null_transfer_leakage [L46]; the duplicate message goes to warns
    # [L49] and degr [L48].
    sections = sorted(pd.unique(sec).tolist())  # [L03] fixed report order of sections
    if XY is not None:
        _dups = []
        for s in sections:
            m = sec == s
            g = np.rint(XY[m]) if coord_kind == "grid" else np.asarray(XY[m])
            if len(np.unique(g, axis=0)) < int(m.sum()):
                _dups.append("%s (%d spots on %d cells)"
                             % (s, int(m.sum()), len(np.unique(g, axis=0))))
        if _dups:
            msg = ("duplicate coordinates in %d section(s): %s -> lattice "
                   "neighbour links are ill-defined there and torus shifts are "
                   "not permutations; block permutation falls back to FREE for "
                   "those sections (anticonservative floors)"
                   % (len(_dups), "; ".join(_dups[:6])))
            warns.append(msg)
            degr.append(msg)
    # ------------------------------------------------- the per-section loop
    # WHAT: slice every coerced array by the section mask and run the whole
    # per-section machinery (_section_block) once per section.
    # WHY: every r in this package is within-section by design; pooling spots
    # across sections would let section identity itself act as a predictor.
    # WHERE: Y[m]/P[m] [L01]/[L02], loglib[m] [L05], Comp[m] [L07], XY[m]
    # [L06], istr[m] [L10] and lpred[m] [L11] become the parameters
    # y/yh/loglib/comp/coords/is_train/lib_pred of _section_block; the dict
    # it returns is stored in blocks [L38], the raw material of every
    # aggregate below.
    rows_sec, rows_gene, blocks = [], [], {}  # [L42]/[L43]/[L38] accumulators
    for s in sections:
        m = sec == s     # (n_all,) bool mask: the spots of section s
        if m.sum() < min_spots:
            warns.append("section %s: %d spots < min_spots=%d -> underpowered, "
                         "excluded from cross-section aggregates"
                         % (s, int(m.sum()), min_spots))
            continue
        if verbose:
            sys.stdout.write("[stnull] section %s (n=%d)\n" % (s, int(m.sum())))
            sys.stdout.flush()
        # per-section seeded stream [L18]: seed plus a stable hash of the
        # section name, so adding or removing one section cannot move any
        # other section's numbers
        b = _section_block(
            Y[m], None if P is None else P[m],
            None if loglib is None else loglib[m],
            None if Comp is None else Comp[m],
            None if XY is None else XY[m], coord_kind,
            None if istr is None else istr[m],
            None if lpred is None else lpred[m],
            depth_proxy, null_fit, n_perm, n_boot, pk, top_n,
            np.random.default_rng([seed, _seed_of(s)]))
        b["section"] = s
        b["patient"] = pat[m][0] if pat is not None else None
        # spread of log depth within the section [L05]; consumed by the
        # depth trend (trend_sd_vs_r) in the depth assembly below
        b["sd_loglib"] = float(np.std(loglib[m])) if loglib is not None else np.nan
        blocks[s] = b
        # WHAT: one flat per_section row [L42] per section; r_full is renamed
        # r_model for the table.
        # WHY: a flat CSV-able table is what downstream analysis scripts
        # want; the Stat objects with floors are built separately below.
        # WHERE: rows_sec becomes AuditReport.per_section via the DataFrame
        # after the loop; report.to_csv_dir writes it out.
        row = {"section": s, "patient": b["patient"], "n_spots": b["n_spots"],
               "n_eval": b.get("n_eval", b["n_spots"])}
        for k in ("r_full", "depth_null", "depth_1d", "cross_null", "oracle_null",
                  "depth_comp_null", "r_nbr", "dg", "crg", "drg", "drg_observed",
                  "moran", "smooth_delta", "sd_loglib", "depth_r_of_pred"):
            row[k if k != "r_full" else "r_model"] = b.get(k, np.nan)
        rows_sec.append(row)
        # WHAT: one per_gene row [L43] per (section, gene) with the four
        # rung values kept at full per-gene resolution.
        # WHERE: the vectors were stored by _section_block from
        # drg.ladder_rungs [L31]; gene names come from gene_names [L13];
        # rows_gene becomes AuditReport.per_gene.
        if P is not None and "r_full_pergene" in b:
            for gi, g in enumerate(gene_names):
                rec = {"section": s, "gene": g, "n_spot": b["n_spots"]}
                for k in ("r_full", "dg", "crg", "drg"):
                    v = b.get(k + "_pergene")
                    rec[k] = float(v[gi]) if v is not None else np.nan
                rows_gene.append(rec)
    if not blocks:
        raise InsufficientData("no section survived min_spots=%d" % min_spots)

    # A null that could not be fitted, or a depth proxy that failed its guard,
    # must say so: without this the section simply disappears from the
    # across-section median with an empty warning list.
    # WHERE: null_notes were filled by _null_note inside _section_block; the
    # grouped messages fan out to warns [L49], degr [L48] and na [L50], so
    # the absence is visible in the console caveats, the JSON and the HTML.
    _null_msgs: Dict[str, List[str]] = {}   # message text -> list of affected sections
    for s, b in blocks.items():
        for nm, txt in (b.get("null_notes") or {}).items():
            _null_msgs.setdefault("%s: %s" % (nm, txt), []).append(s)
    for txt, secs in sorted(_null_msgs.items()):
        msg = ("%d/%d section(s) could not fit a zero-pixel null -- %s (%s)"
               % (len(secs), len(blocks), txt, ", ".join(sorted(secs)[:6])))
        warns.append(msg)
        degr.append(msg)
        na.setdefault(txt.split(":")[0], txt)
    # WHAT: the same grouping for sections whose depth proxy failed its
    # guard; lhat_note carries the reason from drg.fit_depth_proxy [L21].
    # WHY: a rejected proxy makes DG/DRG N/A in that section, and the report
    # must say where and why instead of thinning the median silently.
    _lhat_msgs: Dict[str, List[str]] = {}
    for s, b in blocks.items():
        note = b.get("lhat_note", "")
        if note and ("not distinguishable" in note or "did not converge" in note
                     or "failed" in note or "too few spots" in note):
            _lhat_msgs.setdefault(note, []).append(s)
    for txt, secs in sorted(_lhat_msgs.items()):
        msg = ("depth proxy unusable in %d/%d section(s) (%s): %s -> the DG/DRG "
               "rungs are N/A there" % (len(secs), len(blocks),
                                        ", ".join(sorted(secs)[:6]), txt))
        warns.append(msg)
        degr.append(msg)

    per_section = pd.DataFrame(rows_sec)  # [L42] one row per surviving section
    per_gene = pd.DataFrame(rows_gene)    # [L43] one row per (section, gene)
    order = [s for s in sections if s in blocks]  # sections that survived min_spots, report order
    nsp = [blocks[s]["n_spots"] for s in order]   # spots per section: the n_spots of every Stat
    if pk == "block":
        _fell = [s for s in order if blocks[s].get("perm_kind_used") == "perm_free"]
        if _fell:
            degr.append("block permutation requested but fell back to FREE in "
                        "%d/%d section(s) (%s): the torus embedding covered "
                        "<50%% of spots or the lattice was degenerate; floors "
                        "in those sections are anticonservative"
                        % (len(_fell), len(order), ", ".join(_fell[:8])))
    if coord_kind == "micron" and XY is not None:
        degr.append("coord_kind='micron' -> coords snapped to a pitch-estimated "
                    "pseudo-grid for block permutation and the 3x3 smoothing "
                    "lever (kNN is used for neighbour nulls); if the snap is "
                    "poor the permutation falls back to FREE, which is now "
                    "recorded in the entry above and per section in "
                    "perm_kind_used")
    # WHAT: the floor label of the run: the single permutation kind if all
    # sections agree, 'perm_mixed' otherwise.
    # WHY: a Stat must not claim a block floor when some sections fell back
    # to free permutation; the label travels on the number itself [L40].
    # WHERE: perm_kind_used was set in _section_block from what perm_plan
    # [L19] actually delivered; floor_kind is passed to every _combine call
    # below and printed by report.py beside each floor.
    used_kinds = {blocks[s]["perm_kind_used"] for s in order}
    floor_kind = used_kinds.pop() if len(used_kinds) == 1 else "perm_mixed"

    # Three accessors over the per-section blocks, in the fixed section order:
    # col = one value per section, pcol = one permutation array per section
    # (None where the floor could not be built), ng = one finite-gene count.
    # WHERE: all three read blocks [L38] and feed every _combine call below
    # [L39]; the r_per_gene chain [L25] passes through col() on its way from
    # nulls.py to a floored Stat.
    def col(key):
        """Per-section values of one readout, NaN where the section has none."""
        return [blocks[s].get(key, np.nan) for s in order]

    def pcol(key):
        """Per-section permutation arrays; None propagates and kills the floor."""
        return [blocks[s].get(key) for s in order]

    def ng(key):
        """Per-section count of genes that produced a finite r for that readout."""
        return [blocks[s].get(key, 0) for s in order]

    # ---------------------------------------------------------- ladder table
    # WHAT: one row per section holding the four rungs (r_full, dg, crg,
    # drg) plus drg_observed, each with its own 95th-percentile permutation
    # floor and spot-bootstrap CI.
    # WHY: the ladder is the central table: what correlation survives when
    # depth (dg), composition (crg) or both (drg) are partialled out of BOTH
    # truth and prediction; a rung only counts when its CI lower bound clears
    # its own floor (drg_above_floor).
    # WHERE: the rung values and their *_perm/*_ci companions were computed
    # in _section_block via drg.ladder_rungs [L31], drg.perm_null_of_rung
    # [L32] and drg.boot_of_rung [L34]; the frame becomes AuditReport.ladder
    # [L44], is drawn by report._plot('ladder') and quoted by the DRG claim
    # in report.claims.
    lad_rows = []
    for s in order:
        b = blocks[s]
        rec = {"section": s, "patient": b["patient"], "n_spots": b["n_spots"]}
        for k in ("r_full", "dg", "crg", "drg", "drg_observed"):
            v = b.get(k, np.nan)
            rec[k] = v
            pm = b.get(k + "_perm")
            # floor = 95th percentile of the finite permuted draws; fewer
            # than 5 finite draws is not a distribution, so NaN
            rec[k + "_floor"] = (float(np.percentile(pm[np.isfinite(pm)], 95))
                                 if pm is not None and np.isfinite(pm).sum() >= 5
                                 else np.nan)
            ci = b.get(k + "_ci")
            rec[k + "_ci_lo"] = ci[0] if ci else np.nan
            rec[k + "_ci_hi"] = ci[1] if ci else np.nan
        lo, fl = rec.get("drg_ci_lo", np.nan), rec.get("drg_floor", np.nan)
        rec["drg_above_floor"] = (bool(lo > fl) if np.isfinite(lo) and np.isfinite(fl)
                                  else False)
        rec["n_genes"] = b.get("r_full_ngene", 0)
        lad_rows.append(rec)
    ladder_df = pd.DataFrame(lad_rows)

    # ------------------------------------------------------------- headline
    # WHAT: the run-level headline: median-over-genes r (r_med), the mean
    # rule (r_mean), the strongest zero-pixel null [L41], and the fraction
    # of sections where the model beats it.
    # WHY: Pearson r (the linear correlation between predicted and true
    # expression) means nothing on its own; the headline is only readable
    # beside the best predictor that never saw a pixel.
    # WHERE: per-section scalars arrive through col()/pcol() from blocks
    # [L38]; _combine [L39] funnels them into floored Stats via
    # core.make_stat; the Headline is rendered by report.summary and by
    # claim 1 in report.claims.
    headline = None
    if P is not None:
        r_med = _combine(col("r_full"), pcol("r_full_perm"), nsp, ng("r_full_ngene"),
                         floor_kind, rng,
                         fallback_ci=blocks[order[0]].get("r_full_ci"))
        # the floor of the mean-over-genes rule, NOT of the median rule: the
        # two null distributions differ whenever the per-gene r spread is
        # skewed, which it always is on a real panel
        r_mean = _combine(col("r_full_mean"), pcol("r_full_perm_mean"), nsp,
                          ng("r_full_ngene"), floor_kind, rng,
                          note="mean over genes")
        cands = []
        for key, nm in (("depth_null", "depth_null"), ("cross_null", "cross_null"),
                        ("depth_comp_null", "depth_comp_null"), ("r_nbr", "r_nbr")):
            if null_fit == "oracle" and nm != "r_nbr":
                # same rule as the composition oracle (decision D7): a null fit
                # in-sample on the scored spots is an UPPER_BOUND and must not
                # be quoted as the strongest achievable null on the headline.
                continue
            vals = col(key)
            if np.isfinite(np.asarray(vals, float)).any():
                st = _combine(vals, pcol(key + "_perm"), nsp,
                              ng(key + "_ngene"), floor_kind, rng)
                cands.append((nm, st))
        strongest = max(cands, key=lambda kv: (kv[1].value if
                                               np.isfinite(kv[1].value) else -9)) \
            if cands else None
        if strongest is not None:
            key = strongest[0]
            # WHAT: per-section win indicator (model r_full beats the
            # strongest null) and, where floors exist, the permuted version
            # of the same fraction.
            # WHY: a cross-section median can hide the null winning half the
            # sections; the win fraction is the section-resolved form of the
            # same comparison, floored under its own permutation.
            # WHERE: reads blocks [L38]; becomes
            # Headline.frac_sections_model_beats_null.
            wins = np.array([1.0 if (np.isfinite(blocks[s].get("r_full", np.nan))
                                     and np.isfinite(blocks[s].get(key, np.nan))
                                     and blocks[s]["r_full"] > blocks[s][key]) else 0.0
                             for s in order])
            pm_wins = None
            if all(blocks[s].get("r_full_perm") is not None for s in order):
                L = min(len(blocks[s]["r_full_perm"]) for s in order)
                M = np.vstack([blocks[s]["r_full_perm"][:L] for s in order])
                base = np.array([blocks[s].get(key, np.nan) for s in order])[:, None]
                pm_wins = (M > base).mean(axis=0)
            frac = make_stat(float(wins.mean()), pm_wins, None, len(order),
                             int(np.nanmedian(ng("r_full_ngene"))),
                             floor_kind=floor_kind,
                             note="fraction of sections; n= number of sections")
        else:
            frac = na_stat("no null model available" if null_fit != "oracle" else
                           "no OUT-OF-SAMPLE null available: null_fit='oracle' "
                           "nulls are UPPER_BOUND and barred from the headline",
                           len(order), 0)
        headline = Headline(r_median=r_med, r_mean=r_mean,
                            strongest_null=strongest,
                            frac_sections_model_beats_null=frac)

    # ---------------------------------------------------------------- depth
    # WHAT: assemble DepthResult: the depth null, the 1-D depth readout, the
    # DG rung, the share of the headline removed by the depth control, and
    # corr(l_hat, log lib) [L22].
    # WHY: sequencing depth (library size) is the largest known confounder
    # in this field: a model that only predicts "this spot was sequenced
    # deeper" already earns nonzero r whenever the target space carries
    # depth.
    # WHERE: the per-section values were computed in _section_block from
    # loglib [L05] and lhat [L21]; the aggregated Stats follow the [L25]
    # chain; DepthResult is report section [2] and feeds claim 2 in
    # report.claims.
    depth = None
    # when null_fit='oracle' every fitted null is an in-sample upper bound and
    # must say so on the number itself, not only in the ledger
    _ub = ("; UPPER_BOUND: null_fit='oracle' fits on the scored spots, not "
           "eligible to be the strongest null" if null_fit == "oracle" else "")
    if loglib is not None:
        dn = _combine(col("depth_null"), pcol("depth_null_perm"), nsp,
                      ng("depth_null_ngene"), floor_kind, rng,
                      note="zero pixels: one scalar (log lib) per spot" + _ub)
        # depth_1d and corr(l_hat, log lib) are statements about the IMAGE
        # pipeline: they only exist when l_hat came from the model.  Under the
        # default depth_proxy='observed', l_hat IS the measured log library
        # size, so depth_1d would be depth_null refitted on other folds and the
        # correlation would be 1.0 by construction.  Both are declared N/A with
        # that reason rather than printed under a label they do not deserve.
        _obs_reason = ("depth_proxy='observed': l_hat IS the measured log "
                       "library size, so this readout is fixed by construction "
                       "and says nothing about the model. Pass lib_pred= or "
                       "depth_proxy='from_pred' to measure the depth the model "
                       "itself carries.")
        if depth_proxy == "observed":
            d1 = na_stat(_obs_reason, int(np.sum(nsp)), 0)
        elif P is not None and np.isfinite(np.asarray(col("depth_1d"), float)).any():
            d1 = _combine(col("depth_1d"), pcol("depth_1d_perm"), nsp,
                          ng("depth_1d_ngene"), floor_kind, rng,
                          note="image -> 1 predicted depth -> genes" + _ub)
        else:
            d1 = None
        # WHAT: the DG rung Stat, the share 1 - DG/r_full, and the depth
        # carried by the predictions themselves.
        # WHY: DG is the headline recomputed after the depth axis lhat [L21]
        # is partialled out of both sides; the share turns the drop into a
        # fraction of the headline.
        # WHERE: dg values come from drg.ladder_rungs via _section_block
        # [L31]; the share is built by _share_stat below; depth_r_of_pred is
        # [L22] and lands in report section [2].
        if P is not None:
            dg = _combine(col("dg"), pcol("dg_perm"), nsp, ng("dg_ngene"),
                          floor_kind, rng,
                          fallback_ci=blocks[order[0]].get("dg_ci"))
            share = _share_stat(col("r_full"), col("dg"), pcol("r_full_perm"),
                                pcol("dg_perm"), nsp)
            drp = (na_stat(_obs_reason, int(np.sum(nsp)), 0)
                   if depth_proxy == "observed" else
                   _plain_stat(col("depth_r_of_pred"), nsp,
                               "corr(l_hat, log lib): depth carried by the model; "
                               "floor is analytic 0 for an independent predictor",
                               floor_kind="analytic_zero", floor=0.0))
        else:
            dg = na_stat("NO_MODEL", int(np.sum(nsp)), 0)
            share = na_stat("NO_MODEL")
            drp = na_stat("NO_MODEL")
        # WHAT: per-section table of sd(log lib) beside model r and the
        # depth null, then the across-section trend corr(sd(log lib), r).
        # WHY: if the model's r rises exactly where depth varies more, the
        # headline is riding depth; 2000 free permutations of the section
        # order price the trend's own noise (sections, not spots, are the
        # unit here, so free permutation is the right floor).
        # WHERE: sd_loglib was stored per section in the loop above [L05];
        # sd_df and trend become DepthResult.sd_loglib_by_section and
        # DepthResult.trend_sd_vs_r.
        sd_df = pd.DataFrame({"section": order, "sd_log_lib": col("sd_loglib"),
                              "n_spots": nsp,
                              "r_model": col("r_full") if P is not None else np.nan,
                              "depth_null": col("depth_null")})
        trend = None
        if len(order) >= min_sections_for_trend and P is not None:
            a = np.asarray(col("sd_loglib"), float)
            bb = np.asarray(col("r_full"), float)
            ok = np.isfinite(a) & np.isfinite(bb)
            if ok.sum() >= min_sections_for_trend and np.std(a[ok]) > 0 and np.std(bb[ok]) > 0:
                tv = float(np.corrcoef(a[ok], bb[ok])[0, 1])
                nullv = np.array([float(np.corrcoef(
                    a[ok], rng.permutation(bb[ok]))[0, 1]) for _ in range(2000)])
                trend = make_stat(tv, nullv, None, len(order), 1,
                                  floor_kind="perm_free",
                                  note="across-section corr(sd(log lib), model r)")
        depth = DepthResult(
            depth_null=dn, depth_1d=d1, dg=dg, share_depth=share,
            depth_r_of_pred=drp, sd_loglib_by_section=sd_df,
            trend_sd_vs_r=trend,
            lib_denominator_check=(
                "lib_size must be the FULL-TRANSCRIPTOME total count. Using the "
                "panel row sum instead moves the composition null by 0.007-0.083 r "
                "on the reference dataset. Declared space: %s." % t_space.value),
            target_carries_depth=(t_space in DEPTH_CARRYING))

    # ---------------------------------------------------------- composition
    # WHAT: assemble CompositionResult: the cross-fitted class-mean null,
    # the oracle (in-sample) class-mean null, and the CRG rung.
    # WHY: predicting each spot as its tissue-class mean uses no pixels yet
    # scores well wherever composition drives expression; the oracle variant
    # is deliberately an UPPER_BOUND and is labelled so on the Stat itself,
    # in addition to the ledger entry.
    # WHERE: values come from nulls.composition_null via _section_block and
    # follow the [L25] chain; the design was Comp [L07]; CompositionResult
    # is report section [3].
    comp_res = None
    if Comp is not None:
        cn = _combine(col("cross_null"), pcol("cross_null_perm"), nsp,
                      ng("cross_null_ngene"), floor_kind, rng,
                      note="class means, %s" % null_fit + _ub)
        on = _combine(col("oracle_null"), pcol("oracle_null_perm"), nsp,
                      ng("oracle_null_ngene"), floor_kind, rng,
                      note="UPPER_BOUND: class means read off the scored section; "
                           "not eligible to be the strongest null")
        crg = (_combine(col("crg"), pcol("crg_perm"), nsp, ng("crg_ngene"),
                        floor_kind, rng,
                        fallback_ci=blocks[order[0]].get("crg_ci"))
               if P is not None else na_stat("NO_MODEL"))
        comp_res = CompositionResult(cross_null=cn, oracle_null=on, crg=crg,
                                     n_classes=n_classes, class_counts=class_counts,
                                     small_classes_dropped=dropped, kind=comp_kind)

    # -------------------------------------------------------------- spatial
    # WHAT: assemble SpatialResult: the neighbour-mean null r_nbr [L30], the
    # fraction of sections where it beats the model, the ground-truth
    # smoothing lever [L29], Moran's I of the residuals, and the
    # block-vs-free floor contrast [L37].
    # WHY: expression is spatially smooth, so a zero-pixel average of the
    # TRUE y over neighbouring spots is a serious competitor, and a free
    # floor on smooth data is too low; both facts are priced here.
    # WHERE: r_nbr comes from nulls.neighbour_null via _section_block, the
    # smoothing pieces from nulls.smooth_matrix [L29]; SpatialResult is
    # report section [4] and r_nbr is copied into CeilingResult below.
    spatial = None
    if XY is not None:
        rn = _combine(col("r_nbr"), pcol("r_nbr_perm"), nsp, ng("r_nbr_ngene"),
                      floor_kind, rng,
                      note="mean of the neighbours' TRUE y; the image is never used")
        if P is not None:
            wins = np.array([1.0 if (np.isfinite(blocks[s].get("r_nbr", np.nan)) and
                                     np.isfinite(blocks[s].get("r_full", np.nan)) and
                                     blocks[s]["r_nbr"] > blocks[s]["r_full"]) else 0.0
                             for s in order])
            frac_n = make_stat(float(wins.mean()), None, None, len(order), 1,
                               floor_kind="none",
                               note="fraction of sections where a zero-pixel "
                                    "smoother beats the model; n = sections")
            sm = _combine(col("smooth_delta"), pcol("smooth_delta_perm"), nsp,
                          ng("r_full_ngene"), floor_kind, rng,
                          note="delta in the headline when the GROUND TRUTH is "
                               "3x3 smoothed")
            mo = _combine(col("moran"), pcol("moran_perm"), nsp, ng("r_full_ngene"),
                          floor_kind, rng,
                          note="Moran's I of (y - y_hat); high = the model left a "
                               "smooth spatial field on the table")
            # WHAT: the same headline value under its block floor and under
            # the free floor, kept as a pair [L37].
            # WHY: the gap between the two floors is itself a finding: it is
            # how much a free-shuffle floor flatters spatially smooth data.
            # WHERE: r_full_perm_free was drawn in _section_block from a
            # small free perm_plan; the pair becomes
            # SpatialResult.perm_block_vs_free and two HTML rows in
            # report.py.
            bf = None
            if any(blocks[s].get("r_full_perm_free") is not None for s in order):
                blk = _combine(col("r_full"), pcol("r_full_perm"), nsp,
                               ng("r_full_ngene"), "perm_block", rng,
                               note="same value, block floor")
                fre = _combine(col("r_full"), pcol("r_full_perm_free"), nsp,
                               ng("r_full_ngene"), "perm_free", rng,
                               note="same value, FREE floor (anticonservative)")
                bf = (blk, fre)
            spatial = SpatialResult(r_nbr=rn, frac_sections_nbr_beats_model=frac_n,
                                    smooth_lever=sm, moran_I_resid=mo,
                                    perm_block_vs_free=bf)
        else:
            spatial = SpatialResult(r_nbr=rn,
                                    frac_sections_nbr_beats_model=na_stat("NO_MODEL"),
                                    smooth_lever=na_stat("NO_MODEL"),
                                    moran_I_resid=na_stat("NO_MODEL"),
                                    perm_block_vs_free=None)

    # ------------------------------------------------------------ selection
    # WHAT: assemble SelectionResult: test-selected top-N readouts [L45],
    # the legal train-selected version, the split-leakage lever [L46], and
    # the assembled lever price table.
    # WHY: picking the reported genes on the test set inflates r by
    # construction; each levered value is therefore shown beside the floor
    # of ITS OWN selection rule (floor_kind='perm_selection'), never the
    # all-gene floor.
    # WHERE: the per-section (value, null, k) triples were computed in
    # _section_block by levers.topn_value / topn_null [L33][L45];
    # SelectionResult is report section [5].
    selection = None
    if P is not None:
        by_n = {}
        # one floored Stat per requested N; the note on each Stat records
        # that the ranking happened on the test set
        for N in top_n:
            vals = [blocks[s]["topn"][N][0] for s in order]
            perms = [blocks[s]["topn"][N][1] for s in order]
            by_n[N] = _combine(vals, perms, nsp,
                               [blocks[s]["topn"][N][2] for s in order],
                               "perm_selection", rng,
                               note="genes ranked by r ON THE TEST SET")
        # WHAT: the legal counterpart: genes ranked by the TRAIN-side key
        # [L24], scored on the held-out spots.
        # WHY: the gap between by_n and tr_n is the price of test-set
        # selection, the lever this section exists to expose.
        tr_n = None
        if all("topn_train" in blocks[s] for s in order):
            tr_n = {}
            for N in top_n:
                tr_n[N] = _combine([blocks[s]["topn_train"][N][0] for s in order],
                                   [blocks[s]["topn_train"][N][1] for s in order],
                                   nsp, [blocks[s]["topn_train"][N][2] for s in order],
                                   "perm_selection", rng,
                                   note="genes ranked on TRAIN spots (legal)")
        # WHAT: the split-granularity lever: how much r a zero-pixel null
        # gains when it may train on OTHER SECTIONS OF THE SAME PATIENT
        # instead of other patients only.
        # WHY: a section-level split leaks patient identity; measuring the
        # premium with a pixel-free null makes it a lower bound on what a
        # real model can collect, without retraining anything.
        # WHERE: Xl is built here from loglib [L05] and Comp [L07] and
        # handed with the unsliced Y [L01] to levers.null_transfer_leakage
        # [L46]; the median delta becomes SelectionResult.leakage_lever.
        leak = None
        leak_df = None
        if pat is not None and (loglib is not None or Comp is not None):
            Xcols = [np.ones((n, 1))]
            if loglib is not None:
                Xcols.append(loglib.reshape(-1, 1))
            if Comp is not None:
                Xcols.append(Comp)
            Xl = np.hstack(Xcols)  # [L46] (n_all, 1+1+K) zero-pixel design [1, log lib, composition]
            leak_df = _lev.null_transfer_leakage(Y, Xl, sec, pat, order)
            dv = leak_df["delta"].values.astype(float)
            if np.isfinite(dv).any():
                leak = make_stat(float(np.nanmedian(dv)), None, None,
                                 int(np.sum(nsp)), int(np.nanmedian(ng("r_full_ngene"))),
                                 floor_kind="none", status="degraded",
                                 note="zero-pixel leakage premium: null fitted on "
                                      "OTHER SECTIONS OF THE SAME PATIENT minus null "
                                      "fitted on other patients. stnull never trains "
                                      "models, so this is a lower bound on the "
                                      "premium a real model can collect, and it has "
                                      "no permutation floor of its own")
        # WHAT: the lever price table: one row per lever, each as (base
        # value, levered value, delta) with the floor of each side.
        # WHY: a lever is a protocol choice that buys r without changing the
        # model; the table makes every purchase and its noise price explicit
        # in one place.
        # WHERE: becomes SelectionResult.lever_price, printed as the lever
        # table of report section [5].
        rows = []
        base = headline.r_median.value if headline else np.nan
        for N in top_n:
            st = by_n[N]
            rows.append({"lever": "test-selected top-%d genes" % N,
                         "value_base": base, "value_lever": st.value,
                         "delta": st.value - base, "floor_base": headline.r_median.floor,
                         "floor_lever": st.floor, "floor_kind": "perm_selection",
                         "status": st.status,
                         "note": "the floor is what pure selection noise buys"})
        if spatial is not None and np.isfinite(spatial.smooth_lever.value):
            rows.append({"lever": "3x3 smoothing of the ground truth",
                         "value_base": base,
                         "value_lever": base + spatial.smooth_lever.value,
                         "delta": spatial.smooth_lever.value,
                         "floor_base": headline.r_median.floor,
                         "floor_lever": spatial.smooth_lever.floor,
                         "floor_kind": spatial.smooth_lever.floor_kind,
                         "status": spatial.smooth_lever.status,
                         "note": "ground truth is smoothed, the model is not retrained"})
        if leak is not None:
            rows.append({"lever": "section-level split leakage (zero-pixel proxy)",
                         "value_base": np.nan, "value_lever": np.nan,
                         "delta": leak.value, "floor_base": np.nan,
                         "floor_lever": np.nan, "floor_kind": "none",
                         "status": leak.status,
                         "note": "lower bound; stnull does not retrain models"})
        else:
            if pat is None:
                why = "needs patient= (and lib_size or labels) to measure"
            elif leak_df is not None:
                why = ("patient= was given but no patient contributes 2+ sections "
                       "here, so a section-level split would leak nothing "
                       "measurable in THIS subset")
            else:
                why = "needs lib_size or labels to build the zero-pixel null"
            rows.append({"lever": "section-level split leakage",
                         "value_base": np.nan, "value_lever": np.nan,
                         "delta": np.nan, "floor_base": np.nan, "floor_lever": np.nan,
                         "floor_kind": "none", "status": "na", "note": why})
        rows.append({"lever": "target space / denominator change",
                     "value_base": np.nan, "value_lever": np.nan, "delta": np.nan,
                     "floor_base": np.nan, "floor_lever": np.nan,
                     "floor_kind": "none", "status": "na",
                     "note": "out of scope by decision D2: stnull never transforms "
                             "y. Re-run audit() with the other space to price it."})
        selection = SelectionResult(by_n=by_n, train_selected=tr_n,
                                    lever_price=pd.DataFrame(rows),
                                    leakage_lever=leak)

    # -------------------------------------------------------------- ceiling
    # WHAT: assemble CeilingResult: the technical ceiling r_tech [L47]
    # (split-half counts, Spearman-Brown adjusted) and the model's r as a
    # fraction of it.
    # WHY: two half-depth measurements of the SAME spot disagree with each
    # other; no model can beat the agreement of the data with itself, so
    # r_tech prices the measurement-noise ceiling of THIS dataset.
    # WHERE: raw counts [L08] and the full-transcriptome depth [L14] go into
    # ceiling.r_tech (ceiling.py); the per-section vectors follow the same
    # funnel as every null ([L25]-style agg then _combine [L39]);
    # CeilingResult is report section [6].
    ceil_res = None
    rt_stat = None
    method = "na"
    if counts is not None:
        rt_vals, rt_perms, rt_ngenes, rt_nsp = [], [], [], []
        n_ceil_skip = 0
        for s in order:
            m = sec == s
            C = counts[m] if not sp.issparse(counts) else counts[np.where(m)[0]]
            # [L08] (n_sec, g) raw UMI counts of this section, the only
            # input that can price measurement noise
            lf = (lib_size_full if lib_size_full is not None else lib)
            # [L14] full-transcriptome depth when given, else lib [L04]:
            # honest denominators for the half-depth pseudo-replicates
            lfs = None if lf is None else np.asarray(lf, float)[m]
            r, pnull, method, note = _ceil.r_tech(
                C, t_space, lfs, n_rep=n_rep_ceiling,
                rng=np.random.default_rng([seed, 7, _seed_of(s)]))
            if r is None:
                # one section failing must not silently drop every LATER
                # section (the old `break`); skip it, count it, keep going
                n_ceil_skip += 1
                continue
            v, k = agg(r, "median")
            rt_vals.append(v)
            rt_perms.append(pnull)
            rt_ngenes.append(k)
            rt_nsp.append(blocks[s]["n_spots"])
        if n_ceil_skip:
            warns.append("ceiling: %d of %d section(s) produced no r_tech "
                         "estimate and were skipped (%s)"
                         % (n_ceil_skip, len(order), note))
        if rt_vals:
            rt_stat = _combine(rt_vals, rt_perms, rt_nsp, rt_ngenes,
                               "perm_free", rng,
                               note=method + (", %d/%d sections skipped"
                                              % (n_ceil_skip, len(order))
                                              if n_ceil_skip else ""))
    # WHAT: the derived ratio, headline r over the ceiling.
    # WHY: r=0.25 against a ceiling of 0.5 is a different achievement than
    # 0.25 against 0.9; the ratio is marked status='derived' and its floor
    # is derived from the headline floor, not re-permuted.
    # WHERE: reads headline.r_median [L39] and rt_stat [L47]; becomes
    # CeilingResult.r_over_ceiling.
    r_over = None
    if rt_stat is not None and headline is not None and np.isfinite(rt_stat.value) \
            and rt_stat.value > 0:
        r_over = make_stat(headline.r_median.value / rt_stat.value, None, None,
                           int(np.sum(nsp)), rt_stat.n_genes, floor_kind="derived",
                           status="derived",
                           note="model r as a fraction of THIS DATASET's measurement "
                                "noise ceiling; floor of the ratio = %s"
                                % ("%.4f" % (headline.r_median.floor / rt_stat.value)
                                   if headline.r_median.floor is not None else "n/a"))
    if rt_stat is not None or spatial is not None:
        ceil_res = CeilingResult(r_tech=rt_stat,
                                 r_nbr=spatial.r_nbr if spatial else None,
                                 r_over_ceiling=r_over,
                                 method=method if rt_stat is not None else "na")

    # ------------------------------------------------------------- run card
    # WHAT: freeze the run card: version, seed, timestamp, sizes, which
    # inputs were present, sha1 fingerprints of the exact inputs [L51], the
    # resolved modes, and the full degradation ledger [L48].
    # WHY: an audit that cannot prove what it ran on is just another number;
    # the digests let any two reports be matched to their inputs.
    # WHERE: sha1_of is core.sha1_of; RunMeta is serialised by
    # report.to_json, printed at the top of the console report, and its
    # space field [L12] is the comparability key of compare() [L56].
    run = RunMeta(
        stnull_version=VERSION, seed=seed,
        created_utc=datetime.datetime.now(datetime.timezone.utc)
        .strftime("%Y-%m-%dT%H:%M:%SZ"),
        space=t_space, space_note=space_note, n_spots=int(np.sum(nsp)),
        n_genes=G, n_sections=len(order),
        n_patients=int(len(set(pat.tolist()))) if pat is not None else None,
        inputs_present={"y_pred": P is not None, "lib_size": lib is not None,
                        "coords": XY is not None, "labels": labels is not None,
                        "composition": composition is not None,
                        "counts": counts is not None, "patient": pat is not None,
                        "is_train": istr is not None, "lib_pred": lpred is not None},
        inputs_sha1={"y_true": sha1_of(Y), "y_pred": sha1_of(P),
                     "section": sha1_of(sec), "lib_size": sha1_of(lib),
                     "coords": sha1_of(XY), "labels": sha1_of(labels),
                     "composition": sha1_of(Comp), "patient": sha1_of(pat)},
        null_fit=null_fit, depth_proxy=depth_proxy, perm_kind=floor_kind,
        n_perm=n_perm, n_boot=n_boot, degradations=degr)

    # WHAT: the final container: every result object plus the three ledgers
    # (na [L50], warnings [L49], degradations inside run [L48]).
    # WHERE: AuditReport is what report.summary / to_json / to_csv_dir /
    # to_html render; nothing downstream recomputes any statistic.
    rep = AuditReport(run=run, headline=headline, depth=depth,
                      composition=comp_res, spatial=spatial, selection=selection,
                      ceiling=ceil_res, ladder=ladder_df, per_section=per_section,
                      per_gene=per_gene, na=na, warnings=warns)
    if verbose:
        sys.stdout.write("[stnull] done: %d sections, %d spots, %d genes\n"
                         % (len(order), int(np.sum(nsp)), G))
        sys.stdout.flush()
    return rep


#: Sections whose |r_full| is below this contribute no share: 1 - DG/r_full is
#: a ratio, and a near-zero denominator turns section noise into values of
#: order 1e4 that a 2-section median cannot absorb.  The same threshold is used
#: for the observed ratio and for its permutation null, so the two agree.
_SHARE_MIN_RFULL = 0.01


def _share_stat(rfull, dg, rperm, dgperm, nsp):
    """Share of the headline that the depth control removes: 1 - DG/r_full.

    Returns a Stat with ``status='derived'``: it is a ratio of two floored
    numbers, and it is only meaningful read next to r_full and DG themselves.
    """
    # WHAT: keep only sections where both numbers are finite and |r_full|
    # clears _SHARE_MIN_RFULL.
    # WHY: 1 - DG/r_full explodes when the denominator is near zero; the
    # threshold is applied identically to the observed ratio and to its
    # permutation null below, so the two remain comparable.
    # WHERE: rfull/dg are col("r_full")/col("dg") per section [L38].
    rf = np.asarray(rfull, float)    # (n_sections,) headline r per section
    dgv = np.asarray(dg, float)      # (n_sections,) DG rung per section
    ok = np.isfinite(rf) & np.isfinite(dgv) & (np.abs(rf) > _SHARE_MIN_RFULL)
    if ok.sum() == 0:
        return na_stat("cannot form 1 - DG/r_full")
    val = float(np.nanmedian(1.0 - dgv[ok] / rf[ok]))
    # WHAT: the permuted version of the same ratio, draw by draw, with the
    # same denominator threshold.
    # WHY: a floor must be computed under the same rule as the value [L39];
    # fewer than 5 finite permuted medians is not a distribution, so the
    # floor is dropped rather than faked.
    # WHERE: rperm/dgperm are the per-section r_full_perm/dg_perm arrays
    # from blocks [L38], truncated to a common draw count L.
    perm = None
    if all(p is not None for p in rperm) and all(p is not None for p in dgperm):
        L = min(min(len(p) for p in rperm), min(len(p) for p in dgperm))
        A = np.vstack([np.asarray(p, float)[:L] for p in rperm])
        B = np.vstack([np.asarray(p, float)[:L] for p in dgperm])
        with np.errstate(all="ignore"):
            ratio = 1.0 - B / A
        ratio[np.abs(A) < _SHARE_MIN_RFULL] = np.nan
        with _np_quiet():
            perm = np.nanmedian(ratio, axis=0)
        if np.isfinite(perm).sum() < 5:
            perm = None
    return make_stat(val, perm, None, int(np.sum(nsp)), 1,
                     floor_kind="perm_free" if perm is not None else "none",
                     status="derived",
                     note="1 - DG/r_full: the share of the headline that the depth "
                          "control removes. Ratio of two floored numbers; read it "
                          "with r_full and DG above, not alone")


def _plain_stat(vals, nsp, note, floor_kind="none", floor=None):
    """Median-over-sections Stat with no permutation floor of its own.

    WHAT: wraps a list of per-section scalars into a Stat whose value is the
    nanmedian; an analytic floor can be attached (0 for corr(l_hat, log lib)
    under an independent predictor [L22]).
    WHY: some readouts have no meaningful permutation floor, but make_stat
    still forces an explicit floor_kind onto them so the report can never
    print a bare number [L40].
    WHERE: called from the depth assembly in audit() for depth_r_of_pred
    [L22]; the Stat renders through the same report.py cells as every other.
    """
    v = np.asarray(vals, float)
    if not np.isfinite(v).any():
        return na_stat(note)
    st = make_stat(float(np.nanmedian(v)), None, None, int(np.sum(nsp)), 1,
                   floor_kind=floor_kind, note=note)
    if floor is not None:
        # make_stat derives floors from permutation draws only, so the
        # analytic floor is attached by rebuilding the frozen Stat with the
        # same fields and the floor slotted in
        st = Stat(st.value, float(floor), floor_kind, st.ci, st.n_spots,
                  st.n_genes, st.status, st.note)
    return st


# ------------------------------------------------------------ extra entries
def nulls_only(y_true, *, space=None, section=None, **kw) -> AuditReport:
    """audit() with no model: 'how much r can zero pixels buy on my data?'

    WHAT: forwards everything to audit() with y_pred=None [L53].
    WHY: the nulls are worth publishing on their own: they are the floor any
    future model on this data must beat.
    WHERE: same inputs as audit() [L01]-[L14]; y_pred is stripped, so the
    model side of every section block stays off.
    """
    kw.pop("y_pred", None)
    return audit(y_true, None, space=space, section=section, **kw)


def ladder(y_true, y_pred, *, space=None, section=None, lib_size=None,
           labels=None, **kw) -> pd.DataFrame:
    """Just the three-rung ladder table.

    WHAT: runs the full audit() and returns AuditReport.ladder [L54][L44].
    WHY: the ladder is the piece most readers want first; this saves them
    the attribute chain, at the cost of running everything.
    WHERE: same keywords as audit(); the frame is the one described at the
    ladder-table block inside audit().
    """
    rep = audit(y_true, y_pred, space=space, section=section, lib_size=lib_size,
                labels=labels, **kw)
    return rep.ladder


def perm_floor(y_true, y_pred, *, section, rule="all", top_n=None, kind="block",
               coords=None, coord_kind="grid", n_perm=200, seed=0) -> Stat:
    """The permutation floor of one readout rule, standalone.

    WHAT: recomputes, per section, the observed readout ('all' = median over
    genes, or a test-selected top-N) and its permutation null, then funnels
    both through _combine exactly as audit() does [L55][L39].
    WHY: a reviewer who only asks "what would this exact readout rule give
    on shuffled predictions" should not need a full audit to find out.
    WHERE: y_true/y_pred/section get the same coercions as [L01]-[L03];
    core.perm_plan supplies the shared plan [L19]; levers.topn_value /
    topn_null implement the selection rule [L33][L45].
    """
    Y = _mat(y_true, "y_true")
    P = _mat(y_pred, "y_pred")
    sec = _vec(section, "section", Y.shape[0]).astype(str)
    rng = np.random.default_rng(seed)
    vals, perms, nsp = [], [], []
    for s in sorted(pd.unique(sec).tolist()):
        m = sec == s
        y, yh = Y[m], P[m]
        # [L19] (B, n) permuted spot orders, coordinate-aware when coords
        # exist; the same plan is reused for every draw of this section
        pidx, pmask = perm_plan(int(m.sum()), n_perm, kind,
                                None if coords is None else np.asarray(coords)[m],
                                coord_kind, rng)
        pr = _perm_pergene(y, yh, pidx, pmask)  # (B, g) per-gene r under permutation
        r = colcorr(y, yh)                      # (g,) observed per-gene Pearson r [L25]
        if rule == "all":
            vals.append(agg(r, "median")[0])
            perms.append(np.array([np.nanmedian(x) for x in pr]))
        elif rule == "top_n":
            if top_n is None:
                raise BadInput("rule='top_n' needs top_n=<int>")
            vals.append(_lev.topn_value(r, top_n)[0])
            perms.append(_lev.topn_null(pr, top_n))
        else:
            raise BadInput("unknown rule %r" % rule)
        nsp.append(int(m.sum()))
    # the floor label must name the rule that generated it [L40]: a top-N
    # floor is 'perm_selection' regardless of block/free
    kindname = "perm_selection" if rule == "top_n" else (
        "perm_block" if kind == "block" and coords is not None else "perm_free")
    return _combine(vals, perms, nsp, [Y.shape[1]] * len(vals), kindname, rng)


def compare(*reps, on="headline") -> pd.DataFrame:
    """Put several audits side by side, or refuse to.

    WHAT: builds one row per report (headline, floor, strongest null, win
    fraction) and raises NotComparable when the runs differ in space,
    aggregation, panel size or section count [L56].
    WHY: numbers computed in different target spaces or on different panels
    are not on a common scale; refusing is the honest output, and the space
    field exists precisely for this check [L12].
    WHERE: reads only RunMeta and Headline fields of finished AuditReports;
    computes nothing new.
    """
    from .core import NotComparable
    if len(reps) < 2:
        raise BadInput("compare() needs at least two reports")
    # the comparability key: target space [L12], both aggregation rules,
    # panel size, number of sections; any mismatch is a refusal, not a fudge
    keys = {(r.run.space.value, r.run.agg_gene, r.run.agg_section, r.run.n_genes,
             r.run.n_sections) for r in reps}
    if len(keys) > 1:
        raise NotComparable(
            "these reports are not comparable: space/aggregation/panel/split "
            "differ -> %s. stnull does not offer an approximate comparison."
            % sorted(keys))
    rows = []
    for r in reps:
        h = r.headline
        rows.append({"space": r.run.space.value, "sections": r.run.n_sections,
                     "genes": r.run.n_genes,
                     "r_median": h.r_median.value if h else np.nan,
                     "floor": h.r_median.floor if h else np.nan,
                     "strongest_null": h.strongest_null[0] if h and h.strongest_null
                     else "n/a",
                     "null_value": h.strongest_null[1].value if h and h.strongest_null
                     else np.nan,
                     "beats_null_frac": h.frac_sections_model_beats_null.value
                     if h else np.nan})
    return pd.DataFrame(rows)
