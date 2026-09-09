# -*- coding: utf-8 -*-
"""stnull.core: numerical primitives shared by every audit section.

Author: Zigan Wang.

WHAT THIS MODULE IS
    The bottom layer of the package.  It owns four things and nothing else:
    the ``Stat`` container (a number plus the null floor of its own readout
    rule), the vectorised linear algebra used by every readout (column-wise
    Pearson r, SVD orthonormal bases, residualisation, partial r), the
    resampling plans (K-fold, permutation, bootstrap), and the exception
    hierarchy.

WHO CALLS IT
    Everything.  ``nulls`` (fitting the zero-pixel nulls), ``drg`` (the
    attribution ladder), ``ceiling`` (split-half reliability), ``levers``
    (top-N selection) and ``audit`` (the orchestrator) all import from here.
    Nothing in this file imports from them, so the dependency graph is a tree
    rooted at this file.

PLAIN-LANGUAGE TERMS (first-use definitions for readers new to the field)
    * spot: one capture location on a spatial transcriptomics slide, a tiny
      circle of tissue whose RNA molecules are counted.  Rows of every
      matrix in this file are spots.
    * gene: one measured transcript.  Columns of every matrix are genes.
    * section: one tissue slice on the slide.  This module never sees a
      whole run, only one section at a time (the caller slices first).
    * Pearson r: a number in [-1, 1] measuring how well two lists of values
      move together (1 = perfectly together, 0 = no linear relation).
    * permutation floor: the score the same readout rule would produce by
      chance, estimated by shuffling which spot gets which prediction.
    * bootstrap: re-drawing the spots (or sections) with replacement many
      times to see how much a number wobbles, which gives the CI.
    * K-fold cross-fit: split the spots into K parts, fit on K-1 of them,
      predict the held-out part, rotate; every spot gets an out-of-fold
      prediction.

DATA LINEAGE (row numbers refer to docs/CODE_WALKTHROUGH.md (section 4, the variable lineage table))
    Inputs to this module are always ALREADY-SLICED per-section arrays:
    the (n, g) truth/prediction pairs of [L01]/[L02] arrive here as the
    ``A``/``B``/``y``/``yh``/``M`` arguments of :func:`colcorr`,
    :func:`partial_r` and :func:`resid`; the ladder and null design
    matrices of [L32] and [L07] arrive as the ``X`` of :func:`ortho_basis`.
    Outputs leave along four paths:
    * per-gene r from :func:`colcorr` starts the [L25] chain
      (nulls.py:_score and drg.py:_pr call it), is collapsed by
      :func:`agg`, and ends life inside a :func:`make_stat` call;
    * :func:`perm_plan` produces the shared ``pidx``/``pmask`` plan of
      [L19] (born at audit.py:374, rebuilt at audit.py:504) that every
      permutation floor in audit.py and drg.py reuses;
    * :func:`kfold_indices` produces the shared ``folds`` of [L20]
      (audit.py:382) that every null in nulls.py cross-fits on;
    * :func:`boot_ci` produces the ``(lo, hi)`` intervals of [L34], and
      :func:`make_stat` fuses value + floor + CI into the Stat triple of
      [L39]/[L40] that report.py renders (Stat.__str__, Stat.to_dict,
      report.py:_stat_cells).
    :func:`sha1_of` fingerprints raw user inputs into ``inputs_sha1``
    [L51] for the reproducibility card.

KEY ASSUMPTIONS
    * Everything is vectorised over genes: a (n_spot, n_gene) matrix goes in,
      a (n_gene,) vector of Pearson r comes out.  There is no per-gene Python
      loop anywhere in this file.
    * Spots are the sampling unit within a section; genes are the aggregation
      unit within a readout.  Nothing here knows about sections, the caller
      slices the section out before calling in.
    * Pearson r is the only similarity measure, because it is the metric the
      field actually reports; the package audits that metric rather than
      substituting a better-behaved one.

DESIGN RULES ENFORCED HERE (see DECISIONS in the README)
    * ``Stat`` is the only number container in the package, and a value is
      never exposed without its permutation floor / CI / n / status.
    * Zero-variance genes, empty groups and rank-deficient designs return NaN
      and are COUNTED (``n_genes`` of a Stat is the number of genes that
      actually produced a finite r), never silently dropped.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

#: Absolute tolerance for declaring a sum-of-squares numerically zero.  It is
#: deliberately tiny: the alternative (a relative test) needs a scale per
#: column, and the columns this package sees (log1p-CP10K expression) live at
#: O(0.1-5), where 1e-12 is unambiguously "constant".
EPS = 1e-12

__all__ = [
    "Stat", "na_stat", "make_stat", "colcorr", "ortho_basis", "resid",
    "partial_r", "agg", "perm_plan", "boot_ci", "sha1_of", "kfold_indices",
    "StnullError", "MissingSpace", "NotComparable", "LibSizeLooksLikePanelSum",
    "InsufficientData", "BadInput",
]


# --------------------------------------------------------------- exceptions
# One base class, so a caller can wrap the package in a single except clause;
# one subclass per failure the user can act on, because "which input do I
# fix?" is the only question an audit tool's errors need to answer.
class StnullError(Exception):
    """Base class for every error raised by stnull."""


class MissingSpace(StnullError):
    """``space=`` was absent or unrecognised.  It has no default by design."""
    # Raised only by spaces.coerce_space (spaces.py:59), which audit()
    # calls before touching any data ([L12]).


class NotComparable(StnullError):
    """Two audits differ in space / panel / split, so they may not be tabled."""
    # Raised only by audit.compare (audit.py:1556), whose comparability key
    # is built from RunMeta fields including the [L12] space declaration.


class LibSizeLooksLikePanelSum(StnullError):
    """``lib_size`` tracks the panel row sum, not the full-transcriptome total."""
    # The trap behind this error: if lib_size ([L04]) was computed by summing
    # only the audited gene panel, the depth null is fed a circular predictor.
    # Detected in audit.check_inputs (audit.py:224-236).


class InsufficientData(StnullError):
    """Not enough spots / sections / genes to compute what was requested."""


class BadInput(StnullError):
    """A structural problem in the arguments (shape, dtype, unknown option)."""


# ---------------------------------------------------------------- the number
@dataclass(frozen=True)
class Stat:
    """A number that cannot be quoted without its null floor.

    Lineage [L40]: instances are built ONLY by :func:`make_stat` or
    :func:`na_stat`.  Nearly all of them are born inside audit.py's
    ``_combine`` funnel ([L39], audit.py:625), which pours per-section
    scalars and per-section permutation draws into :func:`make_stat`.
    They are consumed by report.py: the console line via ``__str__``
    (report.py:summary), the JSON via :meth:`to_dict`
    (report.py:_sec_json), and the HTML cells via report.py:_stat_cells
    and _stat_table (where the floor column cannot be switched off).

    Attributes
    ----------
    value : float
        The readout itself, on the Pearson-r scale unless the constructing
        call says otherwise (shares and fractions are dimensionless).
    floor : float or None
        The 95th percentile of the permutation null distribution of the SAME
        readout rule (not the mean of it): a one-sided 5% threshold.  ``None``
        means no floor could be computed, and ``floor_kind`` says why.
    floor_kind : str
        How the floor was obtained: ``perm_block`` (torus shift, preserves
        spatial autocorrelation), ``perm_free`` (free permutation,
        anticonservative on autocorrelated data), ``perm_selection`` (the same
        selection rule applied to permuted predictions), ``analytic_zero``
        (the null value is 0 by construction), ``derived`` (a ratio of two
        floored numbers) or ``none``.
    ci : (float, float) or None
        Percentile 95% bootstrap interval.  Section-level where there are >= 3
        sections, spot-level otherwise; the note records which.
    n_spots : int
        Spots that entered the readout (summed over sections for aggregates).
    n_genes : int
        Genes that produced a FINITE r in this readout, not the panel size.
    status : str
        ``ok`` | ``degraded`` (computed, but carries a caveat) | ``na`` (not
        computable) | ``derived`` (a function of other Stats).
    note : str
        Free text; carries the median and max of the null distribution so the
        reader sees the whole null, not only the 95th-percentile threshold.

    Statistical premise
        ``floor`` is interpretable only against a value computed by the same
        rule on the same spots.  Comparing a value with another readout's
        floor is exactly the error this class exists to prevent.
    """

    value: float                       # the observed readout (usually a Pearson r)
    floor: Optional[float]             # 95th pct of the matching permutation null, or None
    floor_kind: str                    # how the floor was built (see class docstring)
    ci: Optional[Tuple[float, float]]  # (lo, hi) percentile bootstrap interval, or None
    n_spots: int                       # spots that entered the readout
    n_genes: int                       # genes with a FINITE r, not panel size
    status: str                        # 'ok' | 'degraded' | 'na' | 'derived'
    note: str = ""                     # free text incl. null median/max

    def excess(self) -> float:
        """value - floor, or NaN when either is missing.  Not a p-value."""
        # WHAT: the margin by which the value clears its own chance level.
        # WHY: readers want one number for "how much beyond chance"; the
        # difference on the r scale is that number.  It is NOT a p-value and
        # is never presented as one.
        # WHERE: serialised into to_dict() below, so it reaches the JSON view
        # that report.py:_sec_json writes.
        if self.floor is None or not np.isfinite(self.value):
            return float("nan")
        return float(self.value - self.floor)

    def above_floor(self) -> Optional[bool]:
        """Is this value distinguishable from its own null?

        Uses the bootstrap CI lower bound when one exists (the stricter test:
        the whole interval must clear the floor) and the point estimate
        otherwise.  Returns None when the question cannot be asked at all.
        """
        # WHAT: a three-valued verdict (True / False / None = unanswerable).
        # WHY the CI lower bound when present: demanding that the WHOLE
        # bootstrap interval sits above the floor is stricter than comparing
        # the point estimate, so a True from this path is the safer claim.
        # WHERE: report.py's claim wording (report.py:225-324) and the
        # ladder's drg_above_floor column (audit.py:1095-1097) both hang
        # their pass/fail language on this method.
        if self.floor is None or not np.isfinite(self.value):
            return None
        if self.ci is not None and np.isfinite(self.ci[0]):
            return bool(self.ci[0] > self.floor)
        return bool(self.value > self.floor)

    def _fmt(self, x) -> str:
        # WHAT: one float to text, with None/NaN/inf all printed as "n/a".
        # WHY: the console line must never show "nan", which readers skip
        # over; "n/a" forces the eye to the note explaining why.
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return "n/a"
        return "%.4f" % x

    def __float__(self) -> float:
        # Convenience so numpy/max() can order Stats by value, e.g. the
        # strongest-null pick at audit.py:1127-1129 ([L41]).
        return float(self.value)

    def __str__(self) -> str:
        # value, floor, CI, n and status print together on one line.  There is
        # deliberately no short form: a bare number is the failure mode this
        # package exists to stop.
        # WHERE: this is the console renderer of [L40]; report.py:summary
        # interpolates the returned string directly into its text report.
        floor = ("[%s floor %s]" % (self.floor_kind, self._fmt(self.floor))
                 if self.floor is not None else "[no floor]")
        ci = ("CI[%s,%s]" % (self._fmt(self.ci[0]), self._fmt(self.ci[1]))
              if self.ci is not None else "CI[n/a]")
        s = "%s  %s  %s  n=%d/%d" % (self._fmt(self.value), floor, ci,
                                     self.n_spots, self.n_genes)
        if self.status != "ok":
            s += "  <%s>" % self.status
        if self.note:
            s += "  (%s)" % self.note
        return s

    __repr__ = __str__

    def to_dict(self) -> dict:
        """JSON-safe view; non-finite floats become ``None``, never strings."""
        # WHAT: flatten the triple into plain JSON types.
        # WHY None instead of "NaN": json.dumps writes bare NaN tokens that
        # strict parsers reject; None round-trips everywhere.
        # WHERE: report.py:_sec_json calls this for every Stat when writing
        # the machine-readable run card ([L40]).
        return dict(value=_j(self.value), floor=_j(self.floor),
                    floor_kind=self.floor_kind,
                    ci_lo=_j(self.ci[0]) if self.ci else None,
                    ci_hi=_j(self.ci[1]) if self.ci else None,
                    n_spots=int(self.n_spots), n_genes=int(self.n_genes),
                    status=self.status, note=self.note,
                    excess=_j(self.excess()))


def _j(x):
    """float -> JSON-safe float (NaN and +/-inf become None)."""
    if x is None:
        return None
    x = float(x)
    return x if np.isfinite(x) else None


def na_stat(reason: str, n_spots: int = 0, n_genes: int = 0,
            status: str = "na") -> Stat:
    """A Stat that says why it has no value.

    WHERE: audit.py reaches for this whenever a whole readout is off the
    table (no coords, no counts, too few spots); the ``reason`` string is
    what report.py prints on the N/A line, and often echoes an entry of the
    ``na`` ledger [L50].

    Parameters
    ----------
    reason : str
        Human-readable explanation; it is printed in the report next to the
        N/A so the reader never has to guess whether a number was suppressed
        or was never computable.
    n_spots, n_genes : int
        Whatever was known at the point of failure; 0 when nothing was.
    status : str
        ``na`` by default; pass ``degraded`` for "computed but do not quote".
    """
    return Stat(float("nan"), None, "none", None, int(n_spots), int(n_genes),
                status, reason)


def make_stat(value, perm_dist=None, ci=None, n_spots=0, n_genes=0,
              floor_kind="perm_free", status="ok", note="") -> Stat:
    """Bundle a readout with the null distribution of the SAME readout rule.

    Lineage: this is the terminal stage of the [L25] chain (per-gene r ->
    section scalar -> across-section scalar -> Stat) and the factory of the
    [L39]/[L40] triple.  Its main caller is audit.py's ``_combine``
    (audit.py:625), which passes value = nanmedian over sections and
    perm_dist = per-draw across-section medians; direct calls also exist for
    the leakage lever (audit.py:1316) and perm_floor (audit.py:1524-1553).

    Parameters
    ----------
    value : float
        The observed readout.
    perm_dist : array-like or None
        Draws of the SAME readout rule under the null (permuted spot order).
        The caller is responsible for that identity; this function cannot
        check it, and getting it wrong is the most damaging mistake available
        here: a floor computed from another rule is not a floor.
    ci : (float, float) or None
        Bootstrap interval, already computed by the caller.
    n_spots, n_genes : int
        Sample sizes recorded on the Stat.
    floor_kind : str
        Label for the null construction; downgraded to ``none`` when no usable
        permutation distribution arrives.
    status, note : str
        Passed through, with the null's median/max appended to the note.

    Returns
    -------
    Stat

    Not applicable when
        ``perm_dist`` holds fewer than 5 finite draws (the 95th percentile of
        four numbers is not a threshold): the Stat comes back with
        ``floor_kind='none'`` and ``status='degraded'``.
    """
    # --- 1. Decide the floor from the permutation draws -------------------
    # WHAT: turn the (B,) vector of null draws into a single threshold.
    # WHY the 95th percentile: a one-sided 5% chance level, matching how the
    # field asks "is this better than shuffling?".
    # WHERE perm_dist comes from: audit.py's _combine (per-draw section
    # medians of [L26]/[L33]-style permuted readouts), drg.perm_null_of_rung
    # (drg.py:225) or levers.topn_null (levers.py:64).
    floor = None                        # None until >= 5 finite draws arrive
    if perm_dist is not None:
        pdist = np.asarray(perm_dist, dtype=float)  # (B,) null draws of this rule
        if pdist.size == 0:
            # n_perm=0: the caller asked for no permutations at all.  The value
            # is computed but cannot be audited, so the Stat is marked 'na'.
            msg = "no permutations were run (n_perm=0): value has no floor"
            return Stat(float(value), None, "none",
                        (float(ci[0]), float(ci[1])) if ci is not None else None,
                        int(n_spots), int(n_genes),
                        "na" if status == "ok" else status,
                        (note + "; " + msg) if note else msg)
        # Drop non-finite draws (a permuted readout can be NaN when every
        # gene became unscorable in that draw); only finite draws can carry
        # a percentile.
        pdist = pdist[np.isfinite(pdist)]
        if pdist.size >= 5:
            # 95th percentile = a one-sided 5% threshold.  The median and max
            # go into the note because a reader who sees only the threshold
            # cannot tell a tight null from a wild one.
            floor = float(np.percentile(pdist, 95))  # the null floor of [L40]
            extra = "perm med=%.4f max=%.4f nperm=%d" % (
                float(np.median(pdist)), float(pdist.max()), pdist.size)
            note = (note + "; " + extra) if note else extra
        else:
            # --- 2. Too few finite draws: degrade, never fake a floor -----
            # WHY: the 95th percentile of < 5 numbers is dominated by one
            # draw; publishing it as a threshold would be noise dressed as a
            # test.  The Stat survives but is stamped 'degraded'.
            floor_kind = "none"
            status = "degraded" if status == "ok" else status
            msg = "permutation null unavailable"
            note = (note + "; " + msg) if note else msg
    else:
        # --- 3. No null was offered at all ---------------------------------
        # WHY: a floor_kind that still says 'perm_*' with no draws behind it
        # would misdescribe the Stat; only non-permutation kinds (e.g.
        # 'analytic_zero', 'derived') may survive a missing perm_dist.
        if floor_kind.startswith("perm"):
            floor_kind = "none"
    # --- 4. Assemble the frozen triple ------------------------------------
    # A non-finite value with an unremarkable status is promoted to 'na' so
    # that report.py prints "n/a" rather than an ok-looking NaN.
    if not np.isfinite(value) and status == "ok":
        status = "na"
    return Stat(float(value), floor, floor_kind,
                (float(ci[0]), float(ci[1])) if ci is not None else None,
                int(n_spots), int(n_genes), status, note)


# ------------------------------------------------------------- linear algebra
def colcorr(A, B) -> np.ndarray:
    """Column-wise Pearson r between two equally shaped matrices.

    Lineage: the atom of the [L25] chain.  Callers pair a truth-side matrix
    with a prediction-side matrix: nulls.py:_score (truth vs null
    prediction), drg.py:_pr (residualised truth vs residualised prediction,
    via :func:`partial_r`), audit.py's per-gene floors (_perm_pergene,
    audit.py:612), the train-side ranking key [L24] (audit.py:495/552), and
    the smoothing lever [L29] (audit.py:567).

    Parameters
    ----------
    A, B : (n, g) or (n,) array-like
        Columns are paired by position: column j of A against column j of B.
        1-D input is treated as a single column.

    Returns
    -------
    (g,) ndarray of r in [-1, 1]; NaN where either column is constant.

    Statistical premise
        Pearson r on the values as given: no transformation, no ranking, no
        winsorising.

    Not applicable when
        A column has zero variance (r is undefined, not zero): the entry is
        NaN so that ``agg`` counts it as unscored instead of averaging in a
        fabricated 0.
    """
    # --- 1. Coerce to float64 (n, g) ---------------------------------------
    # WHY float64: the centred cross-products below cancel; float32 loses
    # digits exactly where near-constant genes need them.
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    if A.ndim == 1:
        A = A[:, None]
    if B.ndim == 1:
        B = B[:, None]
    # --- 2. Centre each column, then form the Pearson ratio ----------------
    # WHAT: r_j = sum(a_j * b_j) / (||a_j|| * ||b_j||) on centred columns,
    # computed for all g columns at once (no per-gene Python loop).
    A = A - A.mean(0, keepdims=True)   # (n, g) column-centred truth side
    B = B - B.mean(0, keepdims=True)   # (n, g) column-centred prediction side
    num = (A * B).sum(0)               # (g,) covariance numerators (x n)
    den = np.sqrt((A * A).sum(0)) * np.sqrt((B * B).sum(0))  # (g,) sd products (x n)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = num / den                  # (g,) per-gene Pearson r, may hold NaN/inf
    # --- 3. Constant columns become NaN, not 0 -----------------------------
    # WHY: r of a zero-variance gene is undefined; writing 0 would drag the
    # median of [L25] toward zero and silently reward dead genes.
    r = np.where(den <= EPS, np.nan, r)
    # Clipping only removes float round-off past +/-1; a genuine |r| > 1 is
    # impossible, so this cannot mask a defect in the caller's data.
    return np.clip(r, -1.0, 1.0)


def ortho_basis(X) -> np.ndarray:
    """Orthonormal basis of col(X) via SVD, rank-deficiency safe.

    Lineage [L32]: the designs it receives are the ladder's X1/X2a/X2
    (built in drg.ladder_rungs, drg.py:196-208, from lhat [L21] and the
    one-hot Comp block [L07]), the observed-DRG design Xo (audit.py:538),
    and the leakage design Xl (audit.py:1310).  A one-hot block plus an
    intercept is collinear BY CONSTRUCTION (the K class columns sum to the
    intercept), and the SVD threshold below is what absorbs that
    redundancy, per the note on [L07].

    Parameters
    ----------
    X : (n, q) array-like
        Design matrix, typically ``[1, l_hat, pi]``.  Columns may be collinear
        (a one-hot block plus an intercept always is).

    Returns
    -------
    (n, rank) ndarray with orthonormal columns; (n, 0) when X has no columns.

    Statistical premise
        Residualising on Q equals residualising on X, but without forming a
        normal-equation inverse.  Singular values below 1e-10 of the largest
        are dropped, so a rank-deficient design loses its redundant directions
        instead of producing an unstable solve.
    """
    # --- 1. Normalise the input to a 2-D design ---------------------------
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    if X.shape[1] == 0:
        # Zero columns: an empty basis, so resid() falls back to centring.
        return np.zeros((X.shape[0], 0))
    # --- 2. Thin SVD, keep only well-conditioned directions ---------------
    # WHAT: X = U s V^T; the kept columns of U span col(X) orthonormally.
    # WHY 1e-10 relative: directions this small are numerical shadows of the
    # collinearity above, and keeping them makes the projection in resid()
    # amplify round-off instead of removing structure.
    U, s, _ = np.linalg.svd(X, full_matrices=False)
    if s.size == 0:
        return np.zeros((X.shape[0], 0))
    keep = s > (s.max() * 1e-10)       # (q,) bool, rank mask
    return U[:, keep]                  # (n, rank) orthonormal basis of col(X)


def resid(M, Q) -> np.ndarray:
    """Residual of every column of M after projection onto the basis Q.

    ``Q`` must be orthonormal (use :func:`ortho_basis`).  With ``Q=None`` or an
    empty basis this reduces to column centring, which is what a partial
    correlation on an intercept-only design amounts to.

    WHERE: called by :func:`partial_r` below and by drg.py's rung machinery
    (perm_null_of_rung drg.py:225, boot_of_rung drg.py:270), always with a
    ``Q`` freshly built by :func:`ortho_basis` from a [L32] design.
    """
    M = np.asarray(M, dtype=np.float64)
    if Q is None or Q.shape[1] == 0:
        # WHY centring here: correlations are centred anyway, so an
        # intercept-only control changes nothing, and returning centred M
        # keeps the two code paths numerically comparable.
        return M - M.mean(0, keepdims=True)
    # Projection residual: M minus its best reconstruction from Q.  Because
    # Q is orthonormal, Q @ (Q.T @ M) IS the least-squares fit, no inverse
    # needed and no instability on rank-deficient inputs.
    return M - Q @ (Q.T @ M)


def partial_r(y, yh, X) -> np.ndarray:
    """Per-gene Pearson r after residualising BOTH y and yh on X.

    Lineage: this is the mathematical core of the attribution ladder [L31];
    drg.py:_pr (drg.py:216) is a thin wrapper around the same two calls, and
    the observed-DRG rung (audit.py:541) reaches it through that wrapper.
    ``X`` is one of the [L32]/[L35] designs; ``y``/``yh`` are the section
    slices of [L01]/[L02].

    Statistical premise
        Both sides are residualised, not only the prediction: the question is
        "do y and y_hat still agree in the directions X cannot explain?".
        Residualising one side alone answers a different question and leaves
        the control's variance sitting inside the other side.
    """
    # One basis, two residualisations, one column-wise correlation: the
    # textbook partial correlation, vectorised over genes.
    Q = ortho_basis(X)                 # (n, rank) control subspace
    return colcorr(resid(y, Q), resid(yh, Q))


def agg(r, how="median") -> Tuple[float, int]:
    """Aggregate a per-gene r vector.

    Lineage [L25]: the second stage of the chain.  Every per-gene vector
    that :func:`colcorr` produces inside a section (null r at audit.py:430,
    rung r at audit.py:515-524, smoothing r at audit.py:567) passes through
    here to become the section scalar that audit.py's ``col()`` accessor
    later collects across sections.

    Parameters
    ----------
    r : (g,) array-like
        Per-gene correlations, NaN allowed.
    how : {'median', 'mean'}
        The literature mixes the two, so the package reports both rather than
        picking a winner.

    Returns
    -------
    (value, n_finite_genes): the count is what a Stat's ``n_genes`` records,
    so a panel where half the genes were unscorable cannot masquerade as a
    full-panel result.
    """
    r = np.asarray(r, dtype=float)     # (g,) per-gene r, NaN = unscorable gene
    n = int(np.isfinite(r).sum())      # genes that actually produced a number
    if n == 0:
        # All genes unscorable: the section contributes NaN upward, and the
        # zero count is what lets _combine and make_stat see the hole.
        return float("nan"), 0
    with np.errstate(invalid="ignore"):
        v = float(np.nanmedian(r)) if how == "median" else float(np.nanmean(r))
    return v, n


def kfold_indices(n: int, k: int, rng) -> list:
    """Random K-fold split of ``n`` spots; returns [(train_idx, test_idx), ...].

    Lineage [L20]: called ONCE per section (audit.py:382, with k=5 and its
    own seeded Generator) and the resulting ``folds`` list is handed to
    every null in nulls.py (linear_null_pred cross-fit branch,
    nulls.py:157-163).  Sharing one partition is deliberate: differences
    between nulls then measure the nulls, not the luck of separate splits.

    ``k`` is clamped to [2, n].  Spots are permuted first, so the folds carry
    no spatial structure, which is precisely why a within-section cross-fit
    is slightly optimistic on autocorrelated data, and why the degradation
    ledger says so whenever ``null_fit='cv_within_section'`` is used.
    """
    # --- 1. Clamp k and shuffle the spot order ----------------------------
    # WHY clamp: k > n would create empty folds; k < 2 is not a split.
    k = max(2, min(int(k), n))
    order = rng.permutation(n)         # (n,) random spot order, breaks spatial runs
    # --- 2. Cut the shuffled order into k nearly equal folds --------------
    folds = np.array_split(order, k)   # list of k index arrays
    out = []
    for i in range(k):
        # Fold i is the held-out test part; the other k-1 folds concatenate
        # into its training part.  Every spot appears in exactly one test set,
        # so the nulls' out-of-fold predictions cover the whole section.
        te = folds[i]
        tr = np.concatenate([folds[j] for j in range(k) if j != i])
        out.append((tr, te))
    return out


# ------------------------------------------------------------- permutations
def perm_plan(n: int, n_perm: int, kind: str, coords=None,
              coord_kind="grid", rng=None):
    """Build ``n_perm`` spot-order permutations.

    Lineage [L19]: audit.py draws this plan once per section (audit.py:374,
    with rng_perm from the [L18] stream split), rebuilds it on the held-out
    subset after the ``ev`` mask (audit.py:504), and reuses the SAME
    ``(idx, mask)`` pair in every floor: _perm_of_pred, _perm_pergene,
    drg.perm_null_of_rung and the Moran loop.  The coords argument is the
    section slice of [L06]; audit.py also calls this with kind='free' for
    the deliberately anticonservative contrast floor [L37]
    (audit.py:582-583) and inside perm_floor [L55].

    Plain language: a "permutation" here is a reshuffled assignment of
    predictions to spots.  The 'block' flavour slides the whole tissue grid
    like a conveyor belt (a torus shift), so neighbouring spots stay
    neighbours and the spatial smoothness of the data survives into the
    null; the 'free' flavour scrambles spots individually and destroys that
    smoothness.

    Parameters
    ----------
    n : int
        Spots in the section.
    n_perm : int
        Number of draws; 0 returns an empty plan (never an exception).
    kind : {'block', 'free'}
        ``block`` = torus shift on the array lattice, which PRESERVES spatial
        autocorrelation; ``free`` = uniform permutation, which does not and is
        therefore an ANTICONSERVATIVE floor on autocorrelated data.
    coords : (n, 2) array-like or None
        Required for ``block``; ignored for ``free``.
    coord_kind : {'grid', 'micron'}
        ``micron`` coordinates are divided by an estimated pitch first.
    rng : numpy Generator or None

    Returns
    -------
    (idx, mask) : ``idx`` is (n_perm, n) int; ``mask`` is (n_perm, n) bool
    marking usable spots, or None when every spot is usable (the free case).

    Not applicable when
        The lattice is 1x1, grid cells are not unique, coordinates are not
        finite, or the torus embedding covers under half the spots: the
        function falls back to free permutation, and ``audit()`` books that
        fallback in its degradation ledger.
    """
    # --- 1. Guards: default rng, and the n_perm=0 escape hatch ------------
    rng = np.random.default_rng(0) if rng is None else rng
    if n_perm <= 0:
        # no permutations requested: an empty plan, never a crash.  Every
        # downstream Stat built from it carries status='na' and no floor.
        return np.zeros((0, n), dtype=np.int64), None
    # --- 2. Block (torus-shift) path: only with usable coordinates --------
    # WHAT: embed the spots in an integer W x H lattice, then shift the whole
    # lattice by a random (dx, dy) with wrap-around.
    # WHY: a shift moves every spot the same way, so local neighbourhoods
    # (and hence spatial autocorrelation) are preserved in the null; that is
    # what makes the resulting floor conservative rather than flattering.
    if kind == "block" and coords is not None and n > 4:
        g = _to_grid(np.asarray(coords, dtype=float), coord_kind)  # (n, 2) lattice cells
        # Non-finite coordinates would cast to INT64_MIN below and then make
        # W/H overflow silently, so the "shift" would not be a permutation at
        # all.  Fall through to free permutation instead; check_inputs reports
        # the non-finite coordinates as an error in the first place.
        if np.isfinite(g).all():
            gx = g[:, 0].astype(np.int64)      # (n,) integer column index per spot
            gy = g[:, 1].astype(np.int64)      # (n,) integer row index per spot
            x0, y0 = int(gx.min()), int(gy.min())
            W = int(gx.max()) - x0 + 1         # lattice width in cells
            H = int(gy.max()) - y0 + 1         # lattice height in cells
            # (W > 1 or H > 1) is required: a 1x1 grid cannot be torus-shifted.
            # Duplicate grid cells would make the LUT write below keep only the
            # last spot per cell, so the "shift" would map several sources onto
            # one target, not a permutation.  Fall through to free instead
            # (audit() discloses this per section in its degradation ledger).
            n_cells = len(np.unique(np.stack([gx, gy], 1), axis=0))
            if W * H <= 4_000_000 and (W > 1 or H > 1) and n_cells == n:
                # --- 2a. Cell -> spot lookup table ------------------------
                # lut[x, y] = spot index occupying that cell, -1 = empty
                # tissue.  The 4M-cell cap above bounds this allocation.
                lut = -np.ones((W, H), dtype=np.int64)
                lut[gx - x0, gy - y0] = np.arange(n)
                idx = np.zeros((n_perm, n), dtype=np.int64)  # (B, n) permuted spot ids
                msk = np.zeros((n_perm, n), dtype=bool)      # (B, n) landed-on-tissue flags
                for b in range(n_perm):
                    # (dx, dy) != (0, 0) always: a zero shift is the identity
                    # permutation, which would deflate the floor.  When W == 1
                    # (single-column grid) the shift must come from dy.
                    if W > 1:
                        dx = int(rng.integers(1, W))
                        dy = int(rng.integers(0, H)) if H > 1 else 0
                    else:
                        dx = 0
                        dy = int(rng.integers(1, H))
                    # --- 2b. Apply the wrapped shift ----------------------
                    # tgt[i] = the spot now sitting where spot i used to sit;
                    # tgt < 0 means the shifted cell fell on empty tissue.
                    tgt = lut[(gx - x0 + dx) % W, (gy - y0 + dy) % H]
                    ok = tgt >= 0
                    idx[b] = np.where(ok, tgt, 0)
                    msk[b] = ok
                # A torus shift over a ragged tissue outline sends many spots
                # onto empty cells.  Below 50% coverage the block null would be
                # computed on a small, oddly shaped subset, so free permutation
                # (disclosed in the ledger) is the lesser evil.
                if msk.mean() >= 0.5:
                    return idx, msk
    # --- 3. Free fallback: independent uniform shuffles -------------------
    # Reached when kind='free' was requested, or when any block precondition
    # failed above.  mask=None means every spot is usable in every draw.
    idx = np.stack([rng.permutation(n) for _ in range(n_perm)])
    return idx, None


def _to_grid(coords: np.ndarray, coord_kind: str) -> np.ndarray:
    """Snap coordinates onto integer lattice cells.

    For ``micron`` input the pitch is estimated as the median nearest-neighbour
    distance, which is the spot-to-spot spacing of any regular capture area; a
    poor snap shows up as duplicate cells and forces the free fallback.

    WHERE: only :func:`perm_plan` calls this; ``coord_kind`` traces back to
    the ``coord_kind`` argument of ``audit()`` (possibly flipped by its
    lattice plausibility check, see [L06]).
    """
    if coord_kind == "grid":
        # Already lattice indices: rounding removes float noise only.
        return np.rint(coords)
    # Micron path: estimate the physical spacing between adjacent spots.
    # k=2 nearest neighbours because the first "neighbour" of a point is
    # itself at distance 0; column -1 is the true nearest other spot.
    from scipy.spatial import cKDTree
    t = cKDTree(coords)
    d, _ = t.query(coords, k=min(2, len(coords)))
    d = np.atleast_2d(d)
    pitch = float(np.median(d[:, -1]))  # microns per lattice step, median = outlier-safe
    pitch = pitch if pitch > 0 else 1.0
    return np.rint(coords / pitch)


def boot_ci(fn, n_items: int, n_boot: int, rng, groups=None):
    """Percentile 95% CI of ``fn(idx)`` under resampling with replacement.

    Lineage [L34]: audit.py wires this to the ladder rungs
    (``fn = lambda idx: drg.boot_of_rung(y, yh, X, idx)``, audit.py:532-533,
    with the rng_boot stream of [L18]), so the residualisation on the [L32]
    design is redone inside every draw; the returned (lo, hi) becomes
    ``out[k+"_ci"]`` and later a Stat's ``ci`` field, including the
    single-section fallback CI at audit.py:1107/1187/1238.

    Parameters
    ----------
    fn : callable(idx) -> float
        Recomputes the readout on the resampled rows.  It must redo any
        residualisation, otherwise the interval understates the uncertainty in
        the control rather than only in the correlation.
    n_items : int
        Number of resamplable units (spots, or sections at the outer level).
    n_boot : int
        Draws; <= 0 disables the interval.
    rng : numpy Generator
    groups : array-like or None
        When given, whole groups are resampled instead of single items, the
        right unit when items inside a group are dependent (spots within a
        patient, say).

    Returns
    -------
    ((lo, hi), note) or (None, note) with the reason in the note.

    Not applicable when
        Fewer than 3 items exist, ``n_boot < 10``, or fewer than 10 draws come
        back finite: a percentile interval from that many draws is noise.
    """
    # --- 1. Refusals that keep garbage intervals out of the report --------
    if n_boot <= 0:
        return None, "bootstrap disabled (n_boot=%d)" % int(n_boot)
    if n_items < 3:
        return None, "bootstrap needs >=3 items (got %d)" % int(n_items)
    # --- 2. Draw the bootstrap replicates ---------------------------------
    vals = np.empty(int(n_boot))       # (n_boot,) one readout per resample
    if groups is None:
        # Item bootstrap: each draw picks n_items rows with replacement and
        # re-runs the full readout on them.
        for b in range(int(n_boot)):
            vals[b] = fn(rng.integers(0, n_items, n_items))
        kind = "spot"
    else:
        # Group bootstrap: resample the GROUP labels, then take every item of
        # each drawn group, so within-group dependence is preserved.
        groups = np.asarray(groups)
        uniq = np.unique(groups)
        pos = {g: np.where(groups == g)[0] for g in uniq}  # group -> row indices
        for b in range(int(n_boot)):
            pick = rng.choice(uniq, size=len(uniq), replace=True)
            vals[b] = fn(np.concatenate([pos[g] for g in pick]))
        kind = "group"
    # --- 3. Percentile interval from the finite draws ---------------------
    # WHY the finite filter: a draw can return NaN (e.g. every gene constant
    # in that resample); percentiles of NaN-laced arrays are meaningless.
    vals = vals[np.isfinite(vals)]
    if vals.size < 10:
        if int(n_boot) < 10:
            return None, "bootstrap needs n_boot>=10 (got %d)" % int(n_boot)
        return None, "bootstrap degenerate (%d finite draws)" % vals.size
    # 2.5th and 97.5th percentiles bracket the central 95%; the note records
    # which resampling unit was used, and make_stat carries it into the Stat.
    return (float(np.percentile(vals, 2.5)),
            float(np.percentile(vals, 97.5))), "ci_kind=%s" % kind


def sha1_of(a) -> str:
    """Short content hash of an input array, for the reproducibility appendix.

    Lineage [L51]: audit.py calls this on the raw user inputs (Y, P, lib,
    labels, ...) at audit.py:1440-1443; the digests land in
    ``RunMeta.inputs_sha1`` and the JSON run card (report.py:437-439), so
    two reports can be checked for having audited the same arrays.

    Object arrays are stringified first so that label vectors hash stably.
    The digest is truncated to 12 hex characters: enough to notice that two
    runs had different inputs, not a cryptographic commitment.
    """
    if a is None:
        return ""
    arr = np.asarray(a)
    if arr.dtype == object:
        # WHY: hashing raw object pointers would change between runs; the
        # string round-trip pins the hash to the CONTENT of the labels.
        arr = np.asarray([str(x) for x in arr.ravel()], dtype="U")
    return hashlib.sha1(np.ascontiguousarray(arr).tobytes()).hexdigest()[:12]
