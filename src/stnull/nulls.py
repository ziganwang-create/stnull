# -*- coding: utf-8 -*-
"""stnull.nulls: zero-pixel null models and neighbourhood operators.

Author: Zigan Wang.

WHAT THIS MODULE IS
    The competitors.  Each null predicts spatial expression from information
    that contains NO pixels: sequencing depth, tissue composition, or the true
    expression of neighbouring spots.  The audit's central comparison is
    "how much of the reported r does a predictor with no image already buy?".

WHO CALLS IT
    ``audit._section_block`` calls every null once per section and hands the
    per-gene r vectors plus the null's prediction matrix to the permutation
    machinery.  ``neighbour_matrix`` / ``smooth_matrix`` / ``morans_I`` are
    also used by the spatial levers in the same function.

WHAT EACH FUNCTION RETURNS
    Every null returns ``(r_per_gene, prediction_matrix, scored_mask)``:
    the r vector (NaN for genes that could not be scored) for the headline,
    the predictions so the caller can permute them against the same readout
    rule, and the mask of spots whose prediction is out-of-sample.

THE NULLS
    depth        y ~ [1, log lib]                  (1 scalar per spot)
    composition  y ~ one-hot(label) / composition  (class-mean lookup)
    depth+comp   y ~ [1, log lib, composition]
    neighbour    y_hat = mean of the TRUE y of the spot's lattice neighbours

FIT MODES (``fit_mode``), i.e. where a null's own parameters come from
    'cv_within_section'  K-fold cross-fitting inside the section (default)
    'train_only'         fit on is_train spots, score on the rest
    'oracle'             fit on all spots of the section = UPPER BOUND

KEY ASSUMPTIONS
    * Least squares is the right estimator for these nulls: they are meant to
      be the SIMPLEST predictor a critic could build, not the strongest one.
      A tuned model here would answer a different question.
    * Sections are audited independently; nothing in this file pools spots
      across sections, because between-section variation in depth and
      composition is itself one of the confounders under audit.

PLAIN-LANGUAGE TERMS (first use here; full glossary in CODE_WALKTHROUGH.md)
    spot: one capture location on the tissue slide, holding a few cells;
        every row of the matrices in this file is one spot.
    UMI / library size: the total number of RNA molecules counted in a spot;
        deeper (higher-count) spots record more of every gene at once.
    section: one tissue slice; all fitting and scoring stays within a section.
    Pearson r: the standard linear correlation in [-1, 1], the field's
        headline metric; computed per gene down the spot axis by
        ``core.colcorr``.
    permutation floor: the r the same readout rule reaches after shuffling
        the spot order, i.e. what pure chance buys under this exact protocol.

DATA LINEAGE (row numbers [L##] refer to docs/CODE_WALKTHROUGH.md (section 4, the variable lineage table))
    Everything this module touches is a per-section slice made inside
    ``audit._section_block`` (audit.py:333):
      Y      [L01] user y_true, coerced at audit.py:794, sliced Y[m] at
             audit.py:988, arriving here as the ``Y`` argument of every null.
      loglib [L05] born ONCE at audit.py:805 as log(max(lib_size, 1));
             reaches depth_null via audit.py:428 and the joint null via
             audit.py:463 (its other consumers live in audit.py and drg.py).
      comp   [L07] one-hot labels (merged classes, audit.py:871-875) or user
             cell-type proportions, sliced at audit.py:990; reaches
             composition_null (audit.py:448/455) and the joint null.
      folds  [L20] ONE shared 5-fold partition per section (audit.py:382,
             built by core.kfold_indices), passed to every null below so a
             difference between two nulls measures the nulls, not two splits.
      coords [L06] feed neighbour_matrix (audit.py:471, rebuilt on the
             held-out subset at 510) and smooth_matrix (audit.py:565).
    Outputs, and where they go next:
      r_per_gene [L25] first return of each null; collapsed by ``core.agg``
             into a section scalar (audit.py:430-475), then col() ->
             _combine -> make_stat -> Stat, the only road to the report.
      pred, scored [L26] second and third returns; audit._perm_of_pred
             (audit.py:588) permutes ``pred`` against the same readout rule
             to build the null's own permutation floor.
      W      [L28] from neighbour_matrix; consumed by neighbour_null
             (audit.py:472) and morans_I (audit.py:573).
      S      [L29] from smooth_matrix; ``S @ y`` at audit.py:566 prices the
             "smooth the ground truth before scoring" lever.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .core import EPS, BadInput, colcorr, kfold_indices

#: The only fit modes that exist.  A typo used to fall through to
#: cross-fitting while the run card still printed the typo as the protocol.
FIT_MODES = ("cv_within_section", "train_only", "oracle")


# ------------------------------------------------------------------ fitting
def linear_null_pred(Y, X, fit_mode="cv_within_section", is_train=None,
                     rng=None, k=5, notes=None, folds=None):
    """Least-squares prediction of every gene from design ``X``.

    Parameters
    ----------
    Y : (n, g) ndarray
        Target expression of one section, in the user's declared space.
    X : (n, p) ndarray
        Zero-pixel design, intercept column included by the caller.
    fit_mode : {'cv_within_section', 'train_only', 'oracle'}
        Where the coefficients come from; see the module docstring.
    is_train : (n,) bool or None
        Required for ``train_only``.
    rng : numpy Generator or None
        Only used by ``cv_within_section``, and only when ``folds`` is None.
    k : int
        Number of cross-fitting folds.
    folds : list of (train_idx, test_idx) or None
        A partition to reuse.  ``audit`` draws ONE partition per section and
        passes it to every null, so that a difference between two nulls
        measures the nulls rather than two different random splits, and so
        that dropping one null cannot move the others.
    notes : list or None
        Optional sink for one human-readable line explaining a bail-out.  The
        caller books it in the report's N/A ledger, so a null that could not
        be fitted says why instead of vanishing into a NaN.

    Returns
    -------
    (pred, scored_mask) : (n, g) float array with NaN where nothing was
    predicted, and the (n,) bool mask of spots that carry an out-of-sample
    prediction under the chosen fit mode.

    Statistical premise
        Vectorised over genes: one lstsq solve per fold for the whole matrix.
        That is exact, not an approximation, least squares with a shared
        design is separable across targets.

    Not applicable when
        The design is wide relative to the section (n < 2*(p+1) for the
        cross-fit, n_train < p+2 for train_only, p >= n-2 for the oracle):
        the fit would interpolate rather than estimate, so an all-NaN result
        is returned with the reason in ``notes``.
    """
    # --- 1. Protocol gate -------------------------------------------------
    # WHAT: reject any fit_mode outside the closed FIT_MODES tuple.
    # WHY: refuse rather than silently cross-fitting; RunMeta prints the
    #      string it was given, so an unrecognised mode would mislabel the
    #      protocol in an archived report.
    # WHERE: fit_mode arrives from audit()'s null_fit argument via each null
    #      wrapper below (depth_null etc., called at audit.py:428-463).
    if fit_mode not in FIT_MODES:
        raise BadInput("unknown fit_mode=%r; use one of %s"
                       % (fit_mode, "|".join(FIT_MODES)))
    # --- 2. Coercion and empty result skeleton ----------------------------
    # WHAT: force float64 copies and pre-fill the outputs with "nothing
    #      predicted" values.
    # WHY: lstsq needs float; starting from all-NaN / all-False means every
    #      early return below is automatically a well-formed N/A result.
    # WHERE: Y is the section slice of [L01]; X is built by the wrapper
    #      (depth_null / composition_null / depth_composition_null); the
    #      pair (pred, scored) is [L26], later permuted by
    #      audit._perm_of_pred (audit.py:588).
    Y = np.asarray(Y, dtype=np.float64)  # (n, g) targets, declared space
    X = np.asarray(X, dtype=np.float64)  # (n, p) design, intercept included
    n = Y.shape[0]                       # spots in this section
    pred = np.full(Y.shape, np.nan)      # (n, g) predictions, NaN = unscored
    scored = np.zeros(n, dtype=bool)     # (n,) True where pred is out-of-sample

    # --- 3. Bail-out bookkeeping ------------------------------------------
    # WHAT: a one-line sink for human-readable failure reasons.
    # WHY: a null that could not be fitted must say why in the report's N/A
    #      ledger instead of vanishing into a NaN.
    # WHERE: `notes` is the per-section list audit._section_block passes in;
    #      its lines end up in the degradation/N/A ledgers [L48]/[L50].
    def _note(msg):
        if notes is not None:
            notes.append(msg)

    # --- 4. Finite-design guard -------------------------------------------
    # WHAT: refuse a design with NaN/inf entries before any solve.
    # WHY: np.linalg.lstsq answers a non-finite design with
    #      LinAlgError('SVD did not converge'), which names neither the input
    #      nor the section.  Degrade to N/A with a sentence instead.
    # WHERE: non-finite entries usually trace back to lib_size [L04] or the
    #      composition matrix [L07] the wrappers stacked into X.
    if not np.isfinite(X).all():
        _note("not fitted: the zero-pixel design contains %d non-finite "
              "value(s) (check lib_size / composition)"
              % int((~np.isfinite(X)).sum()))
        return pred, scored
    p = X.shape[1]  # design columns, intercept counted
    # --- 5. Oracle branch: in-sample upper bound --------------------------
    # WHAT: fit on ALL spots of the section and score on the same spots.
    # WHY: an in-sample fit is an UPPER BOUND by construction; it exists so a
    #      reader can see the ceiling of the null family.  With p >= n-2 it
    #      is not even that, it interpolates and the r saturates at 1.0 on
    #      pure noise, so refuse instead of publishing a saturated fit as
    #      "the ceiling of any pure composition model".
    # WHERE: reached when audit() was called with null_fit='oracle'; the
    #      headline candidate list bars oracle nulls (audit.py:1117-1121).
    if fit_mode == "oracle":
        if p >= n - 2:
            _note("oracle fit saturated: %d design columns on %d spots "
                  "(an in-sample fit with p >= n-2 interpolates)" % (p, n))
            return pred, scored
        # B is the (p, g) coefficient matrix: one least-squares solve serves
        # every gene at once because the design is shared across targets.
        B, *_ = np.linalg.lstsq(X, Y, rcond=None)
        pred = X @ B
        scored[:] = True
    # --- 6. train_only branch: honour the user's own split ----------------
    # WHAT: fit coefficients on is_train spots, score only the held-out rest.
    # WHY: when the audited model was trained on a declared split, the null
    #      must use the SAME split, otherwise null and model see different
    #      information and the comparison is unfair.  The size guard (p+2
    #      train spots, 3 held-out spots) keeps the fit estimating rather
    #      than interpolating and the correlation computable.
    # WHERE: is_train is the section slice of [L10] (audit.py:992); the
    #      scored mask (~tr) later aligns with the ev mask [L23] so model and
    #      null are read on the same spots.
    elif fit_mode == "train_only":
        if is_train is None:
            raise ValueError("fit_mode='train_only' needs is_train")
        tr = np.asarray(is_train, dtype=bool)  # (n,) True = fitting spot
        if tr.sum() < p + 2 or (~tr).sum() < 3:
            _note("not fitted: train_only needs >=%d train spots (has %d) and "
                  ">=3 held-out spots (has %d)"
                  % (p + 2, int(tr.sum()), int((~tr).sum())))
            return pred, scored
        B, *_ = np.linalg.lstsq(X[tr], Y[tr], rcond=None)
        pred = X @ B          # predict everywhere, but only ~tr is scored
        scored = ~tr
    # --- 7. Default branch: K-fold cross-fitting within the section -------
    # WHAT: split the section into k folds; each fold's spots are predicted
    #      by coefficients fitted on the other folds.
    # WHY: every prediction is out-of-sample without needing a user split,
    #      so the null is a fair competitor, not an in-sample fit.  The
    #      n >= 2*(p+1) guard keeps every training fold overdetermined.
    # WHERE: `folds` is normally the shared per-section partition [L20]
    #      (audit.py:382, one partition reused by ALL nulls so that a
    #      difference between two nulls measures the nulls, not two random
    #      splits); kfold_indices (core.py:414) only runs here when a caller
    #      outside audit() did not supply folds.
    else:  # cv_within_section
        rng = np.random.default_rng(0) if rng is None else rng
        if n < 2 * (p + 1):
            _note("not fitted: cross-fitting needs n >= 2*(p+1) = %d spots, "
                  "section has %d" % (2 * (p + 1), n))
            return pred, scored
        if folds is None:
            folds = kfold_indices(n, k, rng)
        for tr, te in folds:
            # Skip a fold whose training half is too small to determine the
            # coefficients; its test spots simply stay NaN/unscored.
            if len(tr) < p + 1:
                continue
            B, *_ = np.linalg.lstsq(X[tr], Y[tr], rcond=None)
            pred[te] = X[te] @ B
            scored[te] = True
    # (pred, scored) is [L26]: the caller both correlates it with the truth
    # (via _score) and permutes it to build the null's own floor.
    return pred, scored


def _score(Y, pred, scored):
    """Correlate a null's predictions with the truth on the scored spots only.

    WHAT: turn (pred, scored) into the (r_per_gene, pred, scored) triple that
        every public null returns.
    WHY: with fewer than 3 scored spots a Pearson r is undefined (2 points
        always correlate perfectly), so the r vector degrades to all-NaN
        while pred/scored still travel, letting the caller book the reason.
    WHERE: Y and pred are restricted to the SAME scored rows, so the null is
        never credited for in-sample spots; the r vector is the head of the
        [L25] chain (colcorr, core.py:291 -> agg -> _combine -> Stat) and
        (pred, scored) continue as [L26] into audit._perm_of_pred.
    """
    if scored.sum() < 3:
        return np.full(Y.shape[1], np.nan), pred, scored
    return colcorr(Y[scored], pred[scored]), pred, scored


# ------------------------------------------------------------------- nulls
def depth_null(Y, loglib, fit_mode="cv_within_section", is_train=None,
               rng=None, notes=None, folds=None):
    """Zero-pixel library-size null: one scalar per spot.

    Parameters
    ----------
    Y : (n, g) ndarray, expression of one section.
    loglib : (n,) array-like, natural log of the per-spot library size
        (full-transcriptome total counts, not the panel row sum).
    fit_mode, is_train, rng, notes : see :func:`linear_null_pred`.

    Returns
    -------
    (r_per_gene, pred, scored_mask)

    Statistical premise
        A per-gene linear regression on one number.  It carries no spatial,
        morphological or biological information whatsoever; whatever r it
        reaches is the part of the metric that sequencing depth alone buys.
        On log1p-CP10K targets that share is large because the transform is
        non-linear in depth, not because dense spots hold more cells.
    """
    # --- Design: X = [1, log lib], the two-column depth-only regression ---
    # WHAT: an intercept plus the single log-depth number per spot.
    # WHY: this is the weakest imaginable competitor, one scalar; anything
    #      the audited model cannot beat here is depth artefact, not biology.
    # WHERE: loglib is the section slice of [L05], born once at audit.py:805
    #      and handed in at audit.py:428; the shared folds are [L20]; the
    #      returned triple enters the [L25]/[L26] chains at audit.py:428-433.
    l = np.asarray(loglib, dtype=np.float64).reshape(-1, 1)  # (n, 1) log depth
    X = np.hstack([np.ones_like(l), l])                      # (n, 2) design
    pred, scored = linear_null_pred(Y, X, fit_mode, is_train, rng,
                                    notes=notes, folds=folds)
    return _score(Y, pred, scored)


def composition_null(Y, comp, fit_mode="cv_within_section", is_train=None,
                     rng=None, notes=None, folds=None):
    """Zero-pixel composition null: class-mean lookup / deconvolution mix.

    Parameters
    ----------
    comp : (n,) or (n, K) array-like
        A one-hot label matrix (class-mean lookup) or continuous cell-type
        proportions.  The intercept is added here.

    Statistical premise
        This is the "a pathologist's label table already predicts expression"
        null.  With one-hot input the least-squares fit IS the per-class mean,
        so the cross-fitted variant answers: how much r does knowing each
        spot's class, and nothing else, buy on held-out spots?

    Not applicable when
        K is large relative to the section: see :func:`linear_null_pred`.
        Deconvolution references with tens of cell types on small sections hit
        this, and the reason is reported rather than silently NaN.
    """
    # --- Design: X = [1, comp], class membership only ---------------------
    # WHAT: an intercept plus the K composition columns; a 1-D label vector
    #      is promoted to a single column first.
    # WHY: with one-hot rows least squares reproduces the per-class mean, so
    #      cross-fitting turns this into "predict each spot by its tissue
    #      class's average profile", exactly the pathologist's-table null.
    #      The intercept makes the one-hot block collinear by construction;
    #      that is deliberate and absorbed downstream by core.ortho_basis
    #      (core.py:331, SVD threshold at 358) wherever this block re-enters
    #      a residualising design.
    # WHERE: comp is the section slice of [L07] (one-hot built by
    #      audit._one_hot at audit.py:875, or user proportions), handed in at
    #      audit.py:448/455; the triple enters the [L25]/[L26] chains there.
    P = np.asarray(comp, dtype=np.float64)  # (n, K) proportions or one-hot
    if P.ndim == 1:
        P = P[:, None]
    X = np.hstack([np.ones((P.shape[0], 1)), P])  # (n, 1+K) design
    pred, scored = linear_null_pred(Y, X, fit_mode, is_train, rng,
                                    notes=notes, folds=folds)
    return _score(Y, pred, scored)


def depth_composition_null(Y, loglib, comp, fit_mode="cv_within_section",
                           is_train=None, rng=None, notes=None, folds=None):
    """The joint zero-pixel null: depth AND composition in one design.

    This is usually the strongest null in the report, because depth and
    composition explain different directions of the expression matrix; a model
    that only beats each of them separately has not beaten their union.
    """
    # --- Design: X = [1, log lib, comp], the union of the two nulls -------
    # WHAT: stack intercept, log depth and the composition columns into one
    #      (n, 2+K) design and fit it exactly like the single nulls.
    # WHY: depth and composition explain different directions of Y, so their
    #      joint fit is usually the strongest zero-pixel competitor; beating
    #      each alone does not imply beating the union.
    # WHERE: loglib is [L05] (handed in at audit.py:463), comp is [L07],
    #      folds is the shared [L20] partition; the triple enters the
    #      [L25]/[L26] chains at audit.py:463-467 and often becomes the
    #      strongest_null [L41] the headline is judged against.
    l = np.asarray(loglib, dtype=np.float64).reshape(-1, 1)  # (n, 1)
    P = np.asarray(comp, dtype=np.float64)                   # (n, K)
    if P.ndim == 1:
        P = P[:, None]
    X = np.hstack([np.ones_like(l), l, P])                   # (n, 2+K)
    pred, scored = linear_null_pred(Y, X, fit_mode, is_train, rng,
                                    notes=notes, folds=folds)
    return _score(Y, pred, scored)


def neighbour_null(Y, W):
    """Predict each spot's TRUE value by the mean of its neighbours' TRUE values.

    Parameters
    ----------
    Y : (n, g) ndarray
    W : (n, n) sparse, row-standardised neighbour weights (self excluded).

    Statistical premise
        Pure spatial smoothing; the image is never consulted.  Because it
        reads held-out ground truth at scoring time, no image model can
        actually implement it, it is a reference point for how much of the
        metric is explained by spatial autocorrelation alone, not a baseline a
        model could be asked to beat fairly.

    Not applicable when
        Fewer than 3 spots have any neighbour (all-NaN result).
    """
    # --- 1. Who can be scored ---------------------------------------------
    # WHAT: mark the spots that have at least one neighbour (row sum of the
    #      row-standardised W is 1 there, 0 for islands).
    # WHY: an island spot has no neighbour mean, so it must be excluded from
    #      the correlation rather than scored against a 0/NaN placeholder.
    # WHERE: W is [L28], built by neighbour_matrix from the coords [L06]
    #      (audit.py:471, rebuilt on the held-out subset at 510).
    Y = np.asarray(Y, dtype=np.float64)
    has = np.asarray(W.sum(1)).ravel() > EPS  # (n,) True = spot has neighbours
    pred = np.full(Y.shape, np.nan)           # (n, g) neighbour means
    if has.sum() < 3:
        # Fewer than 3 scorable spots: Pearson r undefined, degrade to NaN.
        return np.full(Y.shape[1], np.nan), pred, has
    # --- 2. The prediction is one sparse product --------------------------
    # WHAT: W @ Y averages each spot's neighbours' TRUE values (W rows sum
    #      to 1, diagonal is zero, so a spot never sees itself).
    # WHY: no fitting at all; this null has zero free parameters, which is
    #      why it needs no folds, no is_train and no fit_mode.
    # WHERE: the triple feeds audit.py:472-475 and the section scalar
    #      becomes r_nbr [L30] in SpatialResult and CeilingResult.
    pred[has] = (W @ Y)[has]
    return colcorr(Y[has], pred[has]), pred, has


# ------------------------------------------------------------ neighbourhoods
def neighbour_matrix(coords, coord_kind="grid", mode="rook", k=4,
                     standardise=True):
    """Neighbour weights as a scipy CSR matrix.

    Parameters
    ----------
    coords : (n, 2) array-like
        Array coordinates (``grid``) or physical coordinates (``micron``).
    coord_kind : {'grid', 'micron'}
    mode : {'rook', 'queen'}
        ``rook`` = the 4 lattice neighbours at Manhattan distance 1;
        ``queen`` = the 8 lattice neighbours of the 3x3 box, self excluded.
        IGNORED for ``coord_kind='micron'``, where neighbours are the k
        nearest spots regardless of direction.
    k : int
        Neighbour count for the micron branch only.
    standardise : bool
        Row-standardise so each row sums to 1 (the usual convention, and what
        makes ``W @ Y`` a neighbour MEAN).  Pass False to get the raw 0/1
        incidence matrix, needed when the caller intends to add further
        edges and standardise once at the end.

    Returns
    -------
    (n, n) CSR matrix with a zero diagonal.
    """
    # --- 1. Grid branch: exact lattice adjacency --------------------------
    # WHAT: round coordinates to integers, hash every (x, y) cell to its row
    #      index, then link each spot to whichever of its rook/queen lattice
    #      offsets actually exists in the section.
    # WHY: array coordinates (Visium/ST grids) make adjacency exact; a
    #      distance query would blur it.  Missing neighbours (tissue edge,
    #      filtered spots) simply produce no edge, no padding is invented.
    # WHERE: coords is the section slice of [L06] (audit.py:991); the CSR
    #      result is [L28], consumed by neighbour_null and morans_I, and
    #      unstandardised by smooth_matrix below.
    C = np.asarray(coords, dtype=np.float64)  # (n, 2) spot positions
    n = C.shape[0]
    if coord_kind == "grid":
        g = np.rint(C).astype(np.int64)  # (n, 2) integer lattice cells
        # key: lattice cell -> row index, for O(1) neighbour lookup
        key = {(int(a), int(b)): i for i, (a, b) in enumerate(g)}
        if mode == "queen":
            offs = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                    if (dx, dy) != (0, 0)]
        else:
            offs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        rows, cols = [], []  # COO edge lists, one entry per directed edge
        for i, (a, b) in enumerate(g):
            for dx, dy in offs:
                j = key.get((int(a) + dx, int(b) + dy))
                if j is not None:
                    rows.append(i)
                    cols.append(j)
        W = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    # --- 2. Micron branch: k nearest neighbours ---------------------------
    # WHAT: physical coordinates get a KD-tree k-NN query instead of lattice
    #      offsets; `mode` is ignored here.
    # WHY: off-lattice platforms have no rook/queen notion; the k nearest
    #      spots are the natural analogue.
    else:
        from scipy.spatial import cKDTree
        # Query one extra neighbour and drop self EXPLICITLY (by index, not by
        # position): with coincident coordinates the KD-tree breaks the
        # zero-distance tie arbitrarily, so `idx[:, 0]` is NOT guaranteed to be
        # the spot itself and slicing it off used to leave spots as their own
        # neighbours (self-weight 1/k).
        kk = min(k + 2, n)  # query 2 extra so dropping self still leaves k
        _, idx = cKDTree(C).query(C, k=kk)  # (n, kk) nearest-spot indices
        idx = np.atleast_2d(idx)
        rows, cols = [], []
        for i in range(n):
            # Drop self by INDEX, not by position: see the docstring note on
            # coincident coordinates, then keep the first k true neighbours.
            js = [int(j) for j in idx[i] if int(j) != i][:k]
            rows.extend([i] * len(js))
            cols.extend(js)
        W = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    # --- 3. Optional row-standardisation ----------------------------------
    # WHAT: divide each row by its sum so W @ Y is a neighbour MEAN.
    # WHY: smooth_matrix needs the RAW incidence (standardise=False) so it
    #      can add the self-loop before normalising once; standardising here
    #      first would change the smoother's weights (see smooth_matrix).
    if not standardise:
        return W
    return _row_standardise(W)


def _row_standardise(W):
    """Divide every row by its sum; all-zero rows (islands) stay all-zero.

    WHAT: left-multiply by diag(1/row_sum), with islands mapped to 0 rather
        than 1/0.
    WHY: guarding the reciprocal avoids inf rows for isolated spots; those
        rows stay zero, and neighbour_null's `has` mask then excludes them
        from scoring instead of treating 0 as a prediction.
    WHERE: called by neighbour_matrix (standardise=True path of [L28]) and
        by smooth_matrix after the self-loop is added ([L29]).
    """
    deg = np.asarray(W.sum(1)).ravel()  # (n,) row sums (neighbour counts)
    inv = np.zeros_like(deg)
    nz = deg > 0
    inv[nz] = 1.0 / deg[nz]
    return sp.diags(inv) @ W


def smooth_matrix(coords, coord_kind="grid", k=8):
    """3x3 box smoother INCLUDING the centre spot (the classic 'smoothed GT').

    Returns the row-standardised operator S such that ``S @ y`` replaces each
    spot's value by the unweighted mean of the 3x3 box around it (interior
    grid spots: 1/9 on the centre and on each of the 8 neighbours).

    Statistical premise
        This is the smoothing that some pipelines apply to the GROUND TRUTH
        before scoring.  The audit prices that protocol choice by scoring the
        unchanged predictions against the smoothed truth, so the operator has
        to be the same box mean the literature applies, a heavier centre
        weight would understate the lever.

    Implementation note
        The incidence matrix is built UNSTANDARDISED, the self-loop added, and
        the result standardised ONCE.  Adding the identity to an already
        row-standardised W would give the centre weight 1/2 and each neighbour
        1/16, which is a different (much weaker) smoother.
    """
    # WHAT: raw queen incidence + identity, then ONE row-standardisation.
    # WHY: order matters; see the implementation note in the docstring (the
    #      other order gives a much weaker smoother, understating the lever).
    # WHERE: coords is [L06]; the operator S is [L29], applied as ys = S @ y
    #      at audit.py:566, and colcorr(ys, yh) at 567 prices the
    #      smoothed-ground-truth lever in SpatialResult.smooth_lever.
    W = neighbour_matrix(coords, coord_kind, mode="queen", k=k,
                         standardise=False)  # (n, n) raw 0/1 incidence
    n = W.shape[0]
    S = W + sp.eye(n, format="csr")  # self-loop: centre spot joins its box
    return _row_standardise(S)


def morans_I(resid_mat, W):
    """Per-gene Moran's I of a residual matrix under row-standardised W.

    Parameters
    ----------
    resid_mat : (n, g) ndarray, typically ``y - y_hat``.
    W : (n, n) sparse neighbour weights.

    Returns
    -------
    (g,) ndarray; NaN for genes whose residual is constant.

    Statistical premise
        I > 0 means the residual is still spatially smooth, i.e. the model
        left structure on the table that a smoother could have picked up.  The
        expectation under no autocorrelation is -1/(n-1), not 0, so small
        positive values are not automatically meaningful, read I against its
        permutation floor, which the audit attaches.

    Not applicable when
        Rows without neighbours (islands) still enter the n/S0 scale factor
        while contributing nothing to the numerator; that is the textbook
        convention, and it deflates I on ragged tissue outlines.
    """
    # --- 1. Centre and collect the scale terms ----------------------------
    # WHAT: subtract each gene's mean across spots, then take S0 = sum of all
    #      weights (Moran's normalising constant).
    # WHY: Moran's I is defined on deviations from the mean; with a
    #      row-standardised W, S0 equals the number of non-island rows, and
    #      S0 <= 0 means there are no edges at all, so I is undefined.
    # WHERE: resid_mat is R = y - y_hat built at audit.py:572; W is [L28]
    #      (the ev-subset rebuild from audit.py:510); the same code runs on
    #      permuted residual blocks in the loop at audit.py:575-579 to give
    #      I its permutation floor.
    R = np.asarray(resid_mat, dtype=np.float64)  # (n, g) residuals
    R = R - R.mean(0, keepdims=True)             # per-gene centred
    S0 = float(W.sum())                          # total weight, scalar
    if S0 <= 0:
        return np.full(R.shape[1], np.nan)
    # --- 2. The I ratio, vectorised over genes ----------------------------
    # WHAT: numerator = cross-product of each residual with its neighbour
    #      average, denominator = the residual's own sum of squares; scale by
    #      n/S0.  One sparse product serves all g genes.
    # WHY: this is the textbook estimator; constant-residual genes (den ~ 0)
    #      are returned as NaN rather than +-inf, matching how NaN travels
    #      through the [L25]-style aggregation downstream.
    num = (R * (W @ R)).sum(0)  # (g,) spatial cross-products
    den = (R * R).sum(0)        # (g,) sums of squares
    with np.errstate(invalid="ignore", divide="ignore"):
        I = (R.shape[0] / S0) * num / den
    return np.where(den <= EPS, np.nan, I)
