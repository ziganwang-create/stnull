# -*- coding: utf-8 -*-
"""stnull.drg: the attribution ladder: r_full -> DG -> CRG -> DRG.

Author: Zigan Wang.

WHAT THIS MODULE IS
    The part of the audit that asks what survives when the known confounders
    are controlled.  Each rung is a per-gene partial correlation between the
    truth and the predictions after residualising BOTH on a design:

        r_full   X = [1]                     no control
        DG       X = [1, l_hat]              depth controlled
        CRG      X = [1, pi]                 composition controlled
        DRG      X = [1, l_hat, pi]          both controlled

    DRG is a property of THESE PREDICTIONS.  It is never an upper bound on the
    information content of H&E, see ``stnull.report.claims()``, which refuses
    that reading explicitly.

WHO CALLS IT
    ``audit._section_block``: once per section for the point estimates, again
    for the permutation floor of each rung, and once per bootstrap draw.

WHAT l_hat IS
    The sequencing depth that the model's own predictions carry.  With
    ``depth_proxy='from_pred'`` it is obtained by cross-fitting log(library
    size) on y_hat inside the section (dual/kernel ridge, so n_gene >> n_spot
    is fine).  ``depth_proxy='observed'`` uses the measured log library size
    instead.  NOTE: [1, l_hat] and [1, log lib] are DIFFERENT one-dimensional
    subspaces and neither nests the other, so 'observed' is NOT uniformly more
    conservative: on synthetic data with a real depth-independent signal it
    is 'from_pred' that is badly biased (it can absorb the signal itself).
    'observed' is the validated default; it matched the oracle ladder to
    +/-0.002 in all synthetic scenarios.

DATA LINEAGE (row numbers cite docs/CODE_WALKTHROUGH.md (section 4, the variable lineage table))
    Every function here runs on ONE tissue section at a time; all inputs are
    per-section slices made inside ``audit._section_block`` (audit.py:333).

    Inputs
      y, yh       (n, g) truth and predictions: the [L01]/[L02] slices
                  ``Y[m]`` / ``P[m]`` cut at audit.py:988.
      lhat        (n,) depth control [L21], born at audit.py:386-404 by one of
                  three routes: 'observed' = loglib [L05], 'given' = the
                  user's lib_pred [L11], 'from_pred' = ``fit_depth_proxy``
                  in this file (called at audit.py:395).
      loglib      (n,) log library size [L05] (born once at audit.py:805),
                  the regression TARGET of ``fit_depth_proxy``.
      comp        (n, K) composition block [L07]: one-hot labels (merged and
                  encoded at audit.py:871-875) or user proportions, sliced at
                  audit.py:990.
      perm_idx / perm_mask
                  the shared permutation plan [L19] from ``core.perm_plan``
                  (built at audit.py:374, rebuilt on the held-out subset at
                  audit.py:504), handed to ``perm_null_of_rung``.

    Outputs
      ``ladder_rungs`` returns the four per-gene vectors [L31] plus the design
      matrices [L32]; audit.py:513-524 aggregates them into the ladder Stats,
      the ladder table [L44] and the per_gene table [L43].
      ``perm_null_of_rung`` returns the (B,) permutation floors: per-rung
      floors (audit.py:525), the mean-rule floor [L36] (audit.py:530-531) and
      the free-permutation contrast [L37] (audit.py:584).
      ``boot_of_rung`` returns one bootstrap scalar per call; through
      ``core.boot_ci`` (audit.py:532-533) these become the rung CIs [L34].
      ``fit_depth_proxy`` returns (lhat, r_of_pred, alpha, note): lhat is
      [L21], the two scalars are [L22] (reported as depth_r_of_pred).

    Terms (plain language; full glossary in CODE_WALKTHROUGH.md)
      A "spot" is one measured location on the tissue slide.  "Library size"
      is a spot's total molecule (UMI) count, its sequencing depth.
      "Pearson r" is the ordinary linear correlation between two vectors.
      A "permutation floor" is the value the same readout reaches when the
      pairing between truth and prediction is destroyed by shuffling spots:
      anything below the floor is indistinguishable from chance.
"""
from __future__ import annotations

import numpy as np

from .core import colcorr, kfold_indices, ortho_basis, resid

#: Ridge penalties searched by the from_pred depth proxy.  A grid, not a
#: continuous optimiser, because the selection has to be repeatable under
#: permutation for the guard below to have a null.
DEFAULT_ALPHAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
_GUARD_NPERM = 30  # permutations for the from_pred selection guard


# --------------------------------------------------- depth proxy (from_pred)
def fit_depth_proxy(yhat, loglib, rng=None, k=5, alphas=DEFAULT_ALPHAS):
    """Cross-fitted kernel-ridge prediction of log library size from y_pred.

    Parameters
    ----------
    yhat : (n, g) ndarray
        The model's predictions for one section, in the declared target space.
    loglib : (n,) array-like
        Natural log of the observed library size, same spots.
    rng : numpy Generator or None
    k : int
        Cross-fitting folds; the same folds are reused for every alpha and for
        every guard permutation, so the alpha comparison is paired.
    alphas : sequence of float
        Ridge penalty grid.

    Returns
    -------
    (l_hat, r_of_pred, alpha, note) where ``r_of_pred`` = corr(l_hat, loglib)
    is "how much sequencing depth the model's predictions carry".  ``l_hat``
    is all-NaN when the proxy is rejected, and ``note`` says why.

    Statistical premise
        The kernel is the linear kernel of standardised predictions, so this
        is ridge regression in the n_spot-dimensional dual space: the cost is
        driven by spots, not by genes, which is what makes an 800-gene panel
        on a 300-spot section tractable.  Cross-fitting matters because an
        in-sample l_hat would soak up the model's signal wholesale.

    Not applicable when
        n < 10 spots, no alpha converges, or the proxy fails the permutation
        guard below.  In every case l_hat comes back all-NaN and the caller
        must set the depth rungs to N/A rather than residualising on noise.
    """
    # ---- [1] Coerce inputs --------------------------------------------------
    # WHAT: cast both inputs to float64 arrays.
    # WHERE: yhat is the per-section prediction slice P[m] [L02] and loglib is
    # the per-section slice of [L05] (born once at audit.py:805); both are
    # passed in by audit.py:395.
    Yh = np.asarray(yhat, dtype=np.float64)   # (n, g) predictions, one section
    l = np.asarray(loglib, dtype=np.float64).ravel()  # (n,) log library size
    n = Yh.shape[0]                           # spots in this section
    # ---- [2] Small-section guard --------------------------------------------
    # WHAT: refuse to fit on fewer than 10 spots.
    # WHY: the 5-fold cross-fit below would leave ~2-spot folds, and any
    # correlation estimated there is noise.  The all-NaN return is a signal:
    # audit.py:406-408 sees it and sets the depth rungs to N/A instead of
    # residualising on garbage.
    if n < 10:
        return (np.full(n, np.nan), float("nan"), float("nan"),
                "too few spots for a depth proxy")
    # WHERE: audit.py:395 passes the section fit stream (the middle of the
    # three RNG streams split at audit.py:371-373 [L18]), so changing n_perm
    # or n_boot elsewhere cannot move this point estimate.
    rng = np.random.default_rng(0) if rng is None else rng
    # ---- [3] Build the linear kernel of standardised predictions ------------
    # WHAT: centre and scale each gene column of the predictions, then form
    # the (n, n) Gram matrix K = Z Z^T / g.
    # WHY: standardise per gene so that no single high-variance gene dominates
    # the kernel; dividing by n_gene keeps K on the same scale as alpha
    # whatever the panel size, which is what makes one fixed alpha grid usable
    # across 50-gene and 800-gene panels.  ("Panel" = the set of genes the
    # assay measures.)  Working in the (n, n) dual space instead of the (g, g)
    # primal one is what keeps an 800-gene panel on a 300-spot section cheap.
    Z = Yh - Yh.mean(0, keepdims=True)  # (n, g) centred predictions
    sd = Z.std(0, ddof=0)               # (g,) per-gene standard deviation
    sd[sd <= 0] = 1.0                   # constant genes: leave them centred
    Z = Z / sd                          # (n, g) standardised predictions
    K = Z @ Z.T / Z.shape[1]            # (n, n) linear kernel (Gram matrix)
    # ---- [4] One fold partition, shared by every alpha and every guard draw -
    # WHAT: split the section's spots into k cross-fitting folds.
    # WHY: reusing ONE partition makes the alpha comparison paired (only alpha
    # varies) and gives the permutation guard in [6] a null of exactly the
    # selection performed here.  This is a fresh local draw from the fit
    # stream, distinct from the section-wide shared folds [L20] used by the
    # nulls (audit.py:382); the two never mix spots across sections anyway.
    folds = kfold_indices(n, k, rng)    # list of k (train_idx, test_idx) pairs
    # ---- [5] Alpha grid search with out-of-fold scoring ---------------------
    # WHAT: for each ridge penalty alpha, predict loglib out of fold via the
    # dual ridge solve, then keep the alpha whose out-of-fold prediction
    # correlates best with the observed loglib.
    # WHY: cross-fitting is essential, an in-sample l_hat would soak up the
    # model's biological signal wholesale (see the docstring), and the DG/DRG
    # rungs would then residualise the signal away.
    best_a, best_r, best_pred = np.nan, -np.inf, None  # winner: alpha, r, (n,)
    for a in alphas:
        pred = np.full(n, np.nan)       # (n,) out-of-fold prediction of loglib
        for tr, te in folds:
            Ktr = K[np.ix_(tr, tr)]     # train-block kernel
            ybar = l[tr].mean()         # centre the target within the fold
            try:
                # w: dual ridge weights, one per training spot
                w = np.linalg.solve(Ktr + a * np.eye(len(tr)), l[tr] - ybar)
            except np.linalg.LinAlgError:
                continue
            pred[te] = K[np.ix_(te, tr)] @ w + ybar
        ok = np.isfinite(pred)
        # WHY: fewer than 5 scored spots, or a constant prediction, cannot
        # yield a meaningful correlation; skip this alpha entirely.
        if ok.sum() < 5 or np.std(pred[ok]) <= 0:
            continue
        rv = float(np.corrcoef(pred[ok], l[ok])[0, 1])  # out-of-fold Pearson r
        if np.isfinite(rv) and rv > best_r:
            best_a, best_r, best_pred = a, rv, pred
    # WHAT: no alpha produced a usable prediction, report non-convergence.
    # WHERE: the all-NaN lhat again routes audit.py:406-408 to N/A rungs.
    if best_pred is None:
        return (np.full(n, np.nan), float("nan"), float("nan"),
                "depth proxy did not converge")

    # ---- [6] Permutation selection guard ------------------------------------
    # WHAT: re-run the IDENTICAL grid selection _GUARD_NPERM times on permuted
    # loglib (same K, same folds, same alphas) and require the observed best_r
    # to beat the 95th percentile of the permuted maxima.
    # WHY: ``best_r`` is the MAXIMUM over the alpha grid, so it is
    # upward-biased under the null.  Re-running the identical selection on
    # permuted loglib gives the null distribution of that maximum; if the
    # observed value does not clear it, the proxy is not tracking depth and
    # residualising on it would delete whatever signal the model actually has
    # (the false negative documented in
    # tests/test_synthetic.py::test_signal_scenario).
    # WHERE: a rejection here is why lhat [L21] can be all-NaN even when the
    # grid converged; the note travels back to audit.py:395 and into the
    # degradation ledger.
    nperm = int(_GUARD_NPERM)
    if nperm > 0:
        null_max = []                   # permuted grid maxima, one per draw
        for _ in range(nperm):
            lp = rng.permutation(l)     # (n,) loglib in shuffled spot order
            bm = -np.inf                # this draw's best r over the grid
            for a in alphas:
                pr_ = np.full(n, np.nan)
                for tr, te in folds:
                    Ktr = K[np.ix_(tr, tr)]
                    yb = lp[tr].mean()
                    try:
                        w = np.linalg.solve(Ktr + a * np.eye(len(tr)), lp[tr] - yb)
                    except np.linalg.LinAlgError:
                        continue
                    pr_[te] = K[np.ix_(te, tr)] @ w + yb
                ok_ = np.isfinite(pr_)
                if ok_.sum() < 5 or np.std(pr_[ok_]) <= 0:
                    continue
                rv_ = float(np.corrcoef(pr_[ok_], lp[ok_])[0, 1])
                if np.isfinite(rv_) and rv_ > bm:
                    bm = rv_
            if np.isfinite(bm):
                null_max.append(bm)
        if null_max:
            # thr: the 95th percentile of the null maxima, the bar the
            # observed selection must clear (same convention as the package's
            # other permutation floors, core.py:271).
            thr = float(np.quantile(null_max, 0.95))
            if not (best_r > thr):
                return (np.full(n, np.nan), best_r, float(best_a),
                        "depth proxy not distinguishable from noise "
                        "(r=%.3f <= perm95=%.3f, n_perm=%d); depth rung set to N/A "
                        "-- pass lib_size and use depth_proxy='observed'"
                        % (best_r, thr, nperm))
    # ---- [7] Return the accepted proxy --------------------------------------
    # WHERE: best_pred becomes lhat [L21] at audit.py:395 (feeding the depth_1d
    # null at audit.py:440 and the ladder design X1 = [1, lhat] below at
    # ladder_rungs); best_r and best_a become depth_r_of_pred / depth_alpha
    # [L22], surfaced as DepthResult.depth_r_of_pred (audit.py:1190-1195).
    return best_pred, best_r, float(best_a), "kernel ridge, %d-fold cross-fit" % len(folds)


# ------------------------------------------------------------- the four rungs
def ladder_rungs(y, yh, lhat=None, comp=None):
    """Per-gene r at the four rungs. Missing controls -> that rung is None.

    Parameters
    ----------
    y, yh : (n, g) ndarray, truth and predictions for one section.
    lhat : (n,) array-like or None, the depth control.
    comp : (n,) / (n, K) array-like or None, the composition control.

    Returns
    -------
    dict with keys ``r_full``, ``dg``, ``crg``, ``drg`` (each a (g,) array of
    per-gene r, or None when the control is missing) plus ``_designs``, the
    design matrices, so the caller can reuse them for floors and bootstraps.

    Statistical premise
        Partial correlation, both sides residualised (see core.partial_r).
        A rung is a comparison between predictions and truth WITHIN the
        subspace the control cannot explain; it is not a variance
        decomposition and the rungs do not sum to anything.

    Not applicable when
        A control contains non-finite values, that rung is set to None with
        no design, because an SVD on a NaN design raises an error naming
        neither the input nor the section.
    """
    # ---- [1] Coerce and start with the uncontrolled rung --------------------
    # WHERE: y and yh are the per-section slices Y[m]/P[m] [L01]/[L02]; lhat
    # is [L21]; comp is [L07].  Called once per section at audit.py:513.
    y = np.asarray(y, dtype=np.float64)    # (n, g) truth, one section
    yh = np.asarray(yh, dtype=np.float64)  # (n, g) predictions, same spots
    n = y.shape[0]                         # spots in this section
    one = np.ones((n, 1))                  # intercept column of every design
    # r_full: rung 0, plain per-gene Pearson r with no control (core.colcorr,
    # core.py:291).  This is the number papers usually report.
    out = {"r_full": colcorr(y, yh)}       # (g,) per-gene r [L31]
    # ---- [2] Build the control designs [L32] --------------------------------
    # WHAT: X1 = [1, lhat] (depth), X2a = [1, comp] (composition),
    # X2 = [1, lhat, comp] (both).  A missing or non-finite control leaves its
    # design as None, which _pr turns into an absent (None) rung.
    # WHY the finiteness checks: fit_depth_proxy returns all-NaN lhat when the
    # proxy is rejected, and an SVD on a NaN design raises an error naming
    # neither the input nor the section; degrading the rung to N/A is the
    # informative failure mode.
    X1 = X2a = X2 = None
    if lhat is not None and np.isfinite(np.asarray(lhat)).all():
        L = np.asarray(lhat, dtype=np.float64).reshape(-1, 1)  # (n, 1) depth
        X1 = np.hstack([one, L])           # (n, 2) depth design
    if comp is not None:
        P = np.asarray(comp, dtype=np.float64)  # (n, K) composition block
        if P.ndim == 1:
            P = P[:, None]
        # Same finiteness rule as the depth control above: degrade the rung to
        # N/A rather than letting np.linalg.svd raise from inside ortho_basis.
        # NOTE: [1, one-hot block] is collinear by construction (the K columns
        # sum to the intercept); nothing is dropped here, core.ortho_basis
        # (core.py:331) absorbs the redundant direction via its SVD threshold
        # (core.py:358).
        if np.isfinite(P).all():
            X2a = np.hstack([one, P])      # (n, 1+K) composition design
            if X1 is not None:
                X2 = np.hstack([X1, P])    # (n, 2+K) joint design
    # ---- [3] Score the three residualised rungs [L31] -----------------------
    # WHAT: each rung is colcorr(resid(y, Q), resid(yh, Q)) with Q the
    # orthonormal basis of its design: a per-gene partial correlation with
    # BOTH sides projected off the control subspace.
    out["dg"] = _pr(y, yh, X1)             # (g,) or None: depth controlled
    out["crg"] = _pr(y, yh, X2a)           # (g,) or None: composition controlled
    out["drg"] = _pr(y, yh, X2)            # (g,) or None: both controlled
    # ---- [4] Ship the designs so floors/CIs reuse the exact same subspaces --
    # WHERE: popped as ``designs = rungs.pop("_designs")`` at audit.py:514
    # [L32], then fed to perm_null_of_rung (audit.py:525) and boot_of_rung via
    # core.boot_ci (audit.py:532-533).  Floor and CI must residualise on the
    # SAME design as the value, or they are the null/interval of a different
    # statistic.
    out["_designs"] = {"dg": X1, "crg": X2a, "drg": X2}
    return out


def _pr(y, yh, X):
    """partial_r that returns None (rung absent) instead of failing on X=None."""
    if X is None:
        return None
    # Q: orthonormal basis of the design's column space (core.ortho_basis,
    # core.py:331); resid(A, Q) = A - Q (Q^T A) removes from every gene column
    # the part the design explains.  Correlating the two residuals is the
    # textbook partial correlation, applied per gene.
    Q = ortho_basis(X)
    return colcorr(resid(y, Q), resid(yh, Q))


# ------------------------------------------------- floors and intervals
def perm_null_of_rung(y, yh, X, perm_idx, perm_mask=None, how="median"):
    """Null distribution of a residualised rung.

    The design X stays FIXED and only the residual spot order of y_hat is
    permuted, so the null carries no depth/composition structure while the
    control itself is left exactly as it was observed.  Permuting the design
    too would test a different (and weaker) hypothesis.

    Parameters
    ----------
    perm_idx : (n_perm, n) int, permutation plan from ``core.perm_plan``.
    perm_mask : (n_perm, n) bool or None, usable spots per draw.
    how : {'median', 'mean'}
        MUST match the aggregation of the value this floor will sit next to:
        the floor of the median rule is not the floor of the mean rule.

    Returns
    -------
    (n_perm,) ndarray of permuted rung values.
    """
    # ---- [1] Residualise both sides ONCE, outside the loop ------------------
    # WHAT: project truth and predictions off the design; X=None means the
    # r_full rung, where "residualising" is plain per-gene centring.
    # WHY: the design is held fixed under this null (only the spot order of
    # y_hat permutes), so ry and rh are identical across draws; redoing the
    # SVD per draw would cost n_perm extra decompositions for the same result.
    Q = ortho_basis(X) if X is not None else None
    ry = resid(y, Q) if Q is not None else (y - y.mean(0, keepdims=True))  # (n, g)
    rh = resid(yh, Q) if Q is not None else (yh - yh.mean(0, keepdims=True))  # (n, g)
    # ---- [2] One rung value per permutation draw ----------------------------
    # WHAT: correlate the fixed ry against rh in permuted spot order, then
    # collapse the (g,) per-gene r to a scalar with the SAME rule (median or
    # mean) the observed value uses.
    # WHERE: perm_idx/perm_mask are the shared plan [L19] from core.perm_plan
    # (audit.py:374, rebuilt on the held-out subset at audit.py:504); perm_mask
    # flags the spots usable in each draw (a torus-shifted block permutation
    # can leave some spots without a valid image).  The (B,) output becomes
    # out[k+"_perm"] (audit.py:525), out["r_full_perm_mean"] [L36]
    # (audit.py:530-531), the observed-DRG floor (audit.py:542), or the free
    # floor contrast [L37] (audit.py:584).
    out = np.empty(len(perm_idx))          # (B,) permuted rung values
    for b, pm in enumerate(perm_idx):
        if perm_mask is not None:
            m = perm_mask[b]               # (n,) usable spots of this draw
            r = colcorr(ry[m], rh[pm[m]])
        else:
            r = colcorr(ry, rh[pm])
        out[b] = (np.nanmedian(r) if how == "median" else np.nanmean(r)) \
            if np.isfinite(r).any() else np.nan
    return out


def boot_of_rung(y, yh, X, idx, how="median"):
    """One bootstrap draw of a rung, REDOING the residualisation.

    Re-running ``ortho_basis`` on the resampled rows is the point: reusing the
    full-sample basis would treat the control as known exactly and understate
    the interval.
    """
    # WHERE: called through core.boot_ci at audit.py:532-533; idx is the (n,)
    # resampled row index drawn by boot_ci from the bootstrap RNG stream
    # rng_boot [L18].  The scalar draws collect into the rung CI [L34]
    # (out[k+"_ci"], surfaced in the ladder table as *_ci_lo / *_ci_hi).
    if X is None:
        # r_full rung: no design, plain per-gene r on the resampled rows.
        r = colcorr(y[idx], yh[idx])
    else:
        # WHAT: rebuild the orthonormal basis on the RESAMPLED design rows and
        # residualise the resampled data on it (the point of the docstring).
        Q = ortho_basis(X[idx])
        r = colcorr(resid(y[idx], Q), resid(yh[idx], Q))
    if not np.isfinite(r).any():
        return np.nan
    # Collapse per-gene r with the same rule as the observed value, so the CI
    # is an interval for that value and not for a different summary.
    return float(np.nanmedian(r) if how == "median" else np.nanmean(r))
