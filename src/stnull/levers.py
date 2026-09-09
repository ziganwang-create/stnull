# -*- coding: utf-8 -*-
"""stnull.levers, protocol levers: r you can buy without changing the model.

Author: Zigan Wang.

WHAT THIS MODULE IS
    The price list.  Three of the four levers this package prices live here:
    picking the reported genes on the test set (``topn_value``), picking them
    legally on the train side (``train_selected_value``), and splitting by
    section instead of by patient (``null_transfer_leakage``).  The fourth,
    smoothing the ground truth, is priced in ``audit`` with the operator from
    ``nulls.smooth_matrix``.

WHO CALLS IT
    ``audit._section_block`` (per-section top-N values and their nulls) and
    ``audit()`` (the leakage lever and the assembled price table).

KEY ASSUMPTION
    Each lever is reported as (base value, levered value, delta) and each of
    the two values carries the permutation floor of ITS OWN readout rule.  The
    selection lever in particular is meaningless without ``perm_selection``:
    on one real dataset the top-50 readout is 0.325 while the floor of the
    same selection rule applied to permuted predictions is 0.080.

DATA LINEAGE (row numbers refer to docs/CODE_WALKTHROUGH.md (section 4, the variable lineage table))
    In:  per-gene r vectors born in audit._section_block: the observed
         r_full per gene and its (n_perm, g) permuted twin from
         audit._perm_pergene [L33], plus the train-side ranking key [L24].
         For the leakage lever: the full truth matrix Y [L01], the
         zero-pixel design Xl = [1, log lib, composition] assembled in
         audit() from loglib [L05] and Comp [L07] [L46], and the
         section/patient label vectors [L03]/[L09].
    Out: (value, null, n_used) triples stored per section as
         blocks[s]['topn'] [L45], aggregated by audit._combine [L39] into
         SelectionResult.by_n; train_selected_value fills the legal top-N
         column; the null_transfer_leakage DataFrame becomes leak_df and
         the leakage row of the lever price table [L46].
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .core import colcorr


# ------------------------------------------------------------ gene selection
def topn_value(r_vec, n_top, how="median"):
    """Median (or mean) r over the ``n_top`` genes ranked by r ON THE TEST SET.

    Parameters
    ----------
    r_vec : (g,) array-like of per-gene r; NaN genes are excluded from both the
        ranking and the count.
    n_top : int, how many genes to keep; clamped to the number that scored.
    how : {'median', 'mean'}

    Returns
    -------
    (value, n_used)

    Statistical premise
        This readout is BIASED UPWARD by construction: it selects on the same
        noise it then reports.  It is computed here precisely so the report
        can show it beside the floor of the same selection rule applied to
        permuted predictions, the amount pure selection noise buys.
    """
    # WHAT: rank every finite per-gene r and keep the best n_top.
    # WHY: this IS the biased readout being priced, kept verbatim; clamping
    # n_top means a request wider than the scored panel degrades into the
    # all-gene value instead of erroring.
    # WHERE: r_vec is the per-gene r computed in audit._section_block (the
    # observed rf, or one permuted row when called by topn_null) [L33]; the
    # (value, n_used) pair is stored in blocks[s]['topn'][N] [L45].
    r = np.asarray(r_vec, dtype=float)   # (g,) per-gene Pearson r, NaN allowed
    fin = r[np.isfinite(r)]              # genes that actually scored
    if fin.size == 0:
        return float("nan"), 0
    n_top = int(min(n_top, fin.size))    # effective N after clamping to the panel
    top = np.sort(fin)[::-1][:n_top]     # (n_top,) the selected best r values
    v = float(np.median(top)) if how == "median" else float(np.mean(top))
    return v, n_top


def topn_null(perm_r, n_top, how="median"):
    """Apply the SAME top-N selection rule to each permuted per-gene r vector.

    ``perm_r`` is (n_perm, g).  Selecting inside every permutation is the whole
    point: a floor computed on the all-gene rule would be far too low for a
    top-N value, and the comparison would flatter the model.
    """
    # WHAT: run topn_value once per permutation draw.
    # WHY: selecting inside every draw makes the floor the distribution of
    # the SAME biased rule under no signal at all; this is the
    # perm_selection floor the report prints beside each top-N value [L45].
    # WHERE: the rows of perm_r come from audit._perm_pergene [L33]; the
    # (n_perm,) result is stored as the null half of blocks[s]['topn'][N]
    # and funnelled through audit._combine [L39].
    out = np.empty(len(perm_r))          # (n_perm,) one levered value per draw
    for b, r in enumerate(perm_r):
        out[b] = topn_value(r, n_top, how)[0]
    return out


def train_selected_value(r_vec, rank_key, n_top, how="median"):
    """Legal readout: genes ranked by a TRAIN-side key, scored on the test set.

    Parameters
    ----------
    r_vec : (g,) per-gene r on the scored spots.
    rank_key : (g,) ranking statistic computed on TRAIN spots only.
    n_top : int
    how : {'median', 'mean'}

    Returns
    -------
    (value, n_used)

    Statistical premise
        Because the ranking never sees the scored spots, this readout is
        unbiased for the selected subset, it is what a paper reporting "our
        top 50 genes" is entitled to claim.  The gap between it and
        ``topn_value`` is the selection lever.
    """
    # WHAT: rank by the train-side key, with unrankable genes pushed to the
    # bottom, then score the chosen genes on the test side.
    # WHY: a gene with no train-side key must never win the ranking by NaN
    # accident; substituting -inf keeps it last without dropping it from r.
    # NaN test-side r then only shrinks the effective count (returned as
    # n_used) instead of poisoning the median.
    # WHERE: r_vec is the held-out per-gene r and rank_key is train_key from
    # audit._section_block [L24]; the pair (value, n_used) becomes the
    # 'topn_train' entry that audit() aggregates into
    # SelectionResult.train_selected.
    r = np.asarray(r_vec, dtype=float)     # (g,) test-side per-gene r
    key = np.asarray(rank_key, dtype=float)  # (g,) train-side ranking key [L24]
    ok = np.isfinite(key)                  # genes that can be ranked at all
    if ok.sum() == 0:
        return float("nan"), 0
    order = np.argsort(-np.where(ok, key, -np.inf))  # best train key first
    pick = order[:int(min(n_top, ok.sum()))]         # indices of the train-chosen top-N
    vals = r[pick]                         # their test-side r
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan"), 0
    v = float(np.median(vals)) if how == "median" else float(np.mean(vals))
    return v, int(vals.size)


# ---------------------------------------------------------- split granularity
def null_transfer_leakage(Y, X, section, patient, sections_order, how="median"):
    """Split-granularity lever measured on a ZERO-PIXEL null.

    Parameters
    ----------
    Y : (n_all, g) ndarray, every spot of every section, in the target space.
    X : (n_all, p) ndarray, the zero-pixel design ([1, log lib, composition]).
    section, patient : (n_all,) arrays of labels.
    sections_order : sequence, sections to report, in report order.
    how : {'median', 'mean'}, aggregation over genes.

    Returns
    -------
    DataFrame with one row per section: r fitted on the OTHER SECTIONS OF THE
    SAME PATIENT, r fitted on sections of OTHER PATIENTS, their difference,
    and n_spots.

    Statistical premise
        For each section the same null is fitted twice, once on what a
        section-level split leaks (same patient, other sections) and once on
        what a patient-level split allows, and the difference is the leakage
        premium.  This is not a re-training of the user's model; stnull never
        trains models.  It measures the premium available in the data to a
        predictor with no pixels at all, which is a LOWER BOUND on the premium
        a real model can collect.

    Not applicable when
        A patient contributes only one section (nothing to leak: the row stays
        NaN), or either training side has fewer than p+5 spots.
    """
    # WHAT: for each held-out section, define the two training sides the two
    # split granularities would allow.
    # WHY: the only thing that changes between the two fits is the split;
    # design, estimator and scoring stay identical, so the delta isolates
    # the split-granularity premium.
    # WHERE: section/patient are the coerced sec/pat vectors from audit()
    # [L03]/[L09]; sections_order is audit()'s fixed report order.
    rows = []
    section = np.asarray(section)
    patient = np.asarray(patient)
    for s in sections_order:
        m = section == s              # (n_all,) mask of the held-out section
        p = patient[m][0]             # its patient (each section has one patient)
        same = (patient == p) & (~m)  # what a section-level split leaks: same patient, other sections
        other = patient != p          # what a patient-level split allows: other patients only
        rec = {"section": s, "patient": p,
               "r_same_patient": np.nan, "r_other_patient": np.nan,
               "delta": np.nan, "n_spots": int(m.sum())}
        # WHAT: fit the same least-squares null on each training side and
        # score it on the held-out section.
        # WHY: a side too small to fit (fewer than p+5 rows) or a section
        # too small to score (under 3 spots) stays NaN rather than
        # returning a garbage r; NaN rows are simply not counted in the
        # median delta upstream.
        # WHERE: X is the Xl design assembled in audit() [L46]; colcorr is
        # core.colcorr, the same per-gene Pearson r used by every null
        # [L25]; the finished frame is read back in audit() as leak_df.
        for key, tr in (("r_same_patient", same), ("r_other_patient", other)):
            if tr.sum() < X.shape[1] + 5 or m.sum() < 3:
                continue
            B, *_ = np.linalg.lstsq(X[tr], Y[tr], rcond=None)  # (p, g) OLS coefficients, one column per gene
            r = colcorr(Y[m], X[m] @ B)  # (g,) per-gene r of the transferred null on the held-out section
            if np.isfinite(r).any():
                rec[key] = float(np.nanmedian(r) if how == "median"
                                 else np.nanmean(r))
        rec["delta"] = rec["r_same_patient"] - rec["r_other_patient"]
        rows.append(rec)
    return pd.DataFrame(rows)
