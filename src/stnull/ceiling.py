# -*- coding: utf-8 -*-
"""stnull.ceiling: the measurement ceiling r_tech.

Author: Zigan Wang.

WHAT THIS MODULE IS
    The answer to "how high could this metric go at all?".  The evaluation
    correlates predictions with a NOISY measurement y_obs, so no predictor --
    not even one that outputs the true expression t exactly, can beat
    corr(t, y_obs) = sqrt(reliability).  That bound is r_tech.

WHO CALLS IT
    ``audit()``, once per section, only when raw ``counts`` are supplied.  The
    section values are then combined like any other readout and reported in
    section 6 of the report, together with ``r / r_tech``.

THE ESTIMATOR (the only one; needs raw counts)
    1. Binomial(count, 1/2) thinning of the observed counts into two
       INDEPENDENT half-depth counting replicates.  When the full-transcriptome
       library size is given, the 'rest of transcriptome' pseudo-gene is thinned
       too, so full-library denominators stay honest at half depth (method
       'split_half_full'); otherwise panel row sums are used
       ('split_half_panel').
    2. reliability = Spearman-Brown( corr(half1, half2) ), per gene: the
       replicate-replicate correlation corrected back to full depth.
    3. r_tech = sqrt(max(reliability, 0)).

KEY ASSUMPTIONS AND THEIR LIMITS
    * Binomial thinning of a Poisson/negative-binomial count gives two halves
      that are independent GIVEN the latent rate, which is exactly what a
      reliability estimate needs.
    * Spearman-Brown assumes the two halves are parallel tests on a linear
      scale.  log1p is not linear, so at ST-typical sparsity (mean count per
      spot-gene below ~5) the estimate runs high, measured +0.06 to +0.11
      against a Monte-Carlo Bayes-optimal bound on Poisson simulations.  The
      note attached to the result says so whenever the data are that sparse.
    * r_tech bounds COUNTING noise only.  Biological, registration and
      annotation limits are additional and are not measured here.

Zero-signal limit of this estimator is 0 (measured 0.003 on iid-Poisson
synthetic data).  The 0.1.0 estimator ('multinomial_resample': correlate the
observed profile with a replicate RESAMPLED FROM the observed profile) shared
the observation's own sampling noise with the replicate, so its zero-signal
limit was 1/sqrt(2) = 0.707, it reported ceilings around 0.79 on data whose
corrected ceiling is 0.51.  It was removed in 0.2.0, not merely demoted.

If counts are absent, r_tech is N/A.  stnull deliberately does not fall back
to a Poisson approximation: that approximation would be a fabricated number.

DATA LINEAGE (row numbers cite docs/CODE_WALKTHROUGH.md (section 4, the variable lineage table))
    Inputs, all arriving through audit()'s ceiling loop (audit.py:1385-1391):
      counts    (n, g) raw UMI counts [L08], the user's `counts` argument
                row-sliced per section at audit.py:1386.  The only stnull
                input that can price counting noise; without it this whole
                module is skipped and r_tech is N/A.
      space     TargetSpace enum member [L12], from the user's REQUIRED
                `space` argument via spaces.coerce_space (spaces.py:59).
      lib_full  (n,) full-transcriptome library size [L14]; audit.py:1387
                substitutes lib_size [L04] when lib_size_full is absent
                (`lf = lib_size_full or lib`), sliced at audit.py:1388.
    Output:
      ``r_tech`` returns (r_per_gene, perm_null, method, note), the [L47]
      chain: audit.py:1397-1401 aggregates the (g,) vector to a section
      scalar and collects the per-section lists, audit.py:1407-1411 funnels
      them through _combine/make_stat [L39] into CeilingResult.r_tech, and
      audit.py:1413-1421 forms the ratio r / r_tech (r_over_ceiling).
      ``apply_space`` is also imported by nothing else: it exists only to put
      the thinned replicates into the user's declared space.

    Terms (plain language; full glossary in CODE_WALKTHROUGH.md)
      A "spot" is one measured location on the tissue slide; a "UMI" is one
      uniquely tagged mRNA molecule, so raw counts are molecule counts.
      "Library size" is a spot's total UMI count (its sequencing depth).
      "Thinning" a count means splitting it binomially, like dealing each
      molecule to one of two half-depth decks at random.  "Reliability" is
      the fraction of a measurement's variance that is signal rather than
      noise; correlating a noiseless predictor with a noisy target caps at
      sqrt(reliability), which is the ceiling this module estimates.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .core import colcorr
from .spaces import TargetSpace

#: Below this mean count per spot-gene, the Spearman-Brown step is biased
#: upward by roughly +0.06 to +0.11 (simulation, see module docstring) and the
#: result note says so.  Most Visium / HER2ST panels sit well below it.
SPARSE_MEAN_COUNT = 5.0


def _dense(counts):
    """Counts as a dense float array; sparse input is materialised here."""
    # WHERE: the only caller is r_tech below; counts is the per-section slice
    # of the user's raw count matrix [L08].  This is the single place in the
    # ceiling path where a scipy sparse matrix becomes dense, one section at a
    # time, so peak memory is one section's counts, not the whole run's.
    if sp.issparse(counts):
        return np.asarray(counts.todense(), dtype=np.float64)
    return np.asarray(counts, dtype=np.float64)


# ------------------------------------------------------------- target spaces
def apply_space(panel_counts, space, lib_full=None, ref=None):
    """Put raw panel counts into the declared target space.

    Parameters
    ----------
    panel_counts : (n, g) array-like of counts (already dense).
    space : TargetSpace
    lib_full : (n,) array-like or None
        Full-transcriptome library size; used only by the spaces whose
        denominator is not the panel row sum.
    ref : (mu, sd) or None
        Reference moments for ZSCORE; None means standardise in place.

    Returns
    -------
    (n, g) ndarray, or None for ``CUSTOM``, a transform stnull was not told
    about cannot be replicated, and guessing it would fabricate the ceiling.

    Only used to build the measurement replicate; user data is never
    transformed by stnull (decision D2).
    """
    # WHERE: called only from r_tech below, once per thinning replicate and
    # once as a 2-row probe; `space` is the [L12] enum, `lib_full` the half
    # library sizes derived from [L14]/[L08] inside r_tech.
    C = np.asarray(panel_counts, dtype=np.float64)  # (n, g) counts to transform
    # WHAT: log1p of counts-per-10k with the PANEL row sum as denominator.
    # WHY the maximum(.., 1.0): an all-zero spot would otherwise divide by 0.
    if space is TargetSpace.LOG1P_CP10K_PANEL:
        den = np.maximum(C.sum(1, keepdims=True), 1.0)  # (n, 1) panel depth
        return np.log1p(1e4 * C / den)
    # WHAT: same transform, but the denominator is the FULL-transcriptome
    # library size, the honest CP10K when the panel is a small subset.
    if space is TargetSpace.LOG1P_CP10K_FULL:
        den = np.maximum(np.asarray(lib_full, float).reshape(-1, 1), 1.0)
        return np.log1p(1e4 * C / den)
    # WHAT: plain log1p of raw counts, no depth normalisation at all.
    if space is TargetSpace.LOG1P_RAW:
        return np.log1p(C)
    # WHAT: the ST-Net-family space, counts scaled to the MEDIAN library size
    # of the section, then log10(1 + .).  Falls back to panel sums when no
    # full library size is available.
    if space is TargetSpace.LOG10_MEDLIB:
        lib = (np.asarray(lib_full, float).reshape(-1, 1) if lib_full is not None
               else np.maximum(C.sum(1, keepdims=True), 1.0))
        med = float(np.median(np.maximum(lib, 1.0)))  # scalar reference depth
        return np.log10(1.0 + C / np.maximum(lib, 1.0) * med)
    # WHAT: per-gene z-score of log1p-CP10K.  With ref=None the moments come
    # from the replicate itself; passing (mu, sd) would standardise against a
    # fixed reference instead.  Constant genes get sd=1 so they map to 0
    # rather than NaN.
    if space is TargetSpace.ZSCORE:
        den = np.maximum(C.sum(1, keepdims=True), 1.0)
        Z = np.log1p(1e4 * C / den)                   # (n, g) log1p-CP10K
        mu = Z.mean(0, keepdims=True) if ref is None else ref[0]
        sd = Z.std(0, keepdims=True) if ref is None else ref[1]
        sd = np.where(sd <= 0, 1.0, sd)
        return (Z - mu) / sd
    # WHERE: the None return is what r_tech's 2-row probe checks; it turns
    # into method='na' with a reason, never into a silent guess.
    return None  # CUSTOM -> cannot replicate the user's transform


def _sb_sqrt(rh):
    """half-half r -> Spearman-Brown reliability -> ceiling sqrt(reliability).

    The clip before the sqrt is what keeps a negative half-half correlation
    (pure noise, and half the time slightly negative) from producing a NaN or
    a complex number; it maps to a ceiling of 0, which is the honest reading.
    """
    # WHAT: Spearman-Brown prophecy formula for doubling test length: the two
    # halves each carry half the depth, so the full-depth reliability is
    # 2r/(1+r).  WHERE: called only from r_tech, once on the replicate-mean
    # half-half r (the value) and once per permutation draw (the floor).
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = 2.0 * rh / (1.0 + rh)    # (g,) estimated full-depth reliability
    rel = np.clip(rel, -1.0, 1.0)
    return np.sqrt(np.clip(rel, 0.0, 1.0))  # (g,) ceiling = sqrt(reliability)


# ------------------------------------------------------------------- r_tech
def r_tech(counts, space, lib_full=None, n_rep=20, rng=None, n_perm=50):
    """Split-half measurement ceiling of one section.

    Parameters
    ----------
    counts : (n, g) raw integer counts, dense or scipy sparse.
    space : TargetSpace
        The space the user's y_true lives in; the replicate is transformed the
        same way, otherwise the ceiling would not bound the reported metric.
    lib_full : (n,) array-like or None
        Full-transcriptome library size in counts.  When given, the
        rest-of-transcriptome column is thinned alongside the panel so that
        half-depth denominators are honest.
    n_rep : int
        Independent thinning replicates; the RAW half-half r is averaged over
        them before the Spearman-Brown step.
    rng : numpy Generator or None
    n_perm : int
        Permutation draws for the floor of this same readout rule.

    Returns
    -------
    (r_per_gene, perm_null_of_median, method, note).  ``r_per_gene`` is None
    and ``method`` is ``'na'`` when the ceiling cannot be estimated at all.

    Statistical premise
        Averaging the RAW half-half r across replicates and transforming ONCE
        at the end is deliberate: sqrt(clip(.)) is non-negative and very steep
        near zero, so averaging transformed replicates would inflate the
        zero-signal readout from ~0 to ~0.11.  The permutation floor uses the
        identical rule (same replicates, same averaging, same transform, spot
        order of the second half permuted), because a floor built from one
        replicate would sit ~38% too high.

    Not applicable when
        ``space`` is CUSTOM (the transform cannot be replicated), or
        ``lib_full`` is non-finite.  Both return method='na' with a reason,
        and audit() counts the skipped section instead of dropping it silently.
    """
    # ---- [1] Coerce inputs and refuse what cannot be estimated --------------
    # WHERE: audit.py:1389 calls this once per section with the counts slice
    # [L08], the declared space [L12] and the (fallback-resolved) full library
    # size [L14]; the four return values start the [L47] chain.
    rng = np.random.default_rng(0) if rng is None else rng
    C = _dense(counts)                 # (n, g) raw counts, densified
    n, G = C.shape                     # spots, genes in this section
    # WHAT: a 2-row probe of apply_space.  WHY: CUSTOM returns None, and it is
    # cheaper to learn that on 2 rows than after 20 thinning replicates.
    if apply_space(C[:2], space, lib_full[:2] if lib_full is not None else None) is None:
        return None, None, "na", "space='custom': the transform cannot be replicated"
    # Ci: counts as non-negative int64, the form rng.binomial requires.
    # Rounding tolerates counts stored as floats; a truly non-integer matrix
    # was already flagged by check_inputs (audit.py:239-249).
    Ci = np.rint(np.maximum(C, 0.0)).astype(np.int64)
    # ---- [2] Decide the thinning method -------------------------------------
    # WHAT: with a full library size, also thin the molecules OUTSIDE the
    # panel ('rest of transcriptome' pseudo-gene) so that each half's CP10K
    # denominator is a real half-depth library ('split_half_full'); without
    # it, denominators fall back to panel row sums ('split_half_panel').
    # WHERE: lib_full is [L14], already sliced to this section (audit.py:1388).
    if lib_full is not None:
        L = np.asarray(lib_full, float).ravel()  # (n,) full library size
        if not np.isfinite(L).all():
            # np.rint(nan).astype(int64) is INT64_MIN, which reaches
            # rng.binomial as 'n < 0', an error naming neither the input nor
            # the section.  Report it as a skipped section instead.
            return (None, None, "na",
                    "lib_size_full contains %d non-finite value(s) in this "
                    "section: cannot thin the rest-of-transcriptome column"
                    % int((~np.isfinite(L)).sum()))
        method = "split_half_full"
        # WHY the maximum: a user-supplied lib_full below the panel sum is
        # impossible (the panel is a subset of the transcriptome); flooring at
        # the panel sum keeps the rest column non-negative.
        L = np.maximum(L, C.sum(1))
        rest = np.rint(np.maximum(L - C.sum(1), 0.0)).astype(np.int64)  # (n,)
    else:
        method = "split_half_panel"
        rest = None

    # ---- [3] Accumulators for the replicate average -------------------------
    # WHY sums and counts instead of a list: genes can be scorable in some
    # replicates and not others (a half with zero variance gives NaN), so each
    # gene averages over exactly the replicates that scored it.
    acc = np.zeros(G)          # sum of raw half-half r over replicates
    cnt = np.zeros(G)          # replicates that scored each gene
    perm_pm = None             # permutations, drawn ONCE and reused
    perm_acc = perm_cnt = None
    # ---- [4] Thinning replicates --------------------------------------------
    # WHAT: each replicate deals every molecule of every (spot, gene) cell to
    # half 1 or half 2 with probability 1/2; H1 + H2 == Ci exactly, and given
    # the latent expression rate the two halves are independent, which is the
    # property a reliability estimate needs.
    for rep in range(int(n_rep)):
        H1 = rng.binomial(Ci, 0.5).astype(np.float64)  # (n, g) half 1 counts
        H2 = Ci - H1                                   # (n, g) half 2 counts
        # WHAT: build each half's library size.  With the rest column, the
        # off-panel molecules are dealt too, so l1/l2 are honest half-depth
        # FULL libraries; otherwise they are panel row sums, matching what
        # the panel-denominator spaces would see.
        if rest is not None:
            R1 = rng.binomial(rest, 0.5).astype(np.float64)  # (n,) rest, half 1
            l1 = H1.sum(1) + R1                              # (n,) half-1 depth
            l2 = H2.sum(1) + (rest - R1)                     # (n,) half-2 depth
        else:
            l1 = H1.sum(1)
            l2 = H2.sum(1)
        # WHAT: put both halves into the user's declared target space [L12],
        # the same transform y_true lives in, otherwise the ceiling would
        # bound a different metric than the one the audit reports.
        Y1 = apply_space(H1, space, l1)    # (n, g) half 1 in target space
        Y2 = apply_space(H2, space, l2)    # (n, g) half 2 in target space
        # accumulate the RAW half-half r and transform only once at the end:
        # sqrt(clip(.)) is non-negative, so averaging transformed replicates
        # would inflate the zero-signal readout from ~0 to ~0.11
        rh = colcorr(Y1, Y2)               # (g,) raw half-half Pearson r
        ok = np.isfinite(rh)
        acc[ok] += rh[ok]
        cnt[ok] += 1
        # ---- [5] The floor of the same rule ---------------------------------
        # WHAT: score every permutation against every replicate, accumulating
        # exactly like the value does (raw r summed, transformed once at the
        # end in section [7]).
        if rep == 0 and n_perm > 0:
            # Drawn here, at this exact point in the RNG stream, and then
            # reused for every replicate: the floor has to average over the
            # same replicates the value averages over, or it is the null of a
            # different rule.
            perm_pm = [rng.permutation(n) for _ in range(int(n_perm))]  # B x (n,)
            perm_acc = np.zeros((len(perm_pm), G))  # (B, g) summed permuted r
            perm_cnt = np.zeros((len(perm_pm), G))  # (B, g) scoring replicates
        if perm_pm is not None:
            for j, pm in enumerate(perm_pm):
                # WHY permute only the second half's spot order: that breaks
                # the spot pairing (the signal) while keeping both marginal
                # distributions and the thinning noise exactly as observed.
                rp = colcorr(Y1, Y2[pm])   # (g,) permuted half-half r
                okp = np.isfinite(rp)
                perm_acc[j, okp] += rp[okp]
                perm_cnt[j, okp] += 1

    # ---- [6] The ceiling: average the RAW r, transform ONCE -----------------
    # WHAT: per gene, mean half-half r over its scoring replicates, then the
    # single Spearman-Brown + sqrt step.  WHY the order matters: sqrt(clip(.))
    # is non-negative and steep near zero, so transforming each replicate and
    # averaging afterwards would inflate the zero-signal readout from ~0 to
    # ~0.11 (the docstring's statistical premise).
    out = np.full(G, np.nan)           # (g,) per-gene ceiling r_tech
    nz = cnt > 0                       # genes scored by at least one replicate
    out[nz] = _sb_sqrt(acc[nz] / cnt[nz])

    # ---- [7] Collapse each permutation draw with the identical rule ---------
    # WHAT: per draw, the same average-then-transform, then the median over
    # genes, mirroring how audit.py aggregates the value (agg(..., 'median')).
    # WHERE: this (B,) vector is the perm_null return; audit.py:1399 collects
    # it into rt_perms and _combine (audit.py:1407-1411) turns the per-draw
    # medians into the floor of CeilingResult.r_tech [L47].
    perm_vals = None
    if perm_acc is not None:
        perm_vals = np.full(len(perm_acc), np.nan)  # (B,) permuted medians
        for j in range(len(perm_acc)):
            m = perm_cnt[j] > 0
            if not m.any():
                continue
            v = _sb_sqrt(perm_acc[j][m] / perm_cnt[j][m])
            if np.isfinite(v).any():
                perm_vals[j] = float(np.nanmedian(v))

    # ---- [8] Sparsity caveat and return -------------------------------------
    # WHAT: attach the measured bias warning whenever the section is sparser
    # than SPARSE_MEAN_COUNT; the note travels verbatim into the report.
    mean_count = float(Ci.mean()) if Ci.size else float("nan")  # counts per cell
    note = ("%d replicates; binomial thinning + Spearman-Brown; "
            "r_tech = sqrt(reliability)" % n_rep)
    if np.isfinite(mean_count) and mean_count < SPARSE_MEAN_COUNT:
        # Documented, measured limitation rather than a hidden one: at this
        # sparsity the SB correction assumes a linearity that log1p breaks.
        note += ("; mean count per spot-gene = %.2f (< %.0f): Spearman-Brown "
                 "assumes parallel tests on a linear scale, which log1p breaks "
                 "at this sparsity, so r_tech runs about +0.06 to +0.11 HIGH "
                 "here (Poisson simulation) -- read it as an approximate bound"
                 % (mean_count, SPARSE_MEAN_COUNT))
    # WHERE: (out, perm_vals, method, note) is unpacked at audit.py:1389-1391
    # as (r, pnull, method, note): the start of the [L47] chain described in
    # the module docstring's DATA LINEAGE section.
    return out, perm_vals, method, note
