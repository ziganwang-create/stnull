# -*- coding: utf-8 -*-
"""Regression tests for the six FIX_FIRST issues (HANDOFF 2026-08-21 §一.6).

1. micron/perm degradation is recorded in the ledger, not only per section
2. default top_n adapts to a <250-gene panel instead of crashing under strict
3. null_fit='oracle' is flagged UPPER BOUNDS in the degradation ledger
4. null_fit='train_only' without is_train raises (strict) / degrades (lenient)
   and the no-is_train ledger sentence no longer prints a false null_fit
5. duplicate coordinates: perm_plan falls back to free, audit discloses it
6. single-column lattice: block permutation never returns the identity
"""
import numpy as np
import pytest

import stnull
from stnull.core import BadInput, perm_plan


def _toy(n=400, G=60, seed=0, coords="grid"):
    rng = np.random.default_rng(seed)
    y = rng.poisson(2.0, (n, G)).astype(float)
    pred = np.log1p(y) + rng.normal(0, 0.5, y.shape)
    if coords == "grid":
        side = int(np.ceil(np.sqrt(n)))
        xy = np.stack(np.divmod(np.arange(n), side), 1).astype(float)
    else:
        xy = rng.uniform(0, 100, (n, 2))
    sec = np.array(["S1"] * n)
    return y, pred, xy, sec


def _audit(y, pred, xy, sec, **kw):
    kw.setdefault("space", "log1p_cp10k:panel")
    kw.setdefault("space_note", "test")
    kw.setdefault("n_perm", 20)
    kw.setdefault("n_boot", 0)
    kw.setdefault("verbose", False)
    return stnull.audit(y, pred, section=sec, coords=xy, **kw)


# ---------------------------------------------------------------- issue 2
def test_default_topn_survives_small_panel_under_strict():
    y, pred, xy, sec = _toy(G=50)                      # HEST-style 50-gene panel
    rep = _audit(y, pred, xy, sec, strict=True)        # must NOT raise
    assert any("top_n" in w for w in rep.run.degradations + list(rep.warnings)), \
        "the trim must be disclosed"


def test_explicit_overwide_topn_still_raises_under_strict():
    y, pred, xy, sec = _toy(G=50)
    with pytest.raises(BadInput):
        _audit(y, pred, xy, sec, strict=True, top_n=(10, 250))


# ---------------------------------------------------------------- issue 4
def test_train_only_without_is_train_raises_strict():
    y, pred, xy, sec = _toy()
    with pytest.raises(BadInput):
        _audit(y, pred, xy, sec, strict=True, null_fit="train_only",
               lib_size=y.sum(1))


def test_train_only_without_is_train_degrades_lenient():
    y, pred, xy, sec = _toy()
    rep = _audit(y, pred, xy, sec, strict=False, null_fit="train_only",
                 lib_size=y.sum(1))
    assert rep.run.null_fit == "cv_within_section"
    joined = " ".join(rep.run.degradations)
    assert "cv_within_section" in joined
    assert "null_fit=train_only" not in joined, \
        "the ledger must not claim cross-fitting under a train_only label"


# ---------------------------------------------------------------- issue 3
def test_oracle_null_fit_is_flagged_upper_bound():
    y, pred, xy, sec = _toy()
    rep = _audit(y, pred, xy, sec, null_fit="oracle", lib_size=y.sum(1))
    assert any("UPPER BOUND" in d for d in rep.run.degradations)


# ---------------------------------------------------------------- issue 5
def test_perm_plan_duplicate_cells_falls_back_to_free():
    rng = np.random.default_rng(0)
    n = 100
    xy = np.stack(np.divmod(np.arange(n), 10), 1).astype(float)
    xy[1] = xy[0]                                       # one duplicated cell
    idx, msk = perm_plan(n, 10, "block", xy, "grid", rng)
    assert msk is None, "duplicates must force the free fallback"
    for row in idx:                                     # free perms are bijections
        assert len(np.unique(row)) == n


def test_audit_discloses_duplicate_coords():
    y, pred, xy, sec = _toy()
    xy[1] = xy[0]
    rep = _audit(y, pred, xy, sec, strict=False)
    assert any("duplicate coordinates" in d for d in rep.run.degradations)


# ---------------------------------------------------------------- issue 6
def test_single_column_block_perm_is_never_identity():
    rng = np.random.default_rng(0)
    n = 60
    xy = np.stack([np.zeros(n), np.arange(n, dtype=float)], 1)   # one column
    idx, msk = perm_plan(n, 50, "block", xy, "grid", rng)
    assert msk is not None, "a 1xH lattice can be torus-shifted along H"
    ident = np.arange(n)
    for b, row in enumerate(idx):
        use = msk[b]
        assert use.sum() > 0
        assert not np.array_equal(row[use], ident[use]), \
            "block permutation returned the identity on a single-column grid"


# ---------------------------------------------------------------- issue 1
def test_block_fallback_reaches_the_ledger():
    # scattered micron coords whose pseudo-grid embedding is sparse enough to
    # push the torus coverage under 50% -> block must fall back AND say so
    y, pred, xy, sec = _toy(coords="micron")
    rep = _audit(y, pred, xy, sec, strict=False, coord_kind="micron",
                 perm_kind="block")
    joined = " ".join(rep.run.degradations)
    fell_back = "fell back to FREE" in joined
    stayed_block = all(
        b == "perm_block"
        for b in rep.ladder.get("perm_kind_used", [])
    ) if "perm_kind_used" in getattr(rep.ladder, "columns", []) else None
    assert fell_back or ("coord_kind='micron'" in joined), \
        "micron degradation must be visible in the ledger one way or the other"
