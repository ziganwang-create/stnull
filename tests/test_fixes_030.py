# -*- coding: utf-8 -*-
"""Regression tests for the 0.3.0 audit-review fixes.

Author: Zigan Wang.

One test per fixed defect, each written so that it FAILS on 0.2.0:

 1. train_only scored the nulls out-of-sample and the model in-sample
 2. depth_proxy='observed' printed corr(l_hat, log lib) = 1.0 as evidence
    and depth_1d as an 'image -> 1 number' readout
 3. null_fit / depth_proxy / perm_kind / fit_mode were not validated
 4. depth_proxy='given' without lib_pred silently switched the depth rungs off
 5. non-finite lib_size / coords / composition / lib_pred / lib_size_full
    passed validation and crashed inside numpy or scipy
 6. lib_size_full was never validated at all
 7. sparse y_true / y_pred crashed check_inputs
 8. every null reported the MODEL's gene count (0 in nulls-only mode)
 9. headline.r_mean carried the floor of the median rule
10. share_depth exploded on a near-zero r_full
11. a null that could not be fitted vanished without a reason
12. the composition oracle saturated at r = 1.0 on a wide design
13. smooth_matrix was standardised twice, so the 3x3 lever was understated
14. the r_tech floor was built from one replicate while the value averaged all
15. a rejected from_pred depth proxy was republished against a zero floor
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "3")

import numpy as np
import pytest
import scipy.sparse as sps

import stnull
from stnull import audit, check_inputs, nulls_only
from stnull.core import BadInput, InsufficientData, perm_plan


# ------------------------------------------------------------------ fixtures
def toy(n_sec=2, side=12, G=30, seed=0):
    """Two lattice sections with a depth + 2-class composition signal."""
    rng = np.random.default_rng(seed)
    Ys, Ps, secs, libs, xy, labs = [], [], [], [], [], []
    for s in range(n_sec):
        gx, gy = np.meshgrid(np.arange(side), np.arange(side))
        c = np.stack([gx.ravel(), gy.ravel()], 1).astype(float)
        n = c.shape[0]
        lib = np.exp(rng.normal(8.5, 0.6, n))
        lab = np.where(c[:, 0] < side / 2, "tumour", "stroma")
        load = np.log(lib) - np.log(lib).mean()
        theme = (lab == "tumour").astype(float)
        B = rng.normal(0, 1, (2, G))
        base = rng.normal(0, 1, (n, G)) * 0.9
        Y = base + np.outer(load, B[0]) * 0.7 + np.outer(theme, B[1]) * 0.5
        P = (np.outer(load, B[0]) * 0.6 + np.outer(theme, B[1]) * 0.4
             + base * 0.15 + rng.normal(0, .4, (n, G)))
        Ys.append(Y)
        Ps.append(P)
        secs.append(np.array(["S%d" % s] * n))
        libs.append(lib)
        xy.append(c)
        labs.append(lab)
    return (np.vstack(Ys), np.vstack(Ps), np.concatenate(secs),
            np.concatenate(libs), np.vstack(xy), np.concatenate(labs))


KW = dict(space="zscore:per_gene", n_perm=10, n_boot=0, top_n=(5,),
          min_spots=50, verbose=False)


# ------------------------------------------------------------------ fix 1
def test_train_only_scores_the_model_on_the_held_out_spots():
    """A model that memorises the train spots must not win on the headline.

    0.2.0 scored the nulls on ~is_train (nulls.py `scored = ~tr`) but the model
    on ALL spots, so a memorising predictor reported r ~ 0.75 against a null of
    ~0.03.  The model side is now restricted to the same held-out spots.
    """
    Y, P, sec, lib, xy, lab = toy()
    rng = np.random.default_rng(3)
    istr = rng.random(len(sec)) < 0.75
    Pm = np.where(istr[:, None], Y, rng.normal(size=Y.shape))   # memorise train
    rep = audit(Y, Pm, section=sec, lib_size=lib, coords=xy, labels=lab,
                is_train=istr, null_fit="train_only", **KW)
    assert abs(rep.headline.r_median.value) < 0.10, \
        "the model was scored on spots it memorised: %.4f" % rep.headline.r_median.value
    assert any("held-out" in d for d in rep.run.degradations)
    assert "n_eval" in rep.per_section.columns
    assert (rep.per_section["n_eval"] < rep.per_section["n_spots"]).all()


# ------------------------------------------------------------------ fix 2
def test_observed_depth_proxy_does_not_publish_a_tautology():
    """corr(l_hat, log lib) is 1 by construction under depth_proxy='observed'."""
    Y, P, sec, lib, xy, lab = toy()
    rep = audit(Y, P, section=sec, lib_size=lib, coords=xy, labels=lab, **KW)
    drp = rep.depth.depth_r_of_pred
    assert not np.isfinite(drp.value)
    assert drp.above_floor() is None
    assert "by construction" in drp.note
    # depth_1d under 'observed' would be depth_null on another fold split
    assert rep.depth.depth_1d is None or not np.isfinite(rep.depth.depth_1d.value)
    assert not np.isfinite(rep.per_section["depth_1d"].astype(float)).any()


# ------------------------------------------------------------------ fix 3
def test_unknown_options_raise_instead_of_silently_switching():
    Y, P, sec, lib = toy()[:4]
    for kw in (dict(null_fit="orcale"), dict(null_fit="train"),
               dict(depth_proxy="obsevred"), dict(perm_kind="blok")):
        with pytest.raises(BadInput):
            audit(Y, P, section=sec, lib_size=lib, **dict(KW, **kw))


def test_linear_null_pred_rejects_an_unknown_fit_mode():
    from stnull.nulls import linear_null_pred
    Y = np.random.default_rng(0).normal(size=(60, 5))
    X = np.c_[np.ones(60), np.arange(60.0)]
    with pytest.raises(BadInput):
        linear_null_pred(Y, X, fit_mode="cv_within_sections")


# ------------------------------------------------------------------ fix 4
def test_depth_proxy_given_without_lib_pred_is_explicit():
    Y, P, sec, lib = toy()[:4]
    with pytest.raises(BadInput):
        audit(Y, P, section=sec, lib_size=lib, depth_proxy="given", **KW)
    rep = audit(Y, P, section=sec, lib_size=lib, depth_proxy="given",
                strict=False, **KW)
    assert rep.run.depth_proxy == "observed"
    assert any("lib_pred" in d for d in rep.run.degradations)


# ------------------------------------------------------------------ fix 5/6
@pytest.mark.parametrize("field", ["lib_size", "coords", "composition",
                                   "lib_pred", "lib_size_full"])
def test_non_finite_inputs_are_named_by_check_inputs(field):
    Y, P, sec, lib, xy, lab = toy()
    kw = dict(y_true=Y, y_pred=P, section=sec, min_spots=50)
    comp = np.c_[(lab == "tumour").astype(float), (lab == "stroma").astype(float)]
    payload = {"lib_size": lib.copy(), "coords": xy.copy(),
               "composition": comp, "lib_pred": np.log(lib),
               "lib_size_full": lib.copy() * 3.0}[field]
    payload = np.array(payload, dtype=float)
    payload.reshape(-1)[7] = np.nan
    kw[field] = payload
    rep = check_inputs(**kw)
    assert not rep.ok
    assert any(field in e and "non-finite" in e for e in rep.errors), rep.errors
    with pytest.raises(InsufficientData):
        audit(Y, P, section=sec, **dict(KW, **{field: payload}))


def test_non_finite_lib_size_degrades_instead_of_crashing():
    """strict=False must reach a report, not LinAlgError from inside lstsq."""
    Y, P, sec, lib, xy, lab = toy()
    bad = lib.copy()
    bad[7] = np.nan
    rep = audit(Y, P, section=sec, lib_size=bad, labels=lab, strict=False, **KW)
    assert any("non-finite" in w for w in rep.warnings)
    # the affected section drops out of the depth null and says why; the clean
    # section still reports, so the cohort median stays finite
    ps = rep.per_section.set_index("section")
    assert not np.isfinite(float(ps.loc["S0", "depth_null"]))
    assert np.isfinite(float(ps.loc["S1", "depth_null"]))
    assert any("non-finite" in d for d in rep.run.degradations)


def test_ladder_rungs_survive_a_non_finite_composition():
    from stnull.drg import ladder_rungs
    rng = np.random.default_rng(0)
    y = rng.normal(size=(80, 6))
    yh = y * 0.5 + rng.normal(size=y.shape)
    comp = rng.normal(size=(80, 3))
    comp[5, 1] = np.nan
    rungs = ladder_rungs(y, yh, None, comp)          # must not raise
    assert rungs["crg"] is None and rungs["drg"] is None


def test_perm_plan_falls_back_to_free_on_non_finite_coords():
    xy = np.stack(np.divmod(np.arange(100), 10), 1).astype(float)
    xy[3, 0] = np.nan
    idx, msk = perm_plan(100, 5, "block", xy, "grid", np.random.default_rng(0))
    assert msk is None
    for row in idx:
        assert len(np.unique(row)) == 100


# ------------------------------------------------------------------ fix 7
def test_check_inputs_accepts_sparse_matrices():
    Y, P, sec = toy()[:3]
    rep = check_inputs(y_true=sps.csr_matrix(Y), y_pred=sps.csr_matrix(P),
                       section=sec, min_spots=50)
    assert rep.ok, rep.errors
    rep2 = audit(sps.csr_matrix(Y), sps.csr_matrix(P), section=sec, **KW)
    assert np.isfinite(rep2.headline.r_median.value)


# ------------------------------------------------------------------ fix 8
def test_every_null_counts_its_own_genes():
    Y, sec, lib, xy, lab = toy()[0], toy()[2], toy()[3], toy()[4], toy()[5]
    rep = nulls_only(Y, space="zscore:per_gene", section=sec, lib_size=lib,
                     coords=xy, labels=lab, n_perm=10, n_boot=0, min_spots=50,
                     verbose=False)
    assert rep.headline is None
    for st in (rep.depth.depth_null, rep.composition.cross_null,
               rep.composition.oracle_null, rep.spatial.r_nbr):
        assert st.n_genes == Y.shape[1], st


# ------------------------------------------------------------------ fix 9
def test_r_mean_carries_the_floor_of_the_mean_rule():
    Y, P, sec, lib, xy, lab = toy()
    rep = audit(Y, P, section=sec, lib_size=lib, coords=xy, labels=lab,
                **dict(KW, n_perm=40))
    h = rep.headline
    assert h.r_mean.floor is not None and h.r_median.floor is not None
    # the two rules give different nulls on a skewed per-gene r spread
    assert h.r_mean.floor != h.r_median.floor
    assert "nperm" in h.r_mean.note


# ------------------------------------------------------------------ fix 10
def test_share_depth_ignores_a_near_zero_denominator():
    from stnull.audit import _share_stat
    st = _share_stat([0.20, 2e-6], [0.10, 0.05], [None] * 2, [None] * 2,
                     [100, 100])
    assert np.isfinite(st.value) and abs(st.value) < 1.0, st


# ------------------------------------------------------------------ fix 11
def test_an_unfittable_null_says_why():
    """A composition design wider than the section must report a reason."""
    rng = np.random.default_rng(0)
    n, G, K = 60, 25, 40
    Y = rng.normal(size=(n, G))
    P = rng.normal(size=(n, G))
    sec = np.array(["S"] * n)
    comp = rng.random((n, K))
    rep = audit(Y, P, space="custom", space_note="unit test", section=sec,
                composition=comp, n_perm=5, n_boot=0, top_n=(5,), min_spots=50,
                min_genes=20, verbose=False)
    assert not np.isfinite(rep.composition.cross_null.value)
    joined = " ".join(rep.run.degradations) + " ".join(rep.warnings)
    assert "cross_null" in joined and "not fitted" in joined, joined


# ------------------------------------------------------------------ fix 12
def test_composition_oracle_does_not_saturate_on_a_wide_design():
    from stnull.nulls import composition_null
    rng = np.random.default_rng(0)
    Y = rng.normal(size=(80, 30))                  # no composition signal
    comp = rng.random((80, 79))                    # p >= n - 2
    notes = []
    r, _, _ = composition_null(Y, comp, "oracle", notes=notes)
    assert not np.isfinite(r).any(), "in-sample oracle interpolated to r=1"
    assert notes and "saturated" in notes[0]


# ------------------------------------------------------------------ fix 13
def test_smooth_matrix_is_an_unweighted_3x3_box_mean():
    from stnull.nulls import smooth_matrix
    gx, gy = np.meshgrid(np.arange(5), np.arange(5))
    xy = np.stack([gx.ravel(), gy.ravel()], 1).astype(float)
    S = smooth_matrix(xy, "grid").toarray()
    centre = 12                                    # the (2, 2) interior spot
    assert S[centre, centre] == pytest.approx(1.0 / 9.0)
    nb = [j for j in range(25) if j != centre and S[centre, j] > 0]
    assert len(nb) == 8
    assert np.allclose([S[centre, j] for j in nb], 1.0 / 9.0)
    assert S.sum(1) == pytest.approx(np.ones(25))


# ------------------------------------------------------------------ fix 14
def test_r_tech_floor_averages_the_same_replicates_as_the_value():
    """The floor must follow the value's rule: more replicates, tighter null.

    0.2.0 built the floor from replicate 0 alone, so it did not move with
    n_rep at all and sat ~38% too high on zero-signal data.
    """
    from stnull.ceiling import r_tech
    from stnull.spaces import TargetSpace
    C = np.random.default_rng(0).poisson(2.0, size=(200, 40)).astype(np.int64)
    f1 = np.percentile(r_tech(C, TargetSpace.LOG1P_CP10K_PANEL, None, n_rep=1,
                              rng=np.random.default_rng(1), n_perm=40)[1], 95)
    f20 = np.percentile(r_tech(C, TargetSpace.LOG1P_CP10K_PANEL, None, n_rep=20,
                               rng=np.random.default_rng(1), n_perm=40)[1], 95)
    assert f20 < f1, "the floor ignored n_rep (%.4f vs %.4f)" % (f20, f1)


def test_r_tech_reports_non_finite_lib_full_instead_of_crashing():
    from stnull.ceiling import r_tech
    from stnull.spaces import TargetSpace
    C = np.random.default_rng(0).poisson(3.0, size=(120, 30)).astype(np.int64)
    lf = C.sum(1) * 3.0
    lf[7] = np.nan
    r, pn, method, note = r_tech(C, TargetSpace.LOG1P_CP10K_FULL, lf, n_rep=2,
                                 rng=np.random.default_rng(0), n_perm=0)
    assert r is None and method == "na" and "non-finite" in note


def test_r_tech_note_discloses_the_sparse_regime_bias():
    from stnull.ceiling import r_tech
    from stnull.spaces import TargetSpace
    C = np.random.default_rng(0).poisson(0.8, size=(150, 30)).astype(np.int64)
    note = r_tech(C, TargetSpace.LOG1P_CP10K_PANEL, None, n_rep=2,
                  rng=np.random.default_rng(0), n_perm=0)[3]
    assert "mean count per spot-gene" in note and "HIGH" in note


# ------------------------------------------------------------------ fix 15
def test_from_pred_with_lib_pred_discloses_the_downgrade():
    Y, P, sec, lib, xy, lab = toy()
    rep = audit(Y, P, section=sec, lib_size=lib, labels=lab,
                lib_pred=np.log(lib), depth_proxy="from_pred", **KW)
    assert rep.run.depth_proxy == "given"
    assert any("downgraded" in w for w in rep.warnings)
    assert any("permutation guard" in d for d in rep.run.degradations)


def test_rejected_from_pred_proxy_is_not_published():
    """A proxy the guard rejects must not reappear with an analytic-zero floor."""
    rng = np.random.default_rng(0)
    n, G = 120, 25
    Y = rng.normal(size=(n, G))
    P = rng.normal(size=(n, G))                    # carries no depth at all
    sec = np.array(["S"] * n)
    lib = np.exp(rng.normal(8.5, 0.5, n))
    rep = audit(Y, P, space="custom", space_note="unit test", section=sec,
                lib_size=lib, depth_proxy="from_pred", n_perm=5, n_boot=0,
                top_n=(5,), min_spots=50, min_genes=20, verbose=False)
    drp = rep.depth.depth_r_of_pred
    assert not np.isfinite(drp.value), drp
    assert any("depth proxy" in d for d in rep.run.degradations)


# --------------------------------------------------- nulls are self-contained
def test_null_values_do_not_depend_on_which_other_nulls_ran():
    """One shared fold partition per section: adding a null cannot move another."""
    Y, P, sec, lib, xy, lab = toy()
    a = audit(Y, P, section=sec, lib_size=lib, **KW)
    b = audit(Y, P, section=sec, lib_size=lib, coords=xy, labels=lab, **KW)
    assert a.depth.depth_null.value == pytest.approx(b.depth.depth_null.value)
