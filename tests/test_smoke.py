# -*- coding: utf-8 -*-
"""Synthetic smoke tests -- no project data required."""
import os

os.environ.setdefault("OMP_NUM_THREADS", "3")

import numpy as np
import pandas as pd
import pytest

import stnull
from stnull import audit, check_inputs, nulls_only, perm_floor
from stnull.core import MissingSpace, Stat


def make_toy(n_sec=3, n_side=9, G=40, seed=0):
    """A dataset whose only real signal is depth + a 2-class composition."""
    rng = np.random.default_rng(seed)
    rows = []
    Ys, Ps, secs, libs, xy, labs, pats, cnts = [], [], [], [], [], [], [], []
    for s in range(n_sec):
        gx, gy = np.meshgrid(np.arange(n_side), np.arange(n_side))
        c = np.stack([gx.ravel(), gy.ravel()], 1).astype(float)
        n = c.shape[0]
        lib = np.exp(rng.normal(8.5, 0.6, n))
        lab = np.where(c[:, 0] < n_side / 2, "tumour", "stroma")
        load = np.log(lib) - np.log(lib).mean()
        theme = (lab == "tumour").astype(float)
        B = rng.normal(0, 1, (2, G))
        base = rng.normal(0, 1, (n, G)) * 0.9
        Y = base + np.outer(load, B[0]) * 0.7 + np.outer(theme, B[1]) * 0.5
        # a "model" that mostly reproduces depth and composition, plus a little
        # genuine residual signal
        P = (np.outer(load, B[0]) * 0.6 + np.outer(theme, B[1]) * 0.4
             + base * 0.15 + rng.normal(0, .4, (n, G)))
        C = rng.poisson(np.clip(np.exp(Y / 2), 0, 50)).astype(np.int64)
        Ys.append(Y)
        Ps.append(P)
        Cs = C
        cnts.append(Cs)
        secs.append(np.array(["S%d" % s] * n))
        libs.append(lib)
        xy.append(c)
        labs.append(lab)
        pats.append(np.array(["P%d" % (s // 2)] * n))
    return (np.vstack(Ys), np.vstack(Ps), np.concatenate(secs),
            np.concatenate(libs), np.vstack(xy), np.concatenate(labs),
            np.concatenate(pats), np.vstack(cnts))


@pytest.fixture(scope="module")
def toy():
    return make_toy()


def test_space_is_required(toy):
    Y, P, sec = toy[0], toy[1], toy[2]
    with pytest.raises(MissingSpace):
        audit(Y, P, section=sec, verbose=False)
    with pytest.raises(MissingSpace):
        audit(Y, P, space="log1p_cp10k", section=sec, verbose=False)
    with pytest.raises(MissingSpace):
        audit(Y, P, space="custom", section=sec, verbose=False)


def test_minimal_call_runs(toy):
    Y, P, sec = toy[0], toy[1], toy[2]
    rep = audit(Y, P, space="zscore:per_gene", section=sec, n_perm=20, n_boot=20,
                top_n=(5, 10), min_spots=20, verbose=False)
    assert rep.headline is not None
    assert np.isfinite(rep.headline.r_median.value)
    assert rep.headline.r_median.floor is not None
    assert "depth" in rep.na and "composition" in rep.na
    assert any("lib_size" in d for d in rep.run.degradations)
    txt = rep.summary()
    txt.encode("ascii")            # summary must be printable on a cp936 console


def test_full_call_and_ladder(toy, tmp_path):
    Y, P, sec, lib, xy, lab, pat, cnt = toy
    rep = audit(Y, P, space="zscore:per_gene", section=sec, lib_size=lib,
                coords=xy, labels=lab, patient=pat, counts=cnt,
                n_perm=30, n_boot=40, top_n=(5, 10), min_spots=20,
                n_rep_ceiling=3, verbose=False)
    lad = rep.ladder
    for c in ("r_full", "dg", "crg", "drg"):
        assert c in lad.columns and lad[c].notna().all()
        assert (c + "_floor") in lad.columns
    # controls must remove signal that is really there
    assert (lad["drg"] < lad["r_full"]).all()
    assert rep.depth is not None and rep.composition is not None
    assert rep.spatial is not None and rep.ceiling is not None
    assert rep.ceiling.r_tech is not None
    # every exposed Stat carries a floor or an explicit reason
    for sec_obj in (rep.headline, rep.depth, rep.composition, rep.spatial):
        for v in sec_obj.__dict__.values():
            if isinstance(v, Stat) and v.status == "ok":
                assert v.floor is not None or v.floor_kind in ("none", "derived",
                                                               "analytic_zero")
    # patient P0 owns two sections -> the zero-pixel leakage lever is measurable
    assert rep.selection.leakage_lever is not None
    assert np.isfinite(rep.selection.leakage_lever.value)
    h = tmp_path / "a.html"
    rep.to_html(h)
    s = h.read_text(encoding="utf-8")
    assert "null_floor" in s and "http://" not in s and "https://" not in s
    rep.to_markdown(tmp_path / "a.md")
    rep.to_json(tmp_path / "a.json")
    files = rep.to_csv_dir(tmp_path / "csv")
    assert len(files) >= 3


def test_selection_floor_is_far_above_zero(toy):
    Y, P, sec = toy[0], toy[1], toy[2]
    rng = np.random.default_rng(1)
    Pn = rng.normal(size=P.shape)            # pure noise predictions
    st = perm_floor(Y, Pn, section=sec, rule="top_n", top_n=5, kind="free",
                    n_perm=50)
    v = perm_floor(Y, Pn, section=sec, rule="all", kind="free", n_perm=50)
    assert st.floor > 0.05                    # selection alone buys real r
    assert abs(v.value) < 0.05                # the all-gene readout does not


def test_nulls_only_mode(toy):
    Y, sec, lib, xy, lab = toy[0], toy[2], toy[3], toy[4], toy[5]
    rep = nulls_only(Y, space="zscore:per_gene", section=sec, lib_size=lib,
                     coords=xy, labels=lab, n_perm=20, n_boot=0, min_spots=20,
                     verbose=False)
    assert rep.headline is None
    assert rep.depth.depth_null.value > 0
    assert "NULLS-ONLY" in rep.summary()


def test_check_inputs_catches_panel_sum(toy):
    Y, P, sec = toy[0], toy[1], toy[2]
    Ypos = np.abs(Y)
    rowsum = np.expm1(Ypos).sum(1)
    r = check_inputs(y_true=Ypos, y_pred=P, section=sec, lib_size=rowsum,
                     min_spots=20)
    assert not r.ok
    assert any("PANEL" in e for e in r.errors)


def test_zero_variance_gene_is_counted_not_dropped(toy):
    Y, P, sec = toy[0].copy(), toy[1], toy[2]
    Y[:, 0] = 3.0
    rep = audit(Y, P, space="zscore:per_gene", section=sec, n_perm=10, n_boot=10,
                top_n=(5, 10), min_spots=20, verbose=False)
    assert rep.run.n_genes == Y.shape[1]
    assert rep.headline.r_median.n_genes == Y.shape[1] - 1
    assert rep.per_gene[rep.per_gene.gene == "g0"]["r_full"].isna().all()


def test_strict_rejects_small_section(toy):
    Y, P, sec = toy[0], toy[1], toy[2].copy()
    sec[:5] = "TINY"
    with pytest.raises(stnull.InsufficientData):
        audit(Y, P, space="zscore:per_gene", section=sec, min_spots=20,
              top_n=(5,), n_perm=5, verbose=False)


def test_point_estimates_do_not_move_with_n_perm(toy):
    Y, P, sec, lib = toy[0], toy[1], toy[2], toy[3]
    kw = dict(space="zscore:per_gene", section=sec, lib_size=lib, min_spots=20,
              top_n=(5, 10), n_boot=0, verbose=False)
    a = audit(Y, P, n_perm=10, **kw)
    b = audit(Y, P, n_perm=40, **kw)
    assert a.headline.r_median.value == pytest.approx(b.headline.r_median.value)
    assert a.depth.depth_null.value == pytest.approx(b.depth.depth_null.value)
    assert a.depth.dg.value == pytest.approx(b.depth.dg.value)


def test_strict_rejects_top_n_wider_than_panel(toy):
    Y, P, sec = toy[0], toy[1], toy[2]
    with pytest.raises(stnull.BadInput):
        audit(Y, P, space="zscore:per_gene", section=sec, min_spots=20,
              top_n=(10, 500), n_perm=5, n_boot=0, verbose=False)


def test_seed_is_process_independent():
    """Point estimates must not depend on PYTHONHASHSEED (regression test)."""
    import subprocess, sys, os, json, textwrap
    code = textwrap.dedent("""
        import os, json
        os.environ["OMP_NUM_THREADS"] = "2"
        import sys
        sys.path.insert(0, r"%s")
        from test_smoke import make_toy
        from stnull import audit
        Y, P, sec, lib = make_toy()[:4]
        rep = audit(Y, P, space="zscore:per_gene", section=sec, lib_size=lib,
                    top_n=(5,), n_perm=5, n_boot=0, min_spots=20, verbose=False)
        print(json.dumps([rep.headline.r_median.value, rep.depth.depth_null.value]))
    """) % os.path.dirname(os.path.abspath(__file__))
    outs = []
    for salt in ("0", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=salt)
        r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                           text=True, env=env)
        assert r.returncode == 0, r.stderr[-800:]
        outs.append(json.loads(r.stdout.strip().splitlines()[-1]))
    assert outs[0] == pytest.approx(outs[1])


def test_cli_help():
    from stnull.cli import main
    with pytest.raises(SystemExit) as e:
        main(["audit", "--help"])
    assert e.value.code == 0


# ------------------------------------------------------------- 0.2.0 fixes
def test_micron_coords_are_detected_not_silent(toy):
    """Micron coordinates under the default coord_kind='grid' must warn, book
    a degradation and switch to kNN/micron handling (was: silent N/A + free
    floors, defect #3 of 0.1.0)."""
    Y, P, sec, lib, xy = toy[0], toy[1], toy[2], toy[3], toy[4]
    xym = xy * 187.0 + 13.0                    # a fake micron pitch
    rep = audit(Y, P, space="zscore:per_gene", section=sec, lib_size=lib,
                coords=xym, n_perm=20, n_boot=0, top_n=(5,), min_spots=20,
                verbose=False)
    assert any("micron" in w for w in rep.warnings)
    assert any("micron" in d for d in rep.run.degradations)
    # the spatial section stays alive instead of going N/A
    assert np.isfinite(rep.spatial.r_nbr.value)
    # and the floor did not silently fall back to free permutation
    assert rep.run.perm_kind == "perm_block"


def test_top_n_default_adapts_to_small_panels(toy):
    """The default top_n must not crash a 50-gene (HEST-style) panel under
    strict=True (defect #4 of 0.1.0); an EXPLICIT oversized top_n still raises
    (see test_strict_rejects_top_n_wider_than_panel)."""
    Y, P, sec = toy[0][:, :30], toy[1][:, :30], toy[2]
    rep = audit(Y, P, space="zscore:per_gene", section=sec, n_perm=10, n_boot=0,
                min_spots=20, min_genes=20, verbose=False)   # top_n = default
    assert set(rep.selection.by_n.keys()) == {10}


def test_oracle_null_fit_is_flagged_and_barred_from_headline(toy):
    Y, P, sec, lib, xy, lab = toy[0], toy[1], toy[2], toy[3], toy[4], toy[5]
    rep = audit(Y, P, space="zscore:per_gene", section=sec, lib_size=lib,
                coords=xy, labels=lab, null_fit="oracle", n_perm=20, n_boot=0,
                top_n=(5,), min_spots=20, verbose=False)
    assert "UPPER_BOUND" in rep.depth.depth_null.note
    assert "UPPER_BOUND" in rep.composition.cross_null.note
    # in-sample nulls may not be quoted as the strongest achievable null
    if rep.headline.strongest_null is not None:
        assert rep.headline.strongest_null[0] == "r_nbr"
    assert any("oracle" in d for d in rep.run.degradations)


def test_train_only_without_is_train_is_explicit(toy):
    Y, P, sec, lib = toy[0], toy[1], toy[2], toy[3]
    with pytest.raises(stnull.BadInput):
        audit(Y, P, space="zscore:per_gene", section=sec, lib_size=lib,
              null_fit="train_only", n_perm=5, n_boot=0, top_n=(5,),
              min_spots=20, verbose=False)


def test_degradation_ledger_never_misstates_null_fit(toy):
    """is_train missing + null_fit='oracle' used to print 'nulls are
    cross-fitted within section (null_fit=oracle)' -- a false sentence."""
    Y, P, sec, lib = toy[0], toy[1], toy[2], toy[3]
    rep = audit(Y, P, space="zscore:per_gene", section=sec, lib_size=lib,
                null_fit="oracle", n_perm=10, n_boot=0, top_n=(5,),
                min_spots=20, verbose=False)
    assert not any(("cross-fitted within section" in d) and ("oracle" in d)
                   for d in rep.run.degradations)


def test_no_self_neighbour_on_duplicate_micron_coords():
    from stnull.nulls import neighbour_matrix
    rng = np.random.default_rng(0)
    C = rng.uniform(0, 1000, (40, 2))
    C[1] = C[0]                                # coincident pair
    W = neighbour_matrix(C, coord_kind="micron", k=4)
    assert W.diagonal().max() == 0.0


def test_block_perm_single_column_never_identity():
    from stnull.core import perm_plan
    n = 60
    coords = np.c_[np.zeros(n), np.arange(n, dtype=float)]
    idx, msk = perm_plan(n, 200, "block", coords, "grid",
                         np.random.default_rng(0))
    assert msk is not None                     # stayed a block plan
    ident = np.arange(n)
    for b in range(idx.shape[0]):
        m = msk[b]
        assert not np.array_equal(idx[b][m], ident[m]), \
            "identity permutation returned at draw %d" % b


def test_n_perm_zero_gives_na_not_crash(toy):
    Y, P, sec, lib = toy[0], toy[1], toy[2], toy[3]
    rep = audit(Y, P, space="zscore:per_gene", section=sec, lib_size=lib,
                n_perm=0, n_boot=0, top_n=(5,), min_spots=20, verbose=False)
    st = rep.headline.r_median
    assert np.isfinite(st.value)
    assert st.floor is None
    assert st.status == "na"
    assert "n_perm=0" in st.note


def test_r_tech_zero_signal_limit_is_near_zero():
    """The 0.1.0 'multinomial_resample' ceiling read ~0.67-0.71 on data with
    ZERO signal (the replicate shared the observation's own sampling noise).
    The 0.2.0 split-half estimator must stay near zero there, and below its
    own permutation floor."""
    from stnull.ceiling import r_tech
    from stnull.spaces import TargetSpace
    rng = np.random.default_rng(0)
    C = rng.poisson(3.0, size=(300, 60)).astype(np.int64)   # iid: no signal
    r, pn, method, _ = r_tech(C, TargetSpace.LOG1P_CP10K_PANEL,
                              np.full(300, 2.0e4), n_rep=5,
                              rng=np.random.default_rng(1), n_perm=20)
    v = float(np.nanmedian(r))
    assert method == "split_half_full"
    assert v < 0.15, "zero-signal r_tech %.3f: a biased ceiling estimator" % v
    assert v < float(np.percentile(pn, 95))


def test_ceiling_skip_is_counted_not_broken(toy):
    """A section with no r_tech estimate must be SKIPPED and counted, not
    silently truncate every later section (the old `break`)."""
    Y, P, sec, cnt = toy[0], toy[1], toy[2], toy[7]
    rep = audit(Y, P, space="custom", space_note="unit test", section=sec,
                counts=cnt, n_perm=10, n_boot=0, top_n=(5,), min_spots=20,
                n_rep_ceiling=2, verbose=False)
    assert rep.ceiling is None or rep.ceiling.r_tech is None
    assert any("no r_tech" in w for w in rep.warnings)
