# -*- coding: utf-8 -*-
"""Synthetic ground-truth validation of the attribution ladder.

Each scenario has a KNOWN answer: the 'oracle ladder', obtained by
residualising on the TRUE latent depth and the TRUE class labels.  The tool
must reproduce it without being told which latent is which.

Run the full 20-seed sweep with:  python tests/run_synth.py
These tests use fewer seeds so that pytest stays under a minute per test.

STATUS as measured on 2026-08-21 (20 seeds x 3 sections x 60 genes each):

  scenario   oracle DRG   stnull DRG (from_pred)   stnull drg_observed
  depth        0.0036            0.0163                  0.0033
  signal       0.1378            0.0635   <-- FALSE NEG  0.1378
  mixed        0.0882            0.1021                  0.0891
  comp         0.0013           -0.0005                  0.0016
  noise        0.0030            0.0021                  0.0027

The `signal` row was a real defect, not test noise: with depth_proxy='from_pred'
(the 0.1.0 default) the cross-fitted l_hat latched onto the model's own
(depth-free) signal -- measured |corr(l_hat, u_true)| up to 0.75 while
corr(l_hat, log lib) was 0.06 -- so residualising on it deleted the very signal
DRG is supposed to certify.  FIXED (0.2.0): the default is now 'observed' and
'from_pred' carries a permutation guard; the old pinned-defect test
(test_signal_scenario_is_a_known_false_negative) was rewritten as
test_signal_scenario_recovers_the_truth, which asserts the fixed behaviour.
Measured with the default config (6 seeds): |tool DRG - oracle DRG| <= 0.0011
on depth/signal/mixed/noise.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "3")
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import synth                                   # noqa: E402
from stnull import audit                       # noqa: E402

N_SEED = int(os.environ.get("STNULL_TEST_SEEDS", "6"))
KW = dict(space="custom", space_note="synthetic", top_n=(5, 10, 30),
          n_perm=100, n_boot=200, verbose=False, seed=0)


def _run(scenario, seed):
    d = synth.make(scenario, 1000 + seed)
    rep = audit(y_true=d["y"], y_pred=d["pred"], section=d["section"],
                lib_size=d["lib"], coords=d["coords"], labels=d["labels"], **KW)
    lad = rep.ladder
    return d, rep, dict(
        oracle=synth.oracle(d),
        r_full=float(np.nanmedian(lad["r_full"])),
        drg=float(np.nanmedian(lad["drg"])),
        drg_obs=float(np.nanmedian(lad["drg_observed"])),
        crg=float(np.nanmedian(lad["crg"])),
        n_above=int(lad["drg_above_floor"].sum()), n_sec=len(lad))


def _sweep(scenario):
    return [_run(scenario, s)[2] for s in range(N_SEED)]


# ------------------------------------------------------- 1. pure depth confound
def test_pure_depth_confound_no_false_positive():
    """y and y_hat driven ONLY by library size -> DRG must not fire."""
    res = _sweep("depth")
    assert np.mean([r["r_full"] for r in res]) > 0.10
    assert sum(r["n_above"] for r in res) == 0, "DRG flagged signal that is 100% depth"
    drg = np.mean([r["drg"] for r in res])
    assert drg < 0.05, "DRG point estimate %.4f leaks depth" % drg


# --------------------------------------------- 2. pure depth-independent signal
def test_signal_scenario_recovers_the_truth():
    """y driven by a latent INDEPENDENT of depth: DRG must recover r_full.

    History: before the 2026-08-21 fix this was a documented FALSE NEGATIVE --
    ``fit_depth_proxy`` picked its alpha by maximising corr(l_hat, loglib), a
    noise lottery when no depth signal exists, so the winning l_hat was often
    y_pred's own leading direction and residualising deleted the very signal
    under test (drg was ~0.46x oracle, worse at larger n_gene).  The fix was
    (a) default depth_proxy='observed' and (b) a permutation guard on the
    from_pred alpha selection.  This test now asserts the fixed behaviour.
    """
    res = _sweep("signal")
    oracle = np.mean([r["oracle"]["drg"] for r in res])
    drg = np.mean([r["drg"] for r in res])
    obs = np.mean([r["drg_obs"] for r in res])
    assert oracle > 0.10                       # the truth really is there
    assert drg >= 0.7 * oracle, ("DRG under-reports the depth-independent signal "
                                 "(%.4f vs oracle %.4f)" % (drg, oracle))
    assert abs(obs - oracle) < 0.01, "drg_observed drifted from the oracle"


def test_depth_proxy_observed_recovers_the_truth_in_every_scenario():
    for sc in ("depth", "signal", "mixed", "comp"):
        res = _sweep(sc)
        o = np.mean([r["oracle"]["drg"] for r in res])
        v = np.mean([r["drg_obs"] for r in res])
        assert abs(v - o) < 0.01, "%s: drg_observed %.4f vs oracle %.4f" % (sc, v, o)


# ------------------------------------------------------------------- 3. mixed
def test_mixed_is_between_zero_and_r_full():
    res = _sweep("mixed")
    rf = np.mean([r["r_full"] for r in res])
    drg = np.mean([r["drg"] for r in res])
    o = np.mean([r["oracle"]["drg"] for r in res])
    assert 0.25 * rf < drg < 0.75 * rf
    assert abs(drg - o) < 0.03
    assert sum(r["n_above"] for r in res) >= 0.7 * sum(r["n_sec"] for r in res)


# ------------------------------------------------------------------- 4. noise
def test_pure_noise_stays_at_the_floor():
    res = _sweep("noise")
    assert abs(np.mean([r["r_full"] for r in res])) < 0.03
    assert abs(np.mean([r["drg"] for r in res])) < 0.03
    assert sum(r["n_above"] for r in res) == 0


# ------------------------------------------------------------- 5. composition
def test_pure_composition_confound_kills_crg_not_r_full():
    res = _sweep("comp")
    assert np.mean([r["r_full"] for r in res]) > 0.15
    assert abs(np.mean([r["crg"] for r in res])) < 0.03


# --------------------------------------------------------------- 6. selection
def test_selection_floor_matches_the_spurious_value():
    """Under pure noise, top-N reporting buys r; the floor must buy the same.

    Aggregate, not per-seed: over 20 seeds the top-N readout and its own
    perm_selection floor agree to within 0.001 on average (top-5 0.1957 vs
    0.1964, top-10 0.1540 vs 0.1565, top-30 0.0740 vs 0.0827).  A single seed
    can still land up to +0.08 above the floor at top-5, and the
    ``above_floor()`` decision fired in 3/20 seeds -- so a top-5 readout on 60
    genes is not powered enough to be read one dataset at a time.
    """
    exc = {}
    fired = 0
    for seed in range(N_SEED):
        d = synth.make("selection", 1000 + seed)
        rep = audit(y_true=d["y"], y_pred=d["pred"], section=d["section"],
                    lib_size=d["lib"], coords=d["coords"], labels=d["labels"], **KW)
        for N, st in rep.selection.by_n.items():
            assert st.floor is not None, "top-%d printed without a floor" % N
            exc.setdefault(N, []).append(st.value - st.floor)
            fired += bool(st.above_floor())
    for N, e in exc.items():
        assert abs(np.mean(e)) < 0.03, (
            "top-%d sits %.4f above its own selection floor on average"
            % (N, np.mean(e)))
    assert fired <= 0.35 * N_SEED * len(exc), (
        "selection floor fired %d times out of %d under pure noise"
        % (fired, N_SEED * len(exc)))
