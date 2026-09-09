# -*- coding: utf-8 -*-
"""Runner: N seeds x scenarios, tool ladder vs oracle ladder. ASCII output only."""
import os, sys, time, json
os.environ.setdefault("OMP_NUM_THREADS", "3")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import synth
from stnull import audit

N_SEED = int(os.environ.get("NSEED", "20"))
OUT = os.environ.get("OUT", "synth_results.csv")
SCEN = os.environ.get("SCEN", "depth,signal,mixed,noise,comp,selection").split(",")
PROXY = os.environ.get("PROXY", "from_pred")

rows = []
t0 = time.time()
for sc in SCEN:
    for seed in range(N_SEED):
        d = synth.make(sc, 1000 + seed)
        o = synth.oracle(d)
        rep = audit(y_true=d["y"], y_pred=d["pred"], space="custom",
                    space_note="synthetic", section=d["section"],
                    lib_size=d["lib"], coords=d["coords"], labels=d["labels"],
                    top_n=(5, 10, 30), n_perm=100, n_boot=200,
                    depth_proxy=PROXY, verbose=False, seed=0)
        lad = rep.ladder
        rec = dict(scenario=sc, seed=seed, proxy=PROXY,
                   o_rfull=o["r_full"], o_dg=o["dg"], o_crg=o["crg"], o_drg=o["drg"],
                   t_rfull=float(np.nanmedian(lad["r_full"])),
                   t_dg=float(np.nanmedian(lad["dg"])),
                   t_crg=float(np.nanmedian(lad["crg"])),
                   t_drg=float(np.nanmedian(lad["drg"])),
                   t_drg_obs=float(np.nanmedian(lad["drg_observed"])),
                   drg_floor=float(np.nanmedian(lad["drg_floor"])),
                   n_sec_drg_above=int(lad["drg_above_floor"].sum()),
                   n_sec=len(lad),
                   rfull_floor=float(rep.headline.r_median.floor or np.nan),
                   rfull_above=bool(rep.headline.r_median.above_floor()),
                   corr_lhat_loglib=float(rep.depth.depth_r_of_pred.value),
                   crg_stat=float(rep.composition.crg.value),
                   crg_floor=float(rep.composition.crg.floor or np.nan),
                   crg_above=bool(rep.composition.crg.above_floor()),
                   dg_above=bool(rep.depth.dg.above_floor()),
                   )
        # per-section drg CI covers 0?
        lo = lad["drg_ci_lo"].values; hi = lad["drg_ci_hi"].values
        rec["n_sec_drgCI_covers0"] = int(np.sum((lo <= 0) & (hi >= 0)))
        # selection
        for N, st in rep.selection.by_n.items():
            rec["top%d" % N] = st.value
            rec["top%d_floor" % N] = st.floor if st.floor is not None else np.nan
            rec["top%d_above" % N] = bool(st.above_floor())
        # diagnostics on l_hat
        lh = []
        for s in sorted(set(d["section"].tolist())):
            m = d["section"] == s
            lh.append(np.nan)
        rec["elapsed"] = time.time() - t0
        rows.append(rec)
        sys.stdout.write("[%s seed=%d] rfull %.3f drg %.3f (oracle %.3f) %.0fs\n"
                         % (sc, seed, rec["t_rfull"], rec["t_drg"], rec["o_drg"],
                            time.time() - t0))
        sys.stdout.flush()
df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)
print("wrote", OUT, df.shape)
