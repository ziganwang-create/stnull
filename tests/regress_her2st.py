# -*- coding: utf-8 -*-
"""HER2ST regression: recompute the project's frozen numbers with stnull.

Read-only on prototype/. Writes only into stnull/tests/.
Stage 1 (--stage nulls): all 36 sections, per fold, no labels.
        -> depth_null (target 0.2298), r_nbr, r_tech, r_full (0.1133/0.1167),
           test-selected top-100 floor (target 0.080)
Stage 2 (--stage drg): 8 annotated sections, annotated spots only, lib_pred=l_hat
        -> r_full 0.1196, DRG 0.0173 (resnet50)
"""
import os, sys, json, time, argparse
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[v] = "3"
import numpy as np, pandas as pd, scipy.sparse as sp

PROTO = r"D:\project\STImage\prototype"
DATA = os.path.join(PROTO, "data")
PREDS = os.path.join(PROTO, "preds")
HERE = os.path.dirname(os.path.abspath(__file__))
from stnull import audit

PATIENTS = list("ABCDEFGH")


def load_fold(p, meta, genes_common):
    z = np.load(os.path.join(PREDS, "ridge_fold_%s.npz" % p), allow_pickle=False)
    rows = z["rows"]; genes = z["genes"].tolist(); pred = z["pred_panel737"].astype(np.float64)
    g2c = {g: i for i, g in enumerate(genes_common)}
    gidx = np.array([g2c[g] for g in genes], dtype=np.int64)
    sub = meta.iloc[rows]
    mats = []
    for sec in pd.unique(sub["section"].values):
        gs = json.loads(open(os.path.join(DATA, "genes_%s.json" % sec), encoding="utf-8").read())
        col = {x: i for i, x in enumerate(gs)}
        sel = [col[genes_common[i]] for i in gidx]
        m = sp.load_npz(os.path.join(DATA, "expr_%s.npz" % sec))
        take = meta[meta.section == sec]["row_idx"].isin(rows).values
        mats.append(np.asarray(m[take][:, sel].todense(), np.float64))
    C = np.vstack(mats)
    den = np.maximum(C.sum(1, keepdims=True), 1.0)
    Y = np.log1p(1e4 * C / den)
    return dict(rows=rows, genes=genes, pred=pred, counts=C, Y=Y, sub=sub)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="nulls")
    ap.add_argument("--nperm", type=int, default=200)
    ap.add_argument("--nboot", type=int, default=100)
    ap.add_argument("--nrep", type=int, default=20)
    ap.add_argument("--nullfit", default="cv_within_section")
    ap.add_argument("--folds", default="ABCDEFGH")
    a = ap.parse_args()
    meta = pd.read_parquet(os.path.join(DATA, "meta.parquet"))
    genes_common = open(os.path.join(DATA, "genes.txt"), encoding="utf-8").read().splitlines()
    out = []
    t0 = time.time()
    for p in a.folds:
        d = load_fold(p, meta, genes_common)
        sub = d["sub"]
        kw = dict(y_true=d["Y"], y_pred=d["pred"], space="log1p_cp10k:panel",
                  section=sub["section"].values, lib_size=sub["lib_size"].values,
                  coords=sub[["arr_x", "arr_y"]].values, coord_kind="grid",
                  patient=sub["patient"].values, genes=d["genes"],
                  top_n=(10, 50, 100), n_perm=a.nperm, n_boot=a.nboot,
                  null_fit=a.nullfit, verbose=False, seed=0)
        if a.stage == "nulls":
            kw.update(counts=d["counts"], lib_size_full=sub["lib_size"].values,
                      n_rep_ceiling=a.nrep)
        else:
            lab = sub["label"].fillna("").astype(str).values
            keep = np.array([bool(x) for x in lab])
            lh = np.load(os.path.join(HERE, "lhat_resnet50.npy"))[d["rows"]]
            for k in ("y_true", "y_pred"):
                kw[k] = kw[k][keep]
            for k in ("section", "lib_size", "patient"):
                kw[k] = kw[k][keep]
            kw["coords"] = kw["coords"][keep]
            kw["labels"] = lab[keep]
            kw["lib_pred"] = lh[keep]
            kw["min_class_spots"] = 1
        rep = audit(**kw)
        ps = rep.per_section
        for _, r in ps.iterrows():
            rec = dict(section=r["section"], patient=p, n_spots=r["n_spots"],
                       r_model=r["r_model"], depth_null=r.get("depth_null", np.nan),
                       r_nbr=r.get("r_nbr", np.nan), dg=r.get("dg", np.nan),
                       crg=r.get("crg", np.nan), drg=r.get("drg", np.nan),
                       drg_observed=r.get("drg_observed", np.nan))
            out.append(rec)
        for N, st in rep.selection.by_n.items():
            out[-1]["top%d" % N] = st.value
            out[-1]["top%d_floor" % N] = st.floor
        if rep.ceiling is not None and rep.ceiling.r_tech is not None:
            out[-1]["r_tech_fold"] = rep.ceiling.r_tech.value
        out[-1]["rfull_floor_fold"] = rep.headline.r_median.floor
        print("[%s] %.0fs  r_full=%.4f depth_null=%.4f r_nbr=%.4f drg=%.4f"
              % (p, time.time()-t0,
                 np.nanmedian([o["r_model"] for o in out if o["patient"]==p]),
                 np.nanmedian([o["depth_null"] for o in out if o["patient"]==p]),
                 np.nanmedian([o["r_nbr"] for o in out if o["patient"]==p]),
                 np.nanmedian([o["drg"] for o in out if o["patient"]==p])))
        sys.stdout.flush()
    df = pd.DataFrame(out)
    f = os.path.join(HERE, "regress_%s_%s.csv" % (a.stage, a.nullfit))
    df.to_csv(f, index=False)
    print("wrote", f)
    for c in ("r_model","depth_null","r_nbr","dg","crg","drg","drg_observed"):
        if c in df and df[c].notna().any():
            print("MEDIAN %-14s = %.4f  (n=%d)" % (c, np.nanmedian(df[c]), df[c].notna().sum()))


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# RESULTS (2026-08-21, this machine)
#
# stage=drg (8 annotated sections, annotated spots, lib_pred = image l_hat):
#   resnet50 : r_full 0.1196  DG 0.0264  CRG 0.0539  DRG 0.0173
#              frozen  0.1196     0.0265     0.0539      0.0173   max |diff| 0.0000
#   phikon_p50: DRG 0.0295 vs frozen 0.0295, max |diff| 0.0000 on all 4 rungs
#
# stage=nulls (36 sections):
#   r_nbr        tool 0.1281 vs ceiling_rtech_rnbr_summary 0.1281, max |diff| 0.00000
#   r_tech       [0.1.0 estimator, VOID] tool 0.7872 vs frozen 0.7863 -- that
#                estimator had a 0.707 zero-signal limit and was removed in
#                0.2.0.  The corrected estimator (split-half + Spearman-Brown,
#                r_tech = sqrt(reliability)) reproduces
#                analysis_out/ceiling_rtech_CORRECTED.csv (median 0.5073) to
#                within 0.007 per section (A1 0.4524/0.4552, B2 0.6417/0.6469,
#                F3 0.3551/0.3620, n_rep=20 vs single frozen draw).
#   r_full       tool 0.1199 vs frozen 0.1133 / 0.1167 (the two frozen columns
#                disagree with each other by up to 0.027 per section)
#   depth_null   tool 0.2021 with the DEFAULT null_fit='cv_within_section'
#                vs frozen 0.2298; with null_fit='oracle' -> 0.2307 (diff +0.0009,
#                max per-section 0.0108).  The frozen number is an in-section fit.
#   top-100 selection floor: 0.0818 (free perm, median of null) vs frozen 0.0806
#                the tool's default floor is perm_block p95 = 0.1552
