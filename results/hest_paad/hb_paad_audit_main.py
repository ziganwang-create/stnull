# -*- coding: utf-8 -*-
"""PAAD (HEST-bench) zero-pixel audit: H1 libnull, H2 zerobio, H3 structure, H4 ceiling, H5 r_nbr.

Adapted from lymph_audit_main.py (methodology frozen there). PAAD specifics:
- 3 Xenium samples, dense uint16 X, per-sample gene panels differ (intersection=159).
- in_tissue = 100% everywhere -> single spot set "all" (== in_tissue); disclosed.
- lib = obs/total_counts = Xenium panel total (no full transcriptome exists).
- Panel tiers: hest_var50 / trainHVG50 / trainHVGmax (<=159; 737 impossible).
Pre-registered predictions in analysis_out/hb_paad_prereg.md (timestamped BEFORE this run).
ASCII prints only.
"""
import os
os.environ["OMP_NUM_THREADS"] = "4"
import json, sys, time
import h5py
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = r"d:\project\STImage\Xiao_data_codes\data\hest_bench\PAAD"
OUT = r"d:\project\STImage\prototype\analysis_out"
SCR = os.path.dirname(os.path.abspath(__file__))
SAMPLES = ["TENX116", "TENX126", "TENX140"]
HER2ST_FIT = dict(intercept=0.02964491191349654, slope=0.28735535650367705,
                  range=[0.389101802316651, 1.4277986686470003])  # from lymph_panels.json

def read_str(ds):
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in ds[:]])

def load(sid):
    f = h5py.File(os.path.join(ROOT, "adata", sid + ".h5ad"), "r")
    X = np.asarray(f["X"][:], dtype=np.float64)
    lib = f["obs/total_counts"][:].astype(np.float64)
    xy = f["obsm/spatial"][:].astype(np.float64)
    var = read_str(f["var/_index"])
    f.close()
    return X, lib, xy, var

print("loading...")
data = {}
VN = {}
for sid in SAMPLES:
    X, lib, xy, var = load(sid)
    data[sid] = dict(X=X, lib=lib, xy=xy, n2i={g: i for i, g in enumerate(var)})
    VN[sid] = var
print("loaded")

var50 = json.load(open(os.path.join(ROOT, "var_50genes.json")))["genes"]
inter = sorted(set(VN[SAMPLES[0]]) & set(VN[SAMPLES[1]]) & set(VN[SAMPLES[2]]))
print("intersection genes:", len(inter))

# ---------------- fold-wise train-side HVG panels (seurat flavor, lymph_pass2 logic)
def hvg_seurat(mean_raw, var_raw, k, eligible, nbin=20):
    disp = np.full(len(mean_raw), np.nan)
    ok = mean_raw > 0
    disp[ok] = var_raw[ok] / mean_raw[ok]
    with np.errstate(divide="ignore", invalid="ignore"):
        ldisp = np.log(disp)
    lmean = np.log1p(mean_raw)
    m = eligible & ok & np.isfinite(ldisp)
    bins = pd.cut(lmean[m], nbin)
    df = pd.DataFrame({"d": ldisp[m], "b": bins})
    gm = df.groupby("b", observed=True)["d"].mean()
    gs = df.groupby("b", observed=True)["d"].std(ddof=1).fillna(0.0)
    bm = np.asarray(pd.Series(bins).map(gm), float)
    bs = np.asarray(pd.Series(bins).map(gs), float)
    one = (bs == 0) | ~np.isfinite(bs)
    bs = np.where(one, 1.0, bs)
    bm = np.where(one & ~np.isfinite(bm), 0.0, bm)
    z = (df["d"].values - bm) / bs
    score = np.full(len(mean_raw), -np.inf)
    score[m] = z
    return np.argsort(-score)[:k]

def counts_on(sid, genes):
    d = data[sid]
    cols = np.array([d["n2i"][g] for g in genes])
    return d["X"][:, cols]

INTER = np.array(inter)
fold_panels = {}
for test in SAMPLES:
    train = [s for s in SAMPLES if s != test]
    Cs = [counts_on(s, inter) for s in train]
    elig = np.logical_and.reduce([(C > 0).sum(0) >= np.ceil(0.10 * C.shape[0]) for C in Cs])
    Call = np.vstack(Cs)
    mu = Call.mean(0); va = Call.var(0, ddof=1)
    n_el = int(elig.sum())
    hvg50 = list(INTER[hvg_seurat(mu, va, 50, elig)])
    hvgmax = list(INTER[hvg_seurat(mu, va, n_el, elig)])
    fold_panels[test] = dict(hvg50=hvg50, hvgmax=hvgmax, n_eligible=n_el,
                             hvg50_overlap_var50=len(set(hvg50) & set(var50)),
                             hvgmax_overlap_var50=len(set(hvgmax) & set(var50)))
    print("fold test=%s eligible=%d hvg50 overlap var50=%d/50" %
          (test, n_el, fold_panels[test]["hvg50_overlap_var50"]))
json.dump(dict(fold_panels=fold_panels, n_intersection=len(inter),
               her2st_fit=HER2ST_FIT),
          open(os.path.join(SCR, "hb_paad_panels.json"), "w"), indent=1)

CACHE = {}
def get_counts(sid, pname, fold=None):
    key = (sid, pname, fold if pname != "hest_var50" else None)
    if key not in CACHE:
        if pname == "hest_var50":
            genes = var50
        elif pname == "trainHVG50":
            genes = fold_panels[fold]["hvg50"]
        elif pname == "trainHVGmax":
            genes = fold_panels[fold]["hvgmax"]
        CACHE[key] = counts_on(sid, genes)
    return CACHE[key]

def transform(C, caliber, lib=None):
    if caliber == "log1p_raw":
        return np.log1p(C)
    if caliber == "cp10k_panel":
        rs = C.sum(1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            Y = np.log1p(1e4 * C / rs)
        Y[rs.ravel() == 0, :] = 0.0
        return Y
    if caliber == "cp10k_full":
        den = np.maximum(np.asarray(lib, float).reshape(-1, 1), 1.0)
        return np.log1p(1e4 * C / den)
    raise ValueError(caliber)

def pear_cols(A, B):
    A = A - A.mean(0, keepdims=True); B = B - B.mean(0, keepdims=True)
    sa = np.sqrt((A * A).sum(0)); sb = np.sqrt((B * B).sum(0))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (A * B).sum(0) / (sa * sb)
    r[(sa == 0) | (sb == 0)] = np.nan
    return r

def loglib_of(sid):
    return np.log(np.maximum(data[sid]["lib"], 1.0))

# ---------------------------------------------------------------- H1 libnull
print("H1 libnull...")
rows = []
CALIBERS = ["log1p_raw", "cp10k_panel", "cp10k_full"]
PANELS = ["hest_var50", "trainHVG50", "trainHVGmax"]
SPOTSET = "all_eq_in_tissue"  # in_tissue == 100% in all samples
for test in SAMPLES:
    train = [s for s in SAMPLES if s != test]
    for pname in PANELS:
        for cal in CALIBERS:
            Ytr, Ltr = [], []
            for s in train:
                C = get_counts(s, pname, fold=test)
                Ytr.append(transform(C, cal, data[s]["lib"]))
                Ltr.append(loglib_of(s))
            Ytr = np.vstack(Ytr); Ltr = np.concatenate(Ltr)
            coef = np.linalg.lstsq(np.column_stack([np.ones_like(Ltr), Ltr]), Ytr,
                                   rcond=None)[0]
            C = get_counts(test, pname, fold=test)
            Y = transform(C, cal, data[test]["lib"])
            L = loglib_of(test)
            pred = np.column_stack([np.ones_like(L), L]) @ coef
            r = pear_cols(Y, pred)
            # within-section 5-fold CV variant
            n = len(L)
            rng = np.random.default_rng(abs(hash((test, pname, cal))) % 2**31)
            perm = rng.permutation(n); folds = np.array_split(perm, 5)
            predcv = np.full(Y.shape, np.nan)
            Xall = np.column_stack([np.ones_like(L), L])
            for te in folds:
                tr = np.setdiff1d(perm, te)
                B = np.linalg.lstsq(Xall[tr], Y[tr], rcond=None)[0]
                predcv[te] = Xall[te] @ B
            rcv = pear_cols(Y, predcv)
            rows.append(dict(sample=test, panel=pname, spotset=SPOTSET, caliber=cal,
                             n_spots=n, n_genes=Y.shape[1],
                             med_r_trainfit=float(np.nanmedian(r)),
                             mean_r_trainfit=float(np.nanmean(r)),
                             med_r_cv=float(np.nanmedian(rcv)),
                             mean_r_cv=float(np.nanmean(rcv))))
    print("  done", test)
lib_df = pd.DataFrame(rows)
lib_df.to_csv(os.path.join(OUT, "hb_paad_audit_libnull.csv"), index=False)
print(lib_df.round(3).to_string())

# official-analog aggregate: log1p_raw + hest_var50, mean over genes then folds
off = lib_df[(lib_df.panel == "hest_var50") & (lib_df.caliber == "log1p_raw")]
print("OFFICIAL-ANALOG H1 mean-r (fold mean): %.4f  per-fold: %s" %
      (off.mean_r_trainfit.mean(), off.mean_r_trainfit.round(4).tolist()))

# --------------------------------------------------------------- H2 zerobio
print("H2 zerobio...")
rows = []
for pname in ("hest_var50", "trainHVGmax"):
    for test in SAMPLES:
        train = [s for s in SAMPLES if s != test]
        G = get_counts(test, pname, fold=test).shape[1]
        tot = np.zeros(G + 1)
        for s in train:
            C = get_counts(s, pname, fold=test)
            tot[:G] += C.sum(0)
            tot[G] += max(data[s]["lib"].sum() - C.sum(), 0.0)
        p = tot / tot.sum()
        rng = np.random.default_rng(abs(hash(("zb", test, pname))) % 2**31)
        sim = {}
        for s in SAMPLES:
            nvec = data[s]["lib"].astype(np.int64)
            d = rng.multinomial(nvec, p)
            sim[s] = d[:, :G].astype(np.float64)
        for cal in ("log1p_raw", "cp10k_panel"):
            res = {}
            for mode in ("real", "sim"):
                Ytr, Ltr = [], []
                for s in train:
                    C = get_counts(s, pname, fold=test) if mode == "real" else sim[s]
                    Ytr.append(transform(C, cal, data[s]["lib"]))
                    Ltr.append(loglib_of(s))
                Ytr = np.vstack(Ytr); Ltr = np.concatenate(Ltr)
                coef = np.linalg.lstsq(np.column_stack([np.ones_like(Ltr), Ltr]), Ytr,
                                       rcond=None)[0]
                C = get_counts(test, pname, fold=test) if mode == "real" else sim[test]
                Y = transform(C, cal, data[test]["lib"])
                L = loglib_of(test)
                pred = np.column_stack([np.ones_like(L), L]) @ coef
                r = pear_cols(Y, pred)
                res[mode] = (float(np.nanmedian(r)), float(np.nanmean(r)))
            rows.append(dict(sample=test, panel=pname, spotset=SPOTSET, caliber=cal,
                             med_r_real=res["real"][0], med_r_sim=res["sim"][0],
                             mean_r_real=res["real"][1], mean_r_sim=res["sim"][1],
                             sim_ge_real=int(res["sim"][0] >= res["real"][0])))
    print("  done", pname)
zb_df = pd.DataFrame(rows)
zb_df.to_csv(os.path.join(OUT, "hb_paad_audit_zerobio.csv"), index=False)
print(zb_df.round(3).to_string())

# -------------------------------------------------------------- H3 structure
print("H3 structure...")
rows = []
for sid in SAMPLES:
    L = loglib_of(sid)
    sd = float(np.std(L))
    pred_r = HER2ST_FIT["intercept"] + HER2ST_FIT["slope"] * sd
    for pname in ("hest_var50", "trainHVGmax"):
        for cal in ("cp10k_panel", "log1p_raw"):
            obs = lib_df[(lib_df["sample"] == sid) & (lib_df.panel == pname) &
                         (lib_df.caliber == cal)]["med_r_trainfit"].iloc[0]
            rows.append(dict(sample=sid, spotset=SPOTSET, panel=pname, caliber=cal,
                             sd_loglib=sd, pred_r_her2stfit=pred_r,
                             obs_med_r=obs, resid=obs - pred_r,
                             extrapolated=int(sd > HER2ST_FIT["range"][1])))
st_df = pd.DataFrame(rows)
corrs = []
for (pname, cal), g in st_df.groupby(["panel", "caliber"]):
    c = float(np.corrcoef(g.sd_loglib, g.obs_med_r)[0, 1])
    corrs.append(dict(spotset=SPOTSET, panel=pname, caliber=cal, corr_sd_vs_r_n3=c,
                      order_obs=">".join(g.sort_values("obs_med_r", ascending=False)["sample"]),
                      order_pred=">".join(g.sort_values("sd_loglib", ascending=False)["sample"])))
st_df.to_csv(os.path.join(OUT, "hb_paad_audit_structure.csv"), index=False)
pd.DataFrame(corrs).to_csv(os.path.join(OUT, "hb_paad_audit_structure_corr.csv"), index=False)
print(st_df.round(3).to_string())
print(pd.DataFrame(corrs).to_string())

# ------------------------------------------------------ H4 ceiling + H5 r_nbr
print("H4 ceiling...")
sys.path.insert(0, r"d:\project\STImage\stnull\src")
from stnull.ceiling import r_tech as stnull_rtech
from stnull.spaces import TargetSpace

SPACE_OF = {"log1p_raw": TargetSpace.LOG1P_RAW,
            "cp10k_panel": TargetSpace.LOG1P_CP10K_PANEL,
            "cp10k_full": TargetSpace.LOG1P_CP10K_FULL}

def my_rtech(C, cal, lib, n_rep=20, rng=None):
    rng = np.random.default_rng(0) if rng is None else rng
    Ci = np.rint(np.maximum(C, 0)).astype(np.int64)
    rest = np.rint(np.maximum(lib - Ci.sum(1), 0)).astype(np.int64)
    RH = []
    for _ in range(n_rep):
        H1 = rng.binomial(Ci, 0.5).astype(np.float64); H2 = Ci - H1
        R1 = rng.binomial(rest, 0.5).astype(np.float64)
        l1 = H1.sum(1) + R1; l2 = H2.sum(1) + (rest - R1)
        Y1 = transform(H1, cal, l1); Y2 = transform(H2, cal, l2)
        RH.append(pear_cols(Y1, Y2))
    rh = np.nanmedian(np.vstack(RH), 0)
    with np.errstate(invalid="ignore"):
        rel = np.clip(2 * rh / (1 + rh), 0, 1)
    return np.sqrt(rel)

def r_nbr(Y, xy, k=6):
    tree = cKDTree(xy)
    _, idx = tree.query(xy, k=k + 1)
    idx = idx[:, 1:]
    pred = Y[idx].mean(1)
    return pear_cols(Y, pred)

rows = []
for pname in ("hest_var50", "trainHVGmax"):
    for sid in SAMPLES:
        C = get_counts(sid, pname, fold=sid)
        lib = data[sid]["lib"]
        xy = data[sid]["xy"]
        for cal in ("log1p_raw", "cp10k_panel", "cp10k_full"):
            seed = abs(hash(("ceil", sid, pname, cal))) % 2**31
            mine = my_rtech(C, cal, lib, rng=np.random.default_rng(seed))
            rt, _, method, _ = stnull_rtech(C, SPACE_OF[cal], lib_full=lib,
                                            n_rep=20, n_perm=0,
                                            rng=np.random.default_rng(seed + 1))
            Y = transform(C, cal, lib)
            rn = r_nbr(Y, xy, k=6)
            rows.append(dict(sample=sid, panel=pname, spotset=SPOTSET, caliber=cal,
                             rtech_med_mine=float(np.nanmedian(mine)),
                             rtech_med_stnull=float(np.nanmedian(rt)),
                             rtech_q25=float(np.nanpercentile(rt, 25)),
                             rtech_q75=float(np.nanpercentile(rt, 75)),
                             rnbr_med=float(np.nanmedian(rn)),
                             rnbr_mean=float(np.nanmean(rn)),
                             method=method))
    print("  done", pname)
ce_df = pd.DataFrame(rows)
ce_df.to_csv(os.path.join(OUT, "hb_paad_audit_ceiling.csv"), index=False)
print(ce_df.round(3).to_string())
print("ALL DONE")
