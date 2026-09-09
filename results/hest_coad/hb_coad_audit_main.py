# -*- coding: utf-8 -*-
"""COAD zero-pixel audit: H1 libnull, H2 zerobio, H3 structure, H4 ceiling + H5 r_nbr.

Adapted from hb_read_audit.py + skcm_audit_main.py (verified pipelines). COAD specifics:
 - 4 Xenium samples, dense uint16 X, two gene panels (412 common); trainHVG panels are
   restricted to the common 412 (hb_coad_panels.json, per fold per spot set).
 - spot sets: "all" (in_tissue degenerate) and "official" (patch barcodes, 83-90 pct).
 - folds (official CSVs): fold0 train={TENX111} test={149,148,147}; fold1 reverse.
   No patient-level leakage under either mapping (hb_coad_struct.json) -> no corrected
   rerun needed.
 - official-metric aggregation: per fold, concatenate test samples (official spot set,
   var50, log1p raw), per-gene pearson, mean over genes, mean over folds (matches bench
   trainer.py + benchmark.py merge_fold_results).
Pre-registered predictions: hb_coad_prereg.json (2026-08-24T03:25:53Z, before this run).
ASCII prints only.
"""
import os
os.environ["OMP_NUM_THREADS"] = "4"
import json, sys, time
import h5py
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = r"d:\project\STImage\Xiao_data_codes\data\hest_bench\COAD"
OUT = r"d:\project\STImage\prototype\analysis_out"
SAMPLES = ["TENX111", "TENX147", "TENX148", "TENX149"]
FOLDS = {0: dict(train=["TENX111"], test=["TENX149", "TENX148", "TENX147"]),
         1: dict(train=["TENX149", "TENX148", "TENX147"], test=["TENX111"])}
FOLD_OF = {s: k for k, v in FOLDS.items() for s in v["test"]}


def read_str(ds):
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in ds[:]])


def load(sid):
    f = h5py.File(os.path.join(ROOT, "adata", sid + ".h5ad"), "r")
    X = f["X"][:].astype(np.float64)
    lib = f["obs/total_counts"][:].astype(np.float64)
    obs_idx = read_str(f["obs/_index"])
    xy = np.stack([f["obs/pxl_col_in_fullres"][:], f["obs/pxl_row_in_fullres"][:]], 1).astype(float)
    var = read_str(f["var/_index"])
    f.close()
    g = h5py.File(os.path.join(ROOT, "patches", sid + ".h5"), "r")
    pb = set(x.decode() if isinstance(x, bytes) else str(x) for x in g["barcode"][:].ravel())
    g.close()
    off = np.array([b in pb for b in obs_idx])
    return X, lib, xy, var, off


print("loading...")
data = {}
for sid in SAMPLES:
    X, lib, xy, var, off = load(sid)
    data[sid] = dict(X=X, lib=lib, xy=xy, off=off,
                     gidx={g: i for i, g in enumerate(var)})
    print(" ", sid, X.shape, "official", int(off.sum()))
print("loaded")

var50 = json.load(open(os.path.join(ROOT, "var_50genes.json")))["genes"]
fold_panels = json.load(open(os.path.join(OUT, "hb_coad_panels.json")))["fold_panels"]

SPOTSETS = ["all", "official"]
CALIBERS = ["log1p_raw", "cp10k_panel", "cp10k_full"]
PANELS = ["hest_var50", "trainHVG50", "trainHVG737"]


def genes_of(pname, fold, spotset):
    if pname == "hest_var50":
        return var50
    key = "fold%d_%s" % (fold, spotset)
    return fold_panels[key]["hvg50" if pname == "trainHVG50" else "hvg737"]


CACHE = {}
def get_counts(sid, pname, fold, spotset):
    if pname == "hest_var50":
        key = (sid, pname)
        genes = var50
    else:
        key = (sid, pname, fold, spotset)
        genes = genes_of(pname, fold, spotset)
    if key not in CACHE:
        gi = data[sid]["gidx"]
        cols = np.array([gi[g] for g in genes])  # panels restricted to common genes
        CACHE[key] = data[sid]["X"][:, cols]
    return CACHE[key]


def mask_of(sid, spotset):
    if spotset == "all":
        return np.ones(data[sid]["X"].shape[0], bool)
    return data[sid]["off"]


def transform(C, caliber, lib=None):
    if caliber == "log1p_raw":
        return np.log1p(C)
    if caliber == "cp10k_panel":
        rsum = C.sum(1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            Y = np.log1p(1e4 * C / rsum)
        Y[rsum.ravel() == 0, :] = 0.0
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


def loglib_of(sid, m):
    return np.log(np.maximum(data[sid]["lib"][m], 1.0))


# ---------------------------------------------------------------- H1 libnull
print("H1 libnull...")
rows = []
fold_rows = []
for k, fv in FOLDS.items():
    train, test_sids = fv["train"], fv["test"]
    for pname in PANELS:
        for ss in SPOTSETS:
            for cal in CALIBERS:
                Ytr, Ltr = [], []
                for s in train:
                    m = mask_of(s, ss)
                    C = get_counts(s, pname, k, ss)[m]
                    Ytr.append(transform(C, cal, data[s]["lib"][m]))
                    Ltr.append(loglib_of(s, m))
                Ytr = np.vstack(Ytr); Ltr = np.concatenate(Ltr)
                coef = np.linalg.lstsq(np.column_stack([np.ones_like(Ltr), Ltr]), Ytr,
                                       rcond=None)[0]
                Yte_cat, Pte_cat = [], []
                for s in test_sids:
                    m = mask_of(s, ss)
                    C = get_counts(s, pname, k, ss)[m]
                    Y = transform(C, cal, data[s]["lib"][m])
                    L = loglib_of(s, m)
                    pred = np.column_stack([np.ones_like(L), L]) @ coef
                    Yte_cat.append(Y); Pte_cat.append(pred)
                    r = pear_cols(Y, pred)
                    # within-section 5-fold CV (stnull-comparable)
                    n = len(L)
                    rng = np.random.default_rng(abs(hash((s, pname, ss, cal))) % 2**31)
                    perm = rng.permutation(n); cvf = np.array_split(perm, 5)
                    predcv = np.full(Y.shape, np.nan)
                    Xall = np.column_stack([np.ones_like(L), L])
                    for te in cvf:
                        tr = np.setdiff1d(perm, te)
                        B = np.linalg.lstsq(Xall[tr], Y[tr], rcond=None)[0]
                        predcv[te] = Xall[te] @ B
                    rcv = pear_cols(Y, predcv)
                    rows.append(dict(sample=s, fold=k, panel=pname, spotset=ss,
                                     caliber=cal, n_spots=n, n_genes=Y.shape[1],
                                     n_genes_nan=int(np.isnan(r).sum()),
                                     med_r_trainfit=float(np.nanmedian(r)),
                                     mean_r_trainfit=float(np.nanmean(r)),
                                     med_r_cv=float(np.nanmedian(rcv)),
                                     mean_r_cv=float(np.nanmean(rcv))))
                # official aggregation: concat test samples of the fold
                Yc = np.vstack(Yte_cat); Pc = np.vstack(Pte_cat)
                rf = pear_cols(Yc, Pc)
                fold_rows.append(dict(fold=k, panel=pname, spotset=ss, caliber=cal,
                                      n_spots=Yc.shape[0], n_genes=Yc.shape[1],
                                      mean_r=float(np.nanmean(rf)),
                                      med_r=float(np.nanmedian(rf)),
                                      n_nan=int(np.isnan(rf).sum())))
    print("  fold", k, "done")
lib_df = pd.DataFrame(rows)
lib_df.to_csv(os.path.join(OUT, "hb_coad_libnull.csv"), index=False)
fold_df = pd.DataFrame(fold_rows)
off = (fold_df.groupby(["panel", "spotset", "caliber"])
       .agg(mean_r_foldavg=("mean_r", "mean"), med_r_foldavg=("med_r", "mean"))
       .reset_index())
fold_df.to_csv(os.path.join(OUT, "hb_coad_libnull_folds.csv"), index=False)
off.to_csv(os.path.join(OUT, "hb_coad_libnull_official.csv"), index=False)
print(lib_df[lib_df.panel == "hest_var50"].round(3).to_string())
print("--- official-style fold-averaged ---")
print(off.round(4).to_string())
OFFICIAL_NULL = float(off[(off.panel == "hest_var50") & (off.spotset == "official") &
                          (off.caliber == "log1p_raw")]["mean_r_foldavg"].iloc[0])
print("OFFICIAL-CALIBER LIBNULL mean-r (fold-concat, 2-fold avg): %.4f" % OFFICIAL_NULL)

# leaderboard confrontation (gen4_hest README.md lines 108-133, COAD column)
BOARD = [("H-Optimus-1", 0.3195), ("GenBio-PathFM", 0.3284), ("H-Optimus-0", 0.3086),
         ("UNI2-h", 0.3015), ("Virchow", 0.3079), ("Virchow2", 0.2581),
         ("Midnight-12k", 0.2908), ("H0-mini", 0.2494), ("OpenMidnight", 0.2728),
         ("Hibou-L", 0.3040), ("GigaPath", 0.2992), ("UNI", 0.2614),
         ("CONCH v1.5", 0.2802), ("GPFM", 0.2480), ("Phikon-v2", 0.2500),
         ("Kaiko ViT-B/8", 0.2683), ("CONCH v1", 0.2489), ("Lunit ViT-S/8", 0.2826),
         ("Phikon", 0.2623), ("Kaiko ViT-B/16", 0.2812), ("Kaiko ViT-L/14", 0.2535),
         ("Kaiko ViT-S/8", 0.2281), ("Kaiko ViT-S/16", 0.2057), ("CTransPath", 0.2382),
         ("MUSK", 0.2365), ("ResNet50", 0.2500)]
lb = pd.DataFrame(BOARD, columns=["model", "coad_r"])
lb["source"] = "gen4_hest README leaderboard (Ridge+PCA256 on patch features)"
null_row = pd.DataFrame([dict(
    model="ZERO-PIXEL library-size null (H1, ours)", coad_r=round(OFFICIAL_NULL, 4),
    source="hb_coad_libnull_official.csv: hest_var50/log1p_raw/official spots, "
           "fold-concat per-gene Pearson mean over genes then over 2 folds")])
lb = pd.concat([null_row, lb], ignore_index=True).sort_values(
    "coad_r", ascending=False).reset_index(drop=True)
lb["rank"] = np.arange(1, len(lb) + 1)
lb["beats_null"] = (lb.coad_r > OFFICIAL_NULL).astype(int)
lb.to_csv(os.path.join(OUT, "hb_coad_leaderboard_compare.csv"), index=False)
print(lb.to_string())
print("models beaten by null: %d / 26" % int((lb.coad_r < OFFICIAL_NULL).sum()))

# --------------------------------------------------------------- H2 zerobio
print("H2 zerobio...")
rows = []
for pname in ("hest_var50", "trainHVG737"):
    for ss in SPOTSETS:
        for k, fv in FOLDS.items():
            train, test_sids = fv["train"], fv["test"]
            genes = genes_of(pname, k, ss)
            G = len(genes)
            tot = np.zeros(G + 1)
            for s in train:
                m = mask_of(s, ss)
                C = get_counts(s, pname, k, ss)[m]
                tot[:G] += C.sum(0)
                tot[G] += max(data[s]["lib"][m].sum() - C.sum(), 0.0)
            p = tot / tot.sum()
            rng = np.random.default_rng(abs(hash(("zb", k, pname, ss))) % 2**31)
            sim = {}
            for s in train + test_sids:
                m = mask_of(s, ss)
                nvec = data[s]["lib"][m].astype(np.int64)
                d = rng.multinomial(nvec, p)
                sim[s] = d[:, :G].astype(np.float64)
            for cal in ("log1p_raw", "cp10k_panel"):
                res = {}
                for mode in ("real", "sim"):
                    Ytr, Ltr = [], []
                    for s in train:
                        m = mask_of(s, ss)
                        C = get_counts(s, pname, k, ss)[m] if mode == "real" else sim[s]
                        Ytr.append(transform(C, cal, data[s]["lib"][m]))
                        Ltr.append(loglib_of(s, m))
                    Ytr = np.vstack(Ytr); Ltr = np.concatenate(Ltr)
                    coef = np.linalg.lstsq(np.column_stack([np.ones_like(Ltr), Ltr]),
                                           Ytr, rcond=None)[0]
                    per = {}
                    for s in test_sids:
                        m = mask_of(s, ss)
                        C = get_counts(s, pname, k, ss)[m] if mode == "real" else sim[s]
                        Y = transform(C, cal, data[s]["lib"][m])
                        L = loglib_of(s, m)
                        pred = np.column_stack([np.ones_like(L), L]) @ coef
                        r = pear_cols(Y, pred)
                        per[s] = (float(np.nanmedian(r)), float(np.nanmean(r)))
                    res[mode] = per
                for s in test_sids:
                    rows.append(dict(sample=s, fold=k, panel=pname, spotset=ss,
                                     caliber=cal,
                                     med_r_real=res["real"][s][0],
                                     med_r_sim=res["sim"][s][0],
                                     mean_r_real=res["real"][s][1],
                                     mean_r_sim=res["sim"][s][1],
                                     sim_ge_real=int(res["sim"][s][0] >= res["real"][s][0])))
        print("  done", pname, ss)
zb_df = pd.DataFrame(rows)
zb_df.to_csv(os.path.join(OUT, "hb_coad_zerobio.csv"), index=False)
print(zb_df.round(3).to_string())

# -------------------------------------------------------------- H3 structure
print("H3 structure...")
FIT = json.load(open(os.path.join(OUT, "lymph_panels.json")))["her2st_fit"]["sd_loglib"]
pre = json.load(open(os.path.join(OUT, "hb_coad_prereg.json")))
rows = []
for ss in SPOTSETS:
    for sid in SAMPLES:
        m = mask_of(sid, ss)
        L = loglib_of(sid, m)
        sd = float(np.std(L))
        pred_r = FIT["intercept"] + FIT["slope"] * sd
        for pname in ("hest_var50", "trainHVG737"):
            for cal in ("cp10k_panel", "log1p_raw"):
                obs = lib_df[(lib_df["sample"] == sid) & (lib_df.panel == pname) &
                             (lib_df.spotset == ss) & (lib_df.caliber == cal)
                             ]["med_r_trainfit"].iloc[0]
                rows.append(dict(sample=sid, spotset=ss, panel=pname, caliber=cal,
                                 sd_loglib=sd, pred_r_her2stfit=pred_r,
                                 obs_med_r=obs, resid=obs - pred_r,
                                 extrapolated=int(sd > FIT["range"][1])))
st_df = pd.DataFrame(rows)
ords = []
for (ss, pname, cal), g in st_df.groupby(["spotset", "panel", "caliber"]):
    order_obs = list(g.sort_values("obs_med_r", ascending=False)["sample"])
    order_pred = list(g.sort_values("sd_loglib", ascending=False)["sample"])
    ords.append(dict(spotset=ss, panel=pname, caliber=cal,
                     corr_sd_vs_r_n4=float(np.corrcoef(g.sd_loglib, g.obs_med_r)[0, 1]),
                     order_obs=">".join(order_obs),
                     order_pred=">".join(order_pred),
                     order_prereg=">".join(pre["predictions"][ss]["order_by_sd_desc"]),
                     order_match=int(order_obs == order_pred)))
st_df.to_csv(os.path.join(OUT, "hb_coad_structure.csv"), index=False)
pd.DataFrame(ords).to_csv(os.path.join(OUT, "hb_coad_structure_corr.csv"), index=False)
print(st_df.round(3).to_string())
print(pd.DataFrame(ords).to_string())
print("NOTE: n=4 samples per condition; ordering is the primary test.")

# ------------------------------------------------------ H4 ceiling + H5 r_nbr
print("H4 ceiling + H5 r_nbr...")
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
for pname in ("hest_var50", "trainHVG737"):
    for ss in SPOTSETS:
        for sid in SAMPLES:
            m = mask_of(sid, ss)
            C = get_counts(sid, pname, FOLD_OF[sid], ss)[m]
            lib = data[sid]["lib"][m]
            xy = data[sid]["xy"][m]
            for cal in CALIBERS:
                seed = abs(hash(("ceil", sid, pname, ss, cal))) % 2**31
                mine = my_rtech(C, cal, lib, rng=np.random.default_rng(seed))
                rt, _, method, _ = stnull_rtech(C, SPACE_OF[cal], lib_full=lib,
                                                n_rep=20, n_perm=0,
                                                rng=np.random.default_rng(seed + 1))
                Y = transform(C, cal, lib)
                rn = r_nbr(Y, xy, k=6)
                rows.append(dict(sample=sid, panel=pname, spotset=ss, caliber=cal,
                                 rtech_med_mine=float(np.nanmedian(mine)),
                                 rtech_med_stnull=float(np.nanmedian(rt)),
                                 rtech_q25=float(np.nanpercentile(rt, 25)),
                                 rtech_q75=float(np.nanpercentile(rt, 75)),
                                 rnbr_med=float(np.nanmedian(rn)),
                                 rnbr_mean=float(np.nanmean(rn)),
                                 method=method))
        print("  done", pname, ss)
ce_df = pd.DataFrame(rows)
ce_df.to_csv(os.path.join(OUT, "hb_coad_ceiling.csv"), index=False)
print(ce_df[ce_df.panel == "hest_var50"].round(3).to_string())
print("ALL DONE", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
