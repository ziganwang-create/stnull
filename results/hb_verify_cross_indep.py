# -*- coding: utf-8 -*-
"""Cross-cohort independent spot-check of official-caliber H1 libnull numbers.

INDEPENDENT code path vs the audit mains:
 - folds parsed from splits/*.csv directly (mains hardcode fold dicts),
 - per-gene fit via closed-form slope=cov(L,y)/var(L) (mains use np.linalg.lstsq),
 - per-gene Pearson via scipy.stats.pearsonr in a loop (mains use vectorized pear_cols),
 - CSR loading via scipy csr_matrix + column slicing.

Targets (recorded values to reproduce, tolerance 0.01):
  LUNG  0.6942 | HCC 0.1484 | COAD 0.3218 (patch subset) | PRAD 0.5401
  CCRCC 0.4945 | IDC 0.4284 (patch subset)
Also computes full-matrix sparsity (zero fraction) per cohort for the master table.
ASCII prints only.
"""
import os
os.environ["OMP_NUM_THREADS"] = "4"
import glob
import json
import h5py
import numpy as np
from scipy.stats import pearsonr
from scipy.sparse import csr_matrix

ROOT = r"d:\project\STImage\Xiao_data_codes\data\hest_bench"
RECORDED = {"LUNG": 0.6942, "HCC": 0.1484, "COAD": 0.3218, "PRAD": 0.5401,
            "CCRCC": 0.4945, "IDC": 0.4284}
PATCH_SUBSET = {"COAD", "IDC"}          # cohorts where official spot set = patch barcodes
SPARSITY_ONLY = ["READ", "PAAD", "SKCM", "LYMPH_IDC"]


def rstr(ds):
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in ds[:]])


def load_sample(task, sid, genes):
    """Return counts (n,|genes|) float64, lib, patch mask, (n_spots, zero_frac)."""
    p = os.path.join(ROOT, task, "adata", sid + ".h5ad")
    f = h5py.File(p, "r")
    var = rstr(f["var/_index"])
    gi = {g: i for i, g in enumerate(var)}
    cols = np.array([gi[g] for g in genes])
    x = f["X"]
    if isinstance(x, h5py.Group):
        shape = tuple(x.attrs["shape"])
        M = csr_matrix((x["data"][:], x["indices"][:], x["indptr"][:]), shape=shape)
        zero_frac = 1.0 - M.nnz / float(shape[0] * shape[1])
        C = np.asarray(M[:, cols].todense(), dtype=np.float64)
        n = shape[0]
    else:
        D = x[:]
        n = D.shape[0]
        zero_frac = 1.0 - np.count_nonzero(D) / float(D.size)
        C = D[:, cols].astype(np.float64)
    lib = f["obs/total_counts"][:].astype(np.float64)
    obs = rstr(f["obs/_index"])
    f.close()
    g = h5py.File(os.path.join(ROOT, task, "patches", sid + ".h5"), "r")
    pb = set(x.decode() if isinstance(x, bytes) else str(x)
             for x in g["barcode"][:].ravel())
    g.close()
    mask = np.array([b in pb for b in obs])
    return C, lib, mask, (n, zero_frac)


def sparsity_of(task):
    """Cheap full-matrix zero fraction across all samples of a task."""
    tot_zero = 0.0
    tot_size = 0.0
    for p in sorted(glob.glob(os.path.join(ROOT, task, "adata", "*.h5ad"))):
        f = h5py.File(p, "r")
        x = f["X"]
        if isinstance(x, h5py.Group):
            shape = tuple(x.attrs["shape"])
            size = float(shape[0] * shape[1])
            nnz = float(x["data"].shape[0])
            tot_zero += size - nnz
            tot_size += size
        else:
            D = x[:]
            tot_zero += D.size - np.count_nonzero(D)
            tot_size += float(D.size)
        f.close()
    return tot_zero / tot_size


def folds_of(task):
    ks = sorted(int(os.path.basename(p).split("_")[1].split(".")[0])
                for p in glob.glob(os.path.join(ROOT, task, "splits", "test_*.csv")))
    folds = {}
    for k in ks:
        def ids(kind):
            path = os.path.join(ROOT, task, "splits", "%s_%d.csv" % (kind, k))
            with open(path) as fh:
                rows = [l.strip().split(",") for l in fh if l.strip()]
            hdr = rows[0]
            j = hdr.index("sample_id")
            return [r[j] for r in rows[1:]]
        folds[k] = dict(train=ids("train"), test=ids("test"))
    return folds


def official_libnull(task):
    genes = json.load(open(os.path.join(ROOT, task, "var_50genes.json")))["genes"]
    folds = folds_of(task)
    sids = sorted({s for v in folds.values() for s in v["train"] + v["test"]})
    data = {}
    zf_num, zf_den = 0.0, 0.0
    for s in sids:
        C, lib, mask, (n, zf) = load_sample(task, s, genes)
        if task not in PATCH_SUBSET:
            mask = np.ones(n, bool)
        data[s] = (C[mask], lib[mask])
        zf_num += zf * n
        zf_den += n
    fold_scores = []
    for k in sorted(folds):
        Ltr = np.concatenate([np.log(np.maximum(data[s][1], 1.0))
                              for s in folds[k]["train"]])
        Ytr = np.log1p(np.vstack([data[s][0] for s in folds[k]["train"]]))
        Lm = Ltr.mean()
        vL = ((Ltr - Lm) ** 2).mean()
        slope = ((Ltr - Lm)[:, None] * (Ytr - Ytr.mean(0))).mean(0) / vL
        intercept = Ytr.mean(0) - slope * Lm
        Lte = np.concatenate([np.log(np.maximum(data[s][1], 1.0))
                              for s in folds[k]["test"]])
        Yte = np.log1p(np.vstack([data[s][0] for s in folds[k]["test"]]))
        P = intercept[None, :] + Lte[:, None] * slope[None, :]
        rs = []
        for gidx in range(Yte.shape[1]):
            y = Yte[:, gidx]
            if y.std() == 0 or P[:, gidx].std() == 0:
                rs.append(np.nan)
            else:
                rs.append(pearsonr(y, P[:, gidx])[0])
        fold_scores.append(float(np.nanmean(rs)))
    off = float(np.mean(fold_scores))
    return off, fold_scores, zf_num / zf_den


print("cohort, recomputed, recorded, delta, fold_scores, sparsity")
results = {}
for task in ["LUNG", "HCC", "COAD", "PRAD", "CCRCC", "IDC"]:
    off, fs, zf = official_libnull(task)
    d = off - RECORDED[task]
    flag = "OK" if abs(d) <= 0.01 else "MISMATCH"
    results[task] = dict(recomputed=round(off, 4), recorded=RECORDED[task],
                         delta=round(d, 4), folds=[round(x, 4) for x in fs],
                         sparsity=round(zf, 4), flag=flag)
    print("%s  %.4f  vs %.4f  d=%+.4f  %s  folds=%s  sparsity=%.4f"
          % (task, off, RECORDED[task], d, flag,
             ",".join("%.4f" % x for x in fs), zf))

for task in SPARSITY_ONLY:
    zf = sparsity_of(task)
    results[task] = dict(sparsity=round(zf, 4))
    print("%s  sparsity=%.4f" % (task, zf))

out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "verify_cross_indep_results.json")
json.dump(results, open(out, "w"), indent=1)
print("saved", out)
