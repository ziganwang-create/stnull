# -*- coding: utf-8 -*-
"""Inspect PAAD (HEST-bench) -- structure + depth pre-table. ASCII prints only.
Adapted from lymph_inspect.py. PAAD is XENIUM (3 samples, panel ~540 genes, dense X).
"""
import os
os.environ["OMP_NUM_THREADS"] = "4"
import json
import h5py
import numpy as np
import pandas as pd

ROOT = r"d:\project\STImage\Xiao_data_codes\data\hest_bench\PAAD"
OUT = r"d:\project\STImage\prototype\analysis_out"
SCRATCH = os.path.dirname(os.path.abspath(__file__))
SAMPLES = ["TENX116", "TENX126", "TENX140"]


def read_str(ds):
    a = ds[:]
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in a])


def load(sid):
    p = os.path.join(ROOT, "adata", sid + ".h5ad")
    f = h5py.File(p, "r")
    X = np.asarray(f["X"][:], dtype=np.float64)  # dense uint16
    obs = {}
    for k in f["obs"].keys():
        o = f["obs"][k]
        if isinstance(o, h5py.Dataset):
            obs[k] = read_str(o) if o.dtype == object else o[:]
        else:  # categorical group
            cats = read_str(o["categories"]); codes = o["codes"][:]
            obs[k] = cats[codes]
    var_names = read_str(f["var/_index"])
    mito = f["var/mito"][:]
    spatial = f["obsm/spatial"][:]
    uns_keys = []
    def walk(name):
        uns_keys.append(name)
    if "uns" in f:
        f["uns"].visit(walk)
    f.close()
    return X, obs, var_names, mito, spatial, uns_keys


def morans_I(v, coords, k=6):
    from scipy.spatial import cKDTree
    v = np.asarray(v, float)
    n = len(v)
    tree = cKDTree(coords)
    _, idx = tree.query(coords, k=k + 1)
    idx = idx[:, 1:]
    z = v - v.mean()
    s2 = (z * z).sum()
    if s2 <= 0:
        return np.nan
    num = (z[:, None] * z[idx]).sum()
    W = n * k
    return (n / W) * (num / s2)


def main():
    rows = []
    store = {}
    var50 = json.load(open(os.path.join(ROOT, "var_50genes.json")))["genes"]
    mean50 = json.load(open(os.path.join(ROOT, "mean_50genes.json")))["genes"]

    VNs = {}
    for sid in SAMPLES:
        X, obs, var_names, mito, spatial, uns_keys = load(sid)
        VNs[sid] = var_names
        n, g = X.shape
        is_int = np.all(np.abs(X - np.round(X)) < 1e-6)
        lib_full = X.sum(1)
        nnz_row = (X > 0).sum(1)
        name2i = {nm: i for i, nm in enumerate(var_names)}
        idx50 = np.array([gn for gn in var50 if gn in name2i])
        miss50 = [gn for gn in var50 if gn not in name2i]
        missm50 = [gn for gn in mean50 if gn not in name2i]
        P50 = X[:, [name2i[gn] for gn in var50 if gn in name2i]]
        lib50 = P50.sum(1)
        intis = obs["in_tissue"].astype(int)
        sd_log_lib = float(np.std(np.log(np.maximum(lib_full, 1.0))))
        sd_log_lib_tis = float(np.std(np.log(np.maximum(lib_full[intis == 1], 1.0)))) if (intis == 1).any() else np.nan
        xy = spatial.astype(float)
        mi_lib = morans_I(np.log(np.maximum(lib_full, 1.0)), xy, 6)

        rows.append(dict(
            sample=sid, n_spots=n, n_genes=g,
            n_in_tissue=int((intis == 1).sum()),
            frac_in_tissue=float((intis == 1).mean()),
            n_umi_total=float(lib_full.sum()),
            lib_median=float(np.median(lib_full)),
            lib_p10=float(np.percentile(lib_full, 10)),
            lib_p90=float(np.percentile(lib_full, 90)),
            sd_log_lib_all=sd_log_lib,
            sd_log_lib_in_tissue=sd_log_lib_tis,
            cv_lib=float(np.std(lib_full) / np.mean(lib_full)),
            lib50_median=float(np.median(lib50)),
            zero_frac_panel50=float((P50 == 0).mean()),
            le1_frac_panel50=float((P50 <= 1).mean()),
            zero_frac_fullpanel=float((X == 0).mean()),
            genes_detected_median=float(np.median(nnz_row)),
            morans_I_loglib_k6=float(mi_lib),
        ))
        store[sid] = dict(
            in_tissue_values=np.unique(intis).tolist(),
            is_integer=bool(is_int), xmax=float(X.max()),
            obs_total_counts_matches=bool(np.allclose(obs["total_counts"].astype(float), lib_full, rtol=1e-3)),
            barcodes_head=obs["_index"][:3].tolist(),
            n_unique_barcodes=int(len(np.unique(obs["_index"]))),
            mito_genes=int(np.asarray(mito).sum()),
            miss_var50=miss50, miss_mean50=missm50,
            dup_genes=int(len(var_names) - len(set(var_names))),
            array_row=(int(obs["array_row"].min()), int(obs["array_row"].max())),
            array_col=(int(obs["array_col"].min()), int(obs["array_col"].max())),
            spatial_col0_is_pxlcol=bool(np.array_equal(spatial[:, 0], obs["pxl_col_in_fullres"])),
            uns_keys=uns_keys[:10],
        )
        print("done", sid, n, g, "int=", bool(is_int), "in_tissue_frac=",
              round(float((intis == 1).mean()), 4), "sdloglib=", round(sd_log_lib, 4))

    # gene universe intersection
    inter = set(VNs[SAMPLES[0]])
    for sid in SAMPLES[1:]:
        inter &= set(VNs[sid])
    union = set().union(*[set(VNs[s]) for s in SAMPLES])
    store["_gene_universe"] = dict(
        n_intersection=len(inter), n_union=len(union),
        per_sample={s: len(VNs[s]) for s in SAMPLES},
        var50_in_intersection=int(len(set(var50) & inter)),
        mean50_in_intersection=int(len(set(mean50) & inter)),
        only_in={s: sorted(set(VNs[s]) - inter)[:20] for s in SAMPLES},
        n_only_in={s: len(set(VNs[s]) - inter) for s in SAMPLES},
    )
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "hb_paad_depth_pretable.csv"), index=False)
    json.dump(store, open(os.path.join(SCRATCH, "hb_paad_struct.json"), "w"), indent=1, default=str)
    print(df.to_string())
    print(json.dumps(store["_gene_universe"], indent=1, default=str))
    print("VAR50 all present in intersection:", len(set(var50) & inter) == 50)


if __name__ == "__main__":
    main()
