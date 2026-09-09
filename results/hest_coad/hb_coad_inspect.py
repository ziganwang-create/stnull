# -*- coding: utf-8 -*-
"""COAD (HEST-bench) inspect: structure + patient mapping + depth pretable + prereg.

Adapted from hb_read_inspect.py + skcm patterns (verified scripts). ASCII prints only.
COAD specifics (probed 2026-08-24, coad_probe.py):
 - 4 Xenium samples, dense uint16 X, 541 genes each, but TWO different panels:
   TENX111 (colon preview panel) vs TENX147/148/149 (CRC study panel); 412 genes common.
 - in_tissue degenerate (all True) -> spot sets "all" and "official" (patch barcodes,
   83-90 pct of obs).
 - splits are (dataset_title, patient)-level per HESTData.py:1249:
   fold0 train={TENX111} test={149,148,147}; fold1 reverse.
 - HEST metadata marks 147/148/149 all as "Patient 1" but subseries = Sample P1/P2/P5 CRC
   (three DISTINCT patients in the source study; metadata patient column is a mislabel).
Outputs (analysis_out, prefix hb_coad_):
  hb_coad_struct.json          structure findings incl. patient mapping argument
  hb_coad_depth_pretable.csv   per sample x spot-set depth pre-table
  hb_coad_panels.json          fold-wise train-side HVG panels (per spot set)
  hb_coad_prereg.json          TIMESTAMPED pre-registered predictions + bands
"""
import os
os.environ["OMP_NUM_THREADS"] = "4"
import json, time
import h5py
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = r"d:\project\STImage\Xiao_data_codes\data\hest_bench\COAD"
OUT = r"d:\project\STImage\prototype\analysis_out"
SAMPLES = ["TENX111", "TENX147", "TENX148", "TENX149"]
# HEST_v1_1_0.csv lines 10-12 + 151: patient column vs subseries
META_PATIENT = {"TENX149": "Patient 1", "TENX148": "Patient 1",
                "TENX147": "Patient 1", "TENX111": "(empty)"}
SUBSERIES = {"TENX149": "Xenium In Situ, Sample P1 CRC (Stage II-A)",
             "TENX148": "Xenium In Situ, Sample P2 CRC",
             "TENX147": "Xenium In Situ, Sample P5 CRC (Stage IV-A)",
             "TENX111": "Human Colon Preview Data, Xenium colon panel, "
                        "stage 2A adenocarcinoma (separate 10x dataset, 8/2023)"}
# best-supported true patient mapping (subseries): three distinct CRC patients + one
# unknown preview donor
TRUE_PATIENT = {"TENX149": "CRCstudy_P1", "TENX148": "CRCstudy_P2",
                "TENX147": "CRCstudy_P5", "TENX111": "Preview_donor_unknown"}


def rs(ds):
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in ds[:]])


def load(sid):
    f = h5py.File(os.path.join(ROOT, "adata", sid + ".h5ad"), "r")
    X = f["X"][:].astype(np.float64)  # dense uint16 counts
    obs_idx = rs(f["obs/_index"])
    tc = f["obs/total_counts"][:].astype(np.float64)
    it = rs(f["obs/in_tissue"]) if f["obs/in_tissue"].dtype == object else f["obs/in_tissue"][:]
    it = np.array([1 if str(x) in ("True", "1", "b'True'") else 0 for x in it])
    xy = np.stack([f["obs/pxl_col_in_fullres"][:], f["obs/pxl_row_in_fullres"][:]], 1).astype(float)
    vn = rs(f["var/_index"])
    f.close()
    g = h5py.File(os.path.join(ROOT, "patches", sid + ".h5"), "r")
    pb = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in g["barcode"][:].ravel()])
    g.close()
    return X, it, obs_idx, tc, xy, vn, pb


def moran(v, coords, k=6):
    v = np.asarray(v, float); n = len(v)
    _, idx = cKDTree(coords).query(coords, k=k + 1)
    idx = idx[:, 1:]
    z = v - v.mean(); s2 = (z * z).sum()
    if s2 <= 0:
        return np.nan
    return (n / (n * k)) * ((z[:, None] * z[idx]).sum() / s2)


def hvg_seurat(mean_raw, var_raw, k, eligible, nbin=20):
    """scanpy flavor='seurat' ranking (verified in lymph_pass2)."""
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


def main():
    var50 = json.load(open(os.path.join(ROOT, "var_50genes.json")))["genes"]

    folds = {}
    for k in (0, 1):
        te = pd.read_csv(os.path.join(ROOT, "splits", "test_%d.csv" % k))
        tr = pd.read_csv(os.path.join(ROOT, "splits", "train_%d.csv" % k))
        folds[k] = dict(test=list(te.sample_id), train=list(tr.sample_id))

    D = {}
    struct = {
        "cohort": "COAD (HEST-bench), 4 Xenium samples",
        "meta_patient_column": META_PATIENT,
        "subseries": SUBSERIES,
        "true_patient_bestsupported": TRUE_PATIENT,
        "patient_mapping_argument": (
            "HESTData.py:1249 groups splits by (dataset_title, patient). COAD has 2 "
            "groups: preview dataset {TENX111} (patient empty) and CRC-study trio "
            "{TENX147,148,149} (patient='Patient 1'). K=2 folds with test sizes 3 and 1 "
            "match exactly. NOTE: the metadata 'Patient 1' label for the trio is a "
            "mislabel - subseries names them Sample P1/P2/P5 CRC, three distinct "
            "patients in the source study (biorxiv 2024.06.04.597233); stage comments "
            "II-A (149) vs IV-A (147) corroborate distinct donors. The mislabel is "
            "CONSERVATIVE for splitting: the trio is never split across train/test, so "
            "no same-patient train/test overlap is demonstrable under either mapping. "
            "Residual risk: TENX111 preview donor is unidentified; its 'stage 2A "
            "adenocarcinoma' matches TENX149's Stage II-A, so same-donor identity with "
            "the trio can be neither confirmed nor excluded from public metadata."),
        "splits": {str(k): v for k, v in folds.items()},
    }
    leak = []
    for k, v in folds.items():
        for mapname, pmap in (("meta", META_PATIENT), ("subseries", TRUE_PATIENT)):
            pt_te = {pmap[s] for s in v["test"]}
            pt_tr = {pmap[s] for s in v["train"]}
            leak.append(dict(fold=k, mapping=mapname,
                             test_patients=sorted(pt_te), train_patients=sorted(pt_tr),
                             overlap=sorted(pt_te & pt_tr)))
    struct["leakage_check"] = leak
    struct["leakage_verdict"] = (
        "NO demonstrable leakage under both mappings (overlap empty in all folds); "
        "official split is (dataset_title,patient)-group-level; no patient-level "
        "corrected rerun needed. TENX111 donor identity unresolvable (flagged).")

    for sid in SAMPLES:
        X, it, obs_idx, tc, xy, vn, pb = load(sid)
        lib_row = X.sum(1)
        pbset = set(pb)
        off = np.array([b in pbset for b in obs_idx])
        D[sid] = dict(X=X, it=it, xy=xy, vn=vn, off=off,
                      gidx={g: i for i, g in enumerate(vn)}, tc=tc)
        struct[sid] = dict(
            n_spots=int(X.shape[0]), n_genes=int(X.shape[1]),
            in_tissue_n=int((it == 1).sum()),
            in_tissue_frac=float((it == 1).mean()),
            is_integer=bool(np.all(np.abs(X - np.round(X)) < 1e-6)),
            xmax=float(X.max()),
            obs_total_counts_matches_rowsum=bool(np.allclose(tc, lib_row, rtol=1e-3)),
            n_patch_barcodes=int(len(pb)),
            patch_barcodes_subset_of_obs=bool(pbset <= set(obs_idx)),
            official_frac_of_obs=float(off.mean()),
            dup_genes=int(len(vn) - len(set(vn))),
            miss_var50=[g for g in var50 if g not in set(vn)],
        )
        print("done", sid, X.shape, "official", int(off.sum()),
              "int=", struct[sid]["is_integer"],
              "tc==rowsum:", struct[sid]["obs_total_counts_matches_rowsum"])

    trio_same_var = bool(
        np.array_equal(np.sort(D["TENX147"]["vn"]), np.sort(D["TENX148"]["vn"])) and
        np.array_equal(np.sort(D["TENX147"]["vn"]), np.sort(D["TENX149"]["vn"])))
    common = set(D[SAMPLES[0]]["vn"])
    for sid in SAMPLES[1:]:
        common &= set(D[sid]["vn"])
    COMMON = sorted(common)
    struct["trio_var_same_geneset"] = trio_same_var
    struct["n_genes_common_all4"] = len(COMMON)
    struct["lib_definition_note"] = (
        "Xenium: obs/total_counts equals the 541-gene panel row sum (verified per "
        "sample above); there is no whole-transcriptome library here, so lib = panel "
        "total. Deviation from the 'full transcriptome' lib convention is inherent to "
        "the technology and recorded, not fixable.")
    struct["all_spots_equals_in_tissue"] = True
    struct["spotsets_used"] = ["all", "official"]
    struct["panel_candidate_space"] = (
        "trainHVG panels restricted to the %d genes common to all 4 samples (both "
        "folds span both Xenium panels)" % len(COMMON))

    # fold-wise train-side panels, per spot set, candidates = common genes
    fold_panels = {}
    for k, v in folds.items():
        for ss in ("all", "official"):
            # pooled mean/var over train spots, aligned to COMMON gene order
            n = 0
            s1 = np.zeros(len(COMMON)); s2 = np.zeros(len(COMMON))
            det = np.zeros(len(COMMON))
            for s in v["train"]:
                m = np.ones(D[s]["X"].shape[0], bool) if ss == "all" else D[s]["off"]
                cols = np.array([D[s]["gidx"][g] for g in COMMON])
                Xa = D[s]["X"][m][:, cols]
                s1 += Xa.sum(0); s2 += (Xa * Xa).sum(0)
                det += (Xa > 0).sum(0)
                n += Xa.shape[0]
            mu = s1 / n
            var = (s2 - n * mu * mu) / (n - 1)
            el = det >= np.ceil(0.10 * n)
            kk = min(737, int(el.sum()))
            key = "fold%d_%s" % (k, ss)
            CN = np.array(COMMON)
            fold_panels[key] = dict(
                hvg50=list(CN[hvg_seurat(mu, var, 50, el)]),
                hvg737=list(CN[hvg_seurat(mu, var, kk, el)]),
                hvg737_k=kk, n_eligible=int(el.sum()), n_train_spots=n)
            ov = len(set(fold_panels[key]["hvg50"]) & set(var50))
            print("panel %s: eligible %d, hvg737_k %d, trainHVG50 vs var50 overlap %d/50"
                  % (key, el.sum(), kk, ov))

    # depth pretable
    her2 = json.load(open(os.path.join(OUT, "lymph_panels.json")))["her2st_fit"]["sd_loglib"]
    rows = []
    for s in SAMPLES:
        fold_of = [k for k, v in folds.items() if s in v["test"]][0]
        for ss in ("all", "official"):
            m = np.ones(D[s]["X"].shape[0], bool) if ss == "all" else D[s]["off"]
            X = D[s]["X"][m]; xy = D[s]["xy"][m]
            lib = D[s]["tc"][m]
            loglib = np.log(np.maximum(lib, 1.0))  # unified definition
            base = dict(sample=s, patient=TRUE_PATIENT[s], fold=fold_of, spot_set=ss,
                        n_spots=int(m.sum()),
                        lib_median=float(np.median(lib)),
                        lib_p10=float(np.percentile(lib, 10)),
                        lib_p90=float(np.percentile(lib, 90)),
                        sd_loglib_fullgenome=float(np.std(loglib)),
                        cv_lib=float(np.std(lib) / np.mean(lib)),
                        moran_I_loglib_k6=float(moran(loglib, xy, 6)),
                        zero_frac_fullpanel=float((X == 0).mean()),
                        genes_detected_median=float(np.median((X > 0).sum(1))))
            base["sd_x_moran"] = base["sd_loglib_fullgenome"] * base["moran_I_loglib_k6"]
            pmap = {"hest_var50": var50,
                    "trainHVG50": fold_panels["fold%d_%s" % (fold_of, ss)]["hvg50"],
                    "trainHVG737": fold_panels["fold%d_%s" % (fold_of, ss)]["hvg737"]}
            for pname, glist in pmap.items():
                gi = np.array([D[s]["gidx"][g] for g in glist if g in D[s]["gidx"]])
                P = X[:, gi]
                plib = P.sum(1)
                base["%s_zero_frac" % pname] = float((P == 0).mean())
                base["%s_le1_frac" % pname] = float((P <= 1).mean())
                base["%s_panel_lib_median" % pname] = float(np.median(plib))
                base["%s_panel_lib_zero_spots" % pname] = int((plib == 0).sum())
            base["pred_libnull_medr_her2fit"] = her2["intercept"] + her2["slope"] * base["sd_loglib_fullgenome"]
            base["extrapolated_beyond_her2_range"] = bool(
                base["sd_loglib_fullgenome"] > her2["range"][1])
            rows.append(base)
            print("row", s, ss, "sd", round(base["sd_loglib_fullgenome"], 3),
                  "moran", round(base["moran_I_loglib_k6"], 3))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "hb_coad_depth_pretable.csv"), index=False)

    # pre-registration (BEFORE any null-model output)
    prereg = dict(
        cohort="COAD (HEST-bench), 4 Xenium samples",
        utc_timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        local_time=time.strftime("%Y-%m-%d %H:%M:%S"),
        basis="HER2ST fit r = 0.0296 + 0.2874*sd(loglib); LYMPH falsified linear "
              "extrapolation at sd>1 (saturation ~0.4 expected there). "
              "Order prediction (by sd loglib) is the PRIMARY prediction. "
              "NOTE lib here = Xenium 541-gene panel total (no full transcriptome).",
        predictions={},
    )
    for ss in ("all", "official"):
        sub = df[df.spot_set == ss].set_index("sample")
        order = list(sub.sort_values("sd_loglib_fullgenome", ascending=False).index)
        pts = {s: dict(sd_loglib=float(sub.loc[s, "sd_loglib_fullgenome"]),
                       point_pred_med_r=float(sub.loc[s, "pred_libnull_medr_her2fit"]),
                       band="point +/- 0.10 (cross-cohort transfer; LYMPH residuals "
                            "reached ~0.1); if sd>1 expect saturation, cap ~0.4; "
                            "Xenium panel-lib may shift the line (SKCM Xenium showed "
                            "libnull well above HER2ST-fit line)",
                       extrapolated=bool(sub.loc[s, "extrapolated_beyond_her2_range"]))
               for s in SAMPLES}
        prereg["predictions"][ss] = dict(
            order_by_sd_desc=order,
            falsification_order="libnull med-r ranking (cp10k_panel caliber) must match "
                                "this order; any inversion between samples whose sd gap "
                                ">0.10 falsifies H3",
            point_predictions=pts)
    prereg["h2_prediction"] = ("zerobio sim >= real med-r in >=7/8 of (sample x spotset) "
                               "cells under cp10k_panel (hest_var50 panel), as in "
                               "HER2ST/LYMPH/READ; falsified if sim < real in >=3/8")
    prereg["official_metric_prediction"] = (
        "official caliber (log1p raw, var50, official patch-barcode spot set, "
        "fold-concat, per-gene pearson mean over genes then over 2 folds): libnull "
        "mean-r predicted in 0.15-0.35 (COAD leaderboard: max 0.3284 GenBio-PathFM, "
        "ResNet50 0.2500, min 0.2057 Kaiko ViT-S/16, 26 rows README.md:108-133). "
        "Prior Xenium cohort SKCM: null 0.44 below all models; prior LYMPH/PAAD null "
        "beat all. Point guess 0.25: null beats >= half of the 26 rows. "
        "Falsified if null mean-r < 0.10.")
    json.dump(prereg, open(os.path.join(OUT, "hb_coad_prereg.json"), "w"), indent=1)
    json.dump(dict(fold_panels=fold_panels, spotsets=["all", "official"],
                   common_genes=COMMON),
              open(os.path.join(OUT, "hb_coad_panels.json"), "w"), indent=1)
    json.dump(struct, open(os.path.join(OUT, "hb_coad_struct.json"), "w"), indent=1,
              default=str)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60)
    print(df[["sample", "patient", "fold", "spot_set", "n_spots", "lib_median",
              "sd_loglib_fullgenome", "moran_I_loglib_k6",
              "pred_libnull_medr_her2fit"]].to_string())
    print("PREREG timestamp:", prereg["utc_timestamp"])
    print("LEAKAGE:", json.dumps(leak))


if __name__ == "__main__":
    main()
