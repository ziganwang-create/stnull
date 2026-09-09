# -*- coding: utf-8 -*-
"""
Compositional-null pilot on HER2ST (the control STimage never ran).  v2 (post-referee).

Question: how much of "H&E -> spatial gene expression" performance on the field's
standard benchmark is reproduced by a predictor that knows ONLY the pathologist's
region label of each spot (no image model at all)?

Null predictor (leave-one-patient-out, LOPO):
  For held-out annotated section S (one per patient: A1,B1,C1,D1,E1,F1,G2,H1),
  predict each labeled spot's expression of gene g as the mean log1p-CP10K of g
  over spots with the SAME region label in the other 7 patients' annotated
  sections. Fallback: training grand mean of g. Note this instantiates
  DOMINANT-REGION IDENTITY (one-hot), a strictly coarser object than fractional
  cell-type composition — i.e., a deliberately weak floor for "composition".

Also computed:
  - oracle same-section null (region means from the held-out section itself);
  - coarse binary null (tumor = invasive + in situ vs everything else);
  - per-section label entropy (moderator analysis);
  - winner's-curse experiment on the null's own predictions:
      * top100_winnerscurse: select the 100 genes with highest null r ON THE TEST
        SECTION ITSELF, report their median r (the field's "top-N most predictable
        genes" reporting style);
      * top100_honest: rank genes by their MEAN null r over the OTHER 7 sections,
        take the top-100 names, evaluate their median r on the held-out section
        (fewer than 100 may exist in a fold's panel; n reported). Caveat: the
        ranking r's come from folds in which the held-out patient contributed to
        training means (second-order optimism; noted in the paper);
      * top100_shuffle_floor: labels randomly permuted within each training section
        (B=20 draws), null recomputed, top-100 selected on the test section —
        the pure noise-ranking inflation floor for the winner's-curse statistic.

Metric: per-gene Pearson r across labeled spots (log1p-CP10K), median over the
top-737 HVG panel (HVGs selected on the 7 training sections only, mirroring the
paper's top-1000 -> ~737 protocol). Canonical genes are evaluated on the full
common gene set (12,689), independent of HVG panel membership.

Reference values read off STimage's own figures for the same sections
(approximate, +/-0.02-0.05): Fig 2e slide-LOOCV (patient-leaky: serial sections
of the test patient are in training) STimage median PCC — A1 ~0.16-0.22,
B1 ~0.55-0.65, C1 ~0.38-0.54, D1 ~0.27-0.42, E1 ~0.12-0.16, F1 ~0.13-0.19,
G2 ~0.34-0.42, H1 ~0.21-0.23 (H read from Supp S8d).  Supp S8d (Visium-trained,
cross-platform + cross-patient, 1,146 predictability-selected genes):
B1 ~0.51-0.57, C1/D1 ~0.23-0.31, A1 ~0.03-0.05, E1/F1 ~0.00, G2 ~0.20-0.22, H1 ~0.21-0.23.

Outputs (6 files): null_summary_by_section.csv, null_pergene_r.csv,
null_canonical_genes.csv, null_winnerscurse.csv, null_entropy.csv, null_summary.json
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(0)
ROOT = Path(r"D:\project\STImage\Xiao_data_codes\data\her2st\data")
OUT = Path(r"D:\project\STImage\pilot")
OUT.mkdir(parents=True, exist_ok=True)

SECTIONS = ["A1", "B1", "C1", "D1", "E1", "F1", "G2", "H1"]
TUMOR_LABELS = {"invasive cancer", "cancer in situ"}
N_HVG = 737
N_TOP = 100
N_SHUFFLE = 20

def load_section(sec):
    cnt = pd.read_csv(ROOT / "ST-cnts" / f"{sec}.tsv.gz", sep="\t", index_col=0)
    lbl = pd.read_csv(ROOT / "ST-pat" / "lbl" / f"{sec}_labeled_coordinates.tsv", sep="\t")
    lbl = lbl.dropna(subset=["x", "y", "label"])
    # guard the coordinate-rounding match (referee request): coordinates must be
    # unambiguous and spot ids unique
    fx = (lbl["x"] - lbl["x"].round()).abs().max()
    fy = (lbl["y"] - lbl["y"].round()).abs().max()
    assert fx < 0.45 and fy < 0.45, f"{sec}: ambiguous coordinate rounding ({fx:.2f},{fy:.2f})"
    lbl["spot"] = lbl["x"].round().astype(int).astype(str) + "x" + lbl["y"].round().astype(int).astype(str)
    lbl = lbl[lbl["label"] != "undetermined"]
    assert not lbl["spot"].duplicated().any() or \
        (lbl.groupby("spot")["label"].nunique().max() == 1), f"{sec}: conflicting duplicate spot labels"
    lbl = lbl.drop_duplicates(subset="spot", keep="first").set_index("spot")
    common = cnt.index.intersection(lbl.index)
    cnt = cnt.loc[common]
    lab = lbl.loc[common, "label"]
    libsize = cnt.sum(axis=1)
    keep = libsize > 100
    cnt, lab, libsize = cnt[keep], lab[keep], libsize[keep]
    expr = np.log1p(cnt.div(libsize, axis=0) * 1e4)
    return expr, lab

print("Loading 8 annotated sections ...")
data = {s: load_section(s) for s in SECTIONS}
ent_rows = []
for s, (e, l) in data.items():
    p = l.value_counts(normalize=True)
    ent = float(-(p * np.log(p)).sum())
    ent_rows.append({"section": s, "n_spots": e.shape[0], "n_labels": len(p),
                     "label_entropy_nats": round(ent, 3),
                     "dominant_label": p.index[0], "dominant_frac": round(float(p.iloc[0]), 3)})
    print(f"  {s}: {e.shape[0]} spots | entropy {ent:.2f} | {dict(l.value_counts())}")
entropy = pd.DataFrame(ent_rows).set_index("section")
entropy.to_csv(OUT / "null_entropy.csv")

genes = None
for e, _ in data.values():
    genes = e.columns if genes is None else genes.intersection(e.columns)
genes = pd.Index(sorted(genes))
print(f"Common genes across sections: {len(genes)}")

def hvg_panel(train_secs):
    big = pd.concat([data[s][0][genes] for s in train_secs], axis=0)
    ok = (big > 0).mean(axis=0) >= 0.10
    var = big.loc[:, ok].var(axis=0)
    return var.sort_values(ascending=False).head(N_HVG).index

def region_means(train_data, panel, coarse=False):
    frames, labels = [], []
    for e, l in train_data:
        frames.append(e[panel])
        labels.append(l.map(lambda x: "tumor" if x in TUMOR_LABELS else "non-tumor") if coarse else l)
    big = pd.concat(frames, axis=0)
    lab = pd.concat(labels, axis=0)
    return big.groupby(lab.values).mean(), big.mean(axis=0)

def pergene_r(y, pred):
    yv, pv = y.values, pred.values.astype(float)
    yc = yv - yv.mean(0); pc = pv - pv.mean(0)
    denom = np.sqrt((yc**2).sum(0) * (pc**2).sum(0))
    with np.errstate(invalid="ignore", divide="ignore"):
        return pd.Series((yc * pc).sum(0) / denom, index=y.columns)

def evaluate(sec, panel, means, gm, coarse=False):
    e, l = data[sec]
    y = e[panel]
    lab = l.map(lambda x: "tumor" if x in TUMOR_LABELS else "non-tumor") if coarse else l
    pred = pd.DataFrame(index=y.index, columns=panel, dtype=float)
    for r in lab.unique():
        rows = lab[lab == r].index
        pred.loc[rows] = (means.loc[r] if r in means.index else gm).values
    return pergene_r(y, pred)

rows, pergene_store, wc_rows = [], {}, []
for held in SECTIONS:
    train = [s for s in SECTIONS if s != held]
    panel = hvg_panel(train)
    train_data = [data[s] for s in train]
    m6, gm = region_means(train_data, panel)
    r_cross = evaluate(held, panel, m6, gm)
    m2, gm2 = region_means(train_data, panel, coarse=True)
    r_bin = evaluate(held, panel, m2, gm2, coarse=True)
    m_or, gm_or = region_means([data[held]], panel)
    r_oracle = evaluate(held, panel, m_or, gm_or)

    rows.append({"section": held, "n_spots": data[held][0].shape[0],
                 "median_r_crosspatient_6cat": np.nanmedian(r_cross),
                 "median_r_binary_tumor": np.nanmedian(r_bin),
                 "median_r_oracle_samesection": np.nanmedian(r_oracle),
                 "q90_r_crosspatient": np.nanquantile(r_cross, 0.90),
                 "frac_genes_r_gt_0.3": float((r_cross > 0.3).mean())})
    pergene_store[held] = pd.DataFrame({"cross6": r_cross, "binary": r_bin, "oracle": r_oracle})

    # ---- winner's-curse experiment (all on the cross-patient null) ----
    r = r_cross.dropna()
    wc = r.sort_values(ascending=False).head(N_TOP).median()
    # honest pre-selection: rank by mean r over the other 7 sections' folds
    # (computed after main loop; placeholder here, filled below)
    # shuffle floor: permute labels within each training section, B draws
    floor_meds = []
    for b in range(N_SHUFFLE):
        shuf = []
        for e_t, l_t in train_data:
            shuf.append((e_t, pd.Series(RNG.permutation(l_t.values), index=l_t.index)))
        m_s, gm_s = region_means(shuf, panel)
        e_h, l_h = data[held]
        pred = pd.DataFrame(index=e_h.index, columns=panel, dtype=float)
        for lab_r in l_h.unique():
            rws = l_h[l_h == lab_r].index
            pred.loc[rws] = (m_s.loc[lab_r] if lab_r in m_s.index else gm_s).values
        r_s = pergene_r(e_h[panel], pred).dropna()
        floor_meds.append(r_s.sort_values(ascending=False).head(N_TOP).median())
    wc_rows.append({"section": held, "median_all_HVG": r.median(),
                    "top100_winnerscurse": wc,
                    "top100_shuffle_floor_mean": float(np.mean(floor_meds)),
                    "top100_shuffle_floor_max": float(np.max(floor_meds))})
    print(f"{held}: cross6 {r.median():.3f} | binary {np.nanmedian(r_bin):.3f} | "
          f"oracle {np.nanmedian(r_oracle):.3f} | wc-top100 {wc:.3f} | floor {np.mean(floor_meds):.3f}")

# honest pre-selection pass (needs all folds' per-gene r)
allpg = pd.concat(pergene_store, names=["section", "gene"])
wc_df = pd.DataFrame(wc_rows).set_index("section")
hon_med, hon_n = [], []
for held in SECTIONS:
    other = allpg.drop(index=held, level=0)["cross6"].groupby(level=1).mean().dropna()
    top_names = other.sort_values(ascending=False).head(N_TOP).index
    r_here = pergene_store[held]["cross6"].reindex(top_names).dropna()
    hon_med.append(r_here.median()); hon_n.append(len(r_here))
wc_df["top100_honest_preselected"] = hon_med
wc_df["n_genes_honest"] = hon_n
wc_df.round(4).to_csv(OUT / "null_winnerscurse.csv")

res = pd.DataFrame(rows).set_index("section")
res.round(4).to_csv(OUT / "null_summary_by_section.csv")
allpg.to_csv(OUT / "null_pergene_r.csv")

# ---- canonical genes: evaluated on the FULL common gene set, not the HVG panel ----
CANON = ["COX6C", "GNAS", "FASN", "ESR1", "ERBB2", "B2M", "CD74", "SPARC", "KRT5",
         "VEGFA", "HSP90AB1", "TFF3", "ATP1A1", "PABPC1", "CD63", "CD81", "TP53", "GATA3", "SCD", "MUC1"]
canon_present = [g for g in CANON if g in genes]
canon_missing = sorted(set(CANON) - set(canon_present))
canon_rows = []
for held in SECTIONS:
    train_data = [data[s] for s in [t for t in SECTIONS if t != held]]
    mC, gmC = region_means(train_data, pd.Index(canon_present))
    rC = evaluate(held, pd.Index(canon_present), mC, gmC)
    for g in canon_present:
        canon_rows.append({"section": held, "gene": g, "r_null": rC[g]})
canon = pd.DataFrame(canon_rows)
canon_med = canon.groupby("gene")["r_null"].median().sort_values(ascending=False)
canon_med.round(4).to_csv(OUT / "null_canonical_genes.csv")
print("\nCanonical genes (FULL 20-gene panel, evaluated on common gene set):")
print(canon_med.round(3).to_string())
print("missing from common gene set:", canon_missing if canon_missing else "none")

summary = {
    "median_over_sections_crosspatient": float(res["median_r_crosspatient_6cat"].median()),
    "median_over_sections_binary": float(res["median_r_binary_tumor"].median()),
    "median_over_sections_oracle": float(res["median_r_oracle_samesection"].median()),
    "winnerscurse_mean_over_sections": float(wc_df["top100_winnerscurse"].mean()),
    "shuffle_floor_mean_over_sections": float(wc_df["top100_shuffle_floor_mean"].mean()),
    "honest_top100_mean_over_sections": float(wc_df["top100_honest_preselected"].mean()),
    "canonical_missing_from_common_geneset": canon_missing,
    "per_section": res.round(4).to_dict(orient="index"),
    "winnerscurse": wc_df.round(4).to_dict(orient="index"),
    "entropy": entropy.to_dict(orient="index"),
}
(OUT / "null_summary.json").write_text(json.dumps(summary, indent=2))
print("\nSaved 6 outputs.")
