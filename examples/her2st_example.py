# -*- coding: utf-8 -*-
"""LOCAL EXAMPLE -- reads this machine's HER2ST prototype, not part of the package.

Author: Zigan Wang.

stnull itself is self-contained and imports nothing from D:\\project\\STImage.
This script only exists to prove the API on real data: it builds a small audit
input (3 sections x 100 genes) out of

    prototype/data/meta.parquet      per-spot table
    prototype/data/expr_<sec>.npz    raw counts (sparse)
    prototype/data/genes_<sec>.json  gene order of that matrix
    prototype/preds/ridge_fold_<P>.npz   frozen patient-level ridge predictions

and then calls stnull.audit().  Delete the PROTO path below and this file is
useless; nothing else in the package refers to it.

Run:  python examples/her2st_example.py

DATA LINEAGE (row numbers refer to docs/CODE_WALKTHROUGH.md (section 4, the variable lineage table))
    This whole file is flow [L17]: it manufactures, from local HER2ST
    files, exactly the objects that audit() expects, and hands them over
    with the same keywords the CLI maps in flow [L16].
    Inputs (local disk, outside the package)
        prototype/data/meta.parquet           per-spot table: section,
                                              patient, label, lib_size,
                                              arr_x/arr_y, row_idx
        prototype/data/expr_<sec>.npz         raw counts, sparse (spots x
                                              genes of that section)
        prototype/data/genes_<sec>.json       gene order of that matrix
        prototype/preds/ridge_fold_<P>.npz    frozen patient-held-out ridge
                                              predictions on a 737-gene
                                              panel (panel = the gene
                                              subset a model scores)
    Outputs (into audit(), matching the [L01]-[L14] coercion layer)
        Y      (n_all, 100)  log1p-CP10K truth on the toy panel -> y_true [L01]
        P      (n_all, 100)  re-projected ridge predictions      -> y_pred [L02]
        C      (n_all, 100)  raw integer counts                  -> counts [L08]
        obs columns: section [L03], lib_size [L04] and lib_size_full
        [L14], arr_x/arr_y coords [L06], label [L07], patient [L09]
        genes  100 names                                         -> genes [L13]
    dump_for_cli() writes the same objects to disk in the formats
    _load_matrix/_load_obs read ([L15]/[L16]), so the CLI can be exercised
    on identical data.
"""
import os

# WHAT: cap every BLAS backend at 3 threads BEFORE numpy is imported.
# WHY: on the 16-core development laptop, unbounded BLAS threading makes the
# small dense solves in the audit slower, not faster, and starves the fans.
# This affects speed only, never a number.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "3"

import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from stnull import audit

PROTO = Path(r"D:\project\STImage\prototype")      # <- local only
SECTIONS = ["A1", "B1", "C1"]                       # one per patient, all labelled
N_GENES = 100                                       # size of the toy gene panel
OUT = Path(__file__).resolve().parent / "out"       # examples/out/, git-ignored


def build():
    """Assemble (Y, P, C, obs, genes) for audit(), flow [L17].

    Y and P end up in the same declared space, log1p-CP10K on the shared
    100-gene panel (CP10K = counts scaled so each spot's panel sums to
    10000, then log1p), because audit() correlates them directly and a
    space mismatch would be a spurious depth signal.
    """
    # --- 1. per-spot metadata and the frozen predictions ----------------
    # WHAT: load the ridge predictions of the fold that held out each
    # section's patient, so P is honestly out-of-sample.
    # WHERE: meta becomes the obs table handed to audit() in main();
    # preds[s] feeds P [L02]; rows[s] aligns prediction rows to meta rows.
    meta = pd.read_parquet(PROTO / "data" / "meta.parquet")   # one row per spot
    panels, preds, rows = {}, {}, {}
    for s in SECTIONS:
        p = s[0]                     # patient id = first letter of the section
        z = np.load(PROTO / "preds" / ("ridge_fold_%s.npz" % p), allow_pickle=False)
        take = z["sections"] == s    # (n_fold_spots,) bool: this section only
        panels[s] = list(z["genes"])                 # that fold's 737-gene panel
        preds[s] = z["pred_panel737"][take].astype(np.float64)  # (n, 737) preds
        rows[s] = z["rows"][take]    # (n,) row_idx keys back into meta
    # WHAT: the folds were trained separately, so their gene panels differ;
    # keep only genes present in all three.
    # WHY sorted(): a deterministic gene order, so reruns are identical.
    common = sorted(set(panels[SECTIONS[0]]).intersection(*[set(panels[s])
                                                            for s in SECTIONS[1:]]))
    print("[example] genes shared by the 3 fold panels: %d" % len(common))

    # --- 2. pick the 100 most-detected shared genes ---------------------
    # rank the shared genes by detection rate so the toy panel is not degenerate
    # WHY: a gene observed in almost no spot has an ill-defined Pearson r
    # (Pearson r = the standard linear correlation, 1 = perfect match);
    # ranking by detection rate keeps the toy panel scorable.  Detection
    # uses raw counts only, never the predictions, so no selection bias
    # in favour of the model is introduced.
    det = np.zeros(len(common))      # summed per-gene detection rate, 3 sections
    counts_by_sec = {}
    for s in SECTIONS:
        # WHAT: load this section's raw sparse counts and slice it to the
        # spots that have predictions and to the shared genes.
        # WHERE: counts_by_sec[s] later becomes both Y (after CP10K) and
        # the counts input C [L08] of ceiling.r_tech.
        gs = json.loads((PROTO / "data" / ("genes_%s.json" % s)).read_text())
        col = {g: i for i, g in enumerate(gs)}       # gene name -> column index
        M = sp.load_npz(PROTO / "data" / ("expr_%s.npz" % s)).tocsr()
        sel = meta.index[meta.section == s]
        pos = meta.loc[sel, "row_idx"].values        # meta row keys, this section
        keep = np.isin(pos, rows[s])                 # spots the ridge fold scored
        C = np.asarray(M[keep][:, [col[g] for g in common]].todense(), np.float64)
        counts_by_sec[s] = C                          # (n_s, len(common)) counts
        det += (C > 0).mean(0)                        # fraction of spots with >0
    order = np.argsort(-det)[:N_GENES]               # indices of the top 100
    # WHY sorted(order): keep the genes in their original panel order, so
    # the panel is a stable subset, not a detection-ranked reordering.
    genes = [common[i] for i in sorted(order)]       # [L13] the 100 gene names
    gpos = {g: i for i, g in enumerate(common)}
    gi = [gpos[g] for g in genes]                    # columns of the toy panel

    # --- 3. build Y, P, C section by section ----------------------------
    Y, P, C_all, keep_rows = [], [], [], []
    for s in SECTIONS:
        # WHAT: truth Y = log1p CP10K of the raw counts on the toy panel.
        # WHY the max(.,1) denominator: a spot with zero panel counts must
        # not divide by zero; its row stays all zero.
        C = counts_by_sec[s][:, gi]                  # (n_s, 100) raw counts
        den = np.maximum(C.sum(1, keepdims=True), 1.0)   # per-spot panel total
        Y.append(np.log1p(1e4 * C / den))            # -> y_true [L01]
        # renormalise the frozen 737-panel prediction onto the same 100-gene panel
        # WHY: the ridge predicted log1p-CP10K over 737 genes; the 100-gene
        # slice of that is in no declared space.  Undo the transform
        # (expm1, /1e4), slice, clip negatives, renormalise over the 100
        # genes and re-apply log1p, so P sits in exactly Y's space.
        prop = np.expm1(preds[s]) / 1e4              # back to proportions
        j = [panels[s].index(g) for g in genes]      # panel columns of our genes
        sub = np.clip(prop[:, j], 0, None)           # (n_s, 100) proportions
        P.append(np.log1p(1e4 * sub / np.maximum(sub.sum(1, keepdims=True), 1e-9)))
        C_all.append(C)
        keep_rows.append(rows[s])
    # WHAT: rebuild obs in exactly the row order of the stacked matrices.
    # WHY: audit() aligns everything by row position; meta's own order
    # differs from the prediction files' order, so reindex by row_idx.
    obs = meta.set_index("row_idx").loc[np.concatenate(keep_rows)].reset_index()
    # WHERE: the five returns map onto audit() inputs in main():
    # Y -> y_true [L01], P -> y_pred [L02], C -> counts [L08] (int64:
    # ceiling.r_tech requires integer UMI counts), obs -> the per-spot
    # keyword columns [L03]/[L04]/[L06]/[L07]/[L09], genes -> genes [L13].
    return (np.vstack(Y), np.vstack(P), np.vstack(C_all).astype(np.int64), obs,
            genes)


def dump_for_cli():
    """Write the same example as plain files, so the CLI can be exercised.

    WHAT: serialise build()'s five objects in the formats cli._load_matrix
    and cli._load_obs read (flows [L15]/[L16]), then print the exact CLI
    command that reproduces main()'s audit from those files.
    WHY: this is the proof that the CLI contract is a pure rename: the same
    data through either entry point must produce the same report.
    """
    Y, P, C, obs, genes = build()
    d = OUT / "cli_inputs"
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "Y.npy", Y)
    np.save(d / "P.npy", P)
    np.savez_compressed(d / "counts.npz", counts=C)
    obs.to_parquet(d / "obs.parquet")
    (d / "genes.txt").write_text("\n".join(genes), encoding="utf-8")
    print("[example] CLI inputs in %s" % d)
    print("  python -m stnull.cli audit --true %s --pred %s --obs %s \\\n"
          "      --space log1p_cp10k:panel --section-col section --lib-col lib_size \\\n"
          "      --coord-cols arr_x,arr_y --label-col label --patient-col patient \\\n"
          "      --counts %s --genes %s --top-n 10,50 --perm 100 --boot 100 \\\n"
          "      --out audit.html --csv-dir out_csv"
          % (d / "Y.npy", d / "P.npy", d / "obs.parquet", d / "counts.npz",
             d / "genes.txt"))


def main():
    """Run the audit on the toy build and write every output format."""
    Y, P, C, obs, genes = build()
    print("[example] Y=%s  P=%s  counts=%s  sections=%s"
          % (Y.shape, P.shape, C.shape, sorted(obs.section.unique())))
    # WHAT: the audit() call, using the same keywords the CLI maps in
    # flow [L16]; each argument comment names the coercion-layer row it
    # lands on inside audit.py.
    # WHY space= is spelled out: audit() has no default target space; the
    # ":panel" suffix plus space_note declare that the CP10K denominator
    # is the 100-gene panel row sum, not the full transcriptome.
    rep = audit(
        y_true=Y, y_pred=P,                              # [L01] / [L02]
        space="log1p_cp10k:panel",                       # [L12] declared space
        space_note="denominator = row sum of the 100-gene example panel",
        section=obs["section"].values,                   # [L03] slice labels
        genes=genes,                                     # [L13] names, cosmetic
        lib_size=obs["lib_size"].values,                 # full transcriptome
        # ^ [L04] library size = total UMI per spot (sequencing depth)
        coords=obs[["arr_x", "arr_y"]].values,           # [L06] grid positions
        coord_kind="grid",
        labels=obs["label"].fillna("unlabelled").values,  # [L07] tissue class
        patient=obs["patient"].values,                   # [L09] patient ids
        counts=C,                                        # [L08] raw counts
        lib_size_full=obs["lib_size"].values,            # [L14] ceiling depth
        top_n=(10, 50),                                  # selection lever sizes
        n_perm=200, n_boot=200, n_rep_ceiling=10,
        seed=0, verbose=True)

    # WHAT: exercise every renderer of report.py on the same AuditReport.
    # WHERE: summary/to_html/to_markdown/to_json/to_csv_dir are the
    # renderer methods defined in report.py; the sample_report.html copy
    # is the file the repository README links to.
    print()
    print(rep.summary())
    OUT.mkdir(exist_ok=True)
    rep.to_html(OUT / "her2st_audit.html")
    # the copy the README links to as the canonical sample report
    rep.to_html(OUT.parent / "sample_report.html")
    rep.to_markdown(OUT / "her2st_audit.md")
    rep.to_json(OUT / "her2st_audit.json")
    rep.to_csv_dir(OUT / "csv")
    print("\n[example] wrote %s" % OUT)


if __name__ == "__main__":
    import sys
    if "--dump" in sys.argv:
        dump_for_cli()
    else:
        main()
