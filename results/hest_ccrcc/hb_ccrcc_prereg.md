# CCRCC zero-pixel audit — PRE-REGISTRATION (written BEFORE any null-model output)

Timestamp: 2026-08-24T03:29:06Z (UTC). Precedes hb_ccrcc_audit_*.csv (none exists yet).
Inputs fixed before this file: hb_ccrcc_depth_pretable.csv + hb_ccrcc_panels.json
(structural facts only; no H1/H2/H4/H5 numbers computed yet).

## Cohort + patient-mapping argument
HEST-bench CCRCC: 24 Visium samples INT1..INT24, dataset "Tertiary lymphoid structures
... renal cell cancer" (Meylan et al. 2022, PMID 35231421). Per
methods/gen4_hest/assets/HEST_v1_1_0.csv lines 381-404: INT24=Patient 1 ... INT1=Patient 24,
i.e. ONE SAMPLE PER PATIENT (24 distinct patients). Official splits: 6 folds x 4 test
samples, train = exact complement, partition verified. Since every sample is a distinct
patient, the official splits are patient-level by construction -> NO train/test patient
leakage. (create_splits K=6 != 24 patients triggered the "distribute patients" branch,
HESTData.py:1281-1294, grouping 4 consecutive patients per fold; groups never split a
patient.)

Known SELECTION leakage (cohort-level, not split-level): var_50genes.json was selected by
pooling ALL 24 samples including every test patient (get_k_genes, hest/utils.py:671-739).
Reproduced here byte-equivalently: 50/50 overlap with pooled-all-samples seurat-HVG on
2,248 eligible common genes (hb_ccrcc_panels.json: var50_repro_overlap).

## Structural facts fixed before prediction (hb_ccrcc_depth_pretable.csv)
- Batch structure: INT13-24 FFPE probe set (17,943 genes), INT1-12 frozen (36,601 genes);
  common pool = 17,943 (FFPE set is a subset of frozen set). Folds 2 and 4 mix protocols
  inside the test set; every training set mixes both. X integer CSR; lib = obs/total_counts
  == X.sum(1) exactly (full transcriptome of each sample's own probe set — DISCLOSED
  caliber mismatch: lib for frozen samples counts 36,601 genes, FFPE 17,943).
- Spot sets DEGENERATE: in_tissue == 1 for 100% of spots in all 24 samples, and the
  patches/*.h5 barcode set == all obs barcodes in all 24 samples. Therefore
  all_spots = in_tissue = official bench set; the audit runs ONE spot set ("all") and the
  planned two-spot-set contrast is void (disclosed here, not hidden).
- Depth structure sd(log max(lib,1)), per sample: range 0.382 (INT12) to 1.394 (INT10),
  median ~0.74. 23/24 samples fall INSIDE the HER2ST fitted range [0.389, 1.428]
  -> point predictions below are interpolations for once, not extrapolations.
- Moran's I of log-lib (k=6): 0.44-0.94, median ~0.77 — depth is strongly spatial.
- Official metric (benchmark.py:292,327-348): within a fold the 4 test samples are
  CONCATENATED, per-gene Pearson over concatenated spots, mean over 50 genes = fold score,
  mean over 6 folds = leaderboard number. Targets = log1p(RAW counts) of var50 on patch
  barcodes (st_dataset normalize=log1p only, no CP10K).
- Pooled (concatenated) test-side sd_loglib per fold — the depth lever the official
  metric actually sees: fold0 0.688, fold1 0.951, fold2 0.651, fold3 1.316, fold4 0.847,
  fold5 0.998. Fold 3 concatenates INT5 (mean loglib 10.51, lib_med 44k) with INT4
  (7.20, lib_med 1.3k) — a 33x median-depth gap INSIDE one test set; between-sample mean
  shifts will dominate its concatenated per-gene correlations.

## Pre-registered predictions (primary = orderings)
HER2ST linear fit r = 0.0296 + 0.2874*sd(loglib), fitted range [0.389, 1.428]; LYMPH
falsified linear extrapolation at sd>1 (saturation ~0.4 expected under cp10k); here only
INT10 (1.394) approaches the boundary; treat all point predictions as "about +-0.05".

P1 (PRIMARY, per-sample ordering): H1 libnull med_r follows sd(loglib) across the 24
   samples within each caliber x panel. Pre-registered test: Spearman(sd_loglib, med_r)
   over n=24 >= +0.5 under cp10k_panel/var50. FALSIFIED if <= 0. Strongest pairwise call:
   INT10 (sd 1.394) in the top-3 med_r; INT12 (sd 0.382) in the bottom-5.
P2: on the same sample, log1p_raw >= cp10k_panel med_r (depth dominates raw-log targets)
   in >= 20/24 samples.
P3: H2 zero-biology sim >= real med_r in the majority of fold x caliber conditions
   (HER2ST/LYMPH precedent), under both cp10k_panel and log1p_raw.
P4 (leaderboard confrontation): official-caliber H1 (log1p_raw, hest_var50, all spots
   = patch spots, train-side fit, FOLD-CONCATENATED per-gene mean r, then mean over 6
   folds). Point prediction from pooled sd via HER2ST fit: fold scores approx
   {f0 0.23, f1 0.30, f2 0.22, f3 0.41, f4 0.27, f5 0.32} -> 6-fold mean approx 0.29.
   Pre-registered band: [0.18, 0.42]. FALSIFICATION BAND: < 0.10 falsifies "depth
   artifact materially supports the CCRCC leaderboard column".
   Leaderboard CCRCC column (methods/gen4_hest/README.md:106-133, 26 model rows):
   max 0.2719 (Virchow2), min 0.0783 (ResNet50). Explicit confrontation prediction:
   the zero-pixel null lands ABOVE the column maximum (null >= 0.2719); weaker fallback:
   null beats >= 20/26 rows. Fold-level order prediction: f3 > f5 > f1 > f4 > f0 > f2
   (by pooled sd).
P5: H4 ceiling r_tech (var50, log1p_raw) HIGH (median lib 1.3k-44k, panel-heavy genes):
   per-sample median r_tech >= 0.8 for >= 20/24 samples; ceiling not the binding
   constraint. INT4 (lib_med 1336, 97% zero matrix) lowest ceiling of the cohort.
P6: H5 r_nbr(k=6) tracks Moran I of lib; ordering Spearman(rnbr_med, moran_I) > 0 across
   24 samples; INT18 (I=0.939) top-3, INT1 (I=0.443) bottom-3 under cp10k_panel/var50.

## Deviations from LYMPH protocol, declared now
- Spot set contrast void (see above): one spot set only.
- trainHVG panels selected on all spots per fold (= the only spot set), train samples
  only, seurat flavor, eligibility = detected in >=10% spots in EVERY train sample;
  trainHVG737 = top min(737, n_eligible); n_eligible per fold 2,248-4,519 (fold 3's
  4,519 comes from dropping the 4 shallow frozen test samples from train).
- lib for cp10k_full and for sd(loglib) = obs/total_counts of each sample's own probe
  set (FFPE 17,943 vs frozen 36,601) — cross-protocol lib is not on a common gene
  universe; disclosed, kept as-is because the official pipeline does the same.
- H1 fits/scoring: per-sample rows AND per-fold concatenated rows (official style);
  the leaderboard confrontation (P4) uses the concatenated version.

CORRECTION (2026-08-24T03:40Z, before any audit output; does not alter any prediction):
P4 misquoted the CCRCC column minimum as "0.0783 (ResNet50)" — 0.0783 is ResNet50's READ
value. Correct CCRCC column extremes: max 0.2719 (Virchow2), min 0.1825 (MUSK);
ResNet50 CCRCC = 0.2252. The "null >= column max 0.2719" and ">= 20/26 rows" predictions
stand unchanged.
