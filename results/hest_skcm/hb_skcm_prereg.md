# SKCM zero-pixel audit — PRE-REGISTRATION (written BEFORE any null-model output)

Cohort: HEST-bench SKCM, 2 Xenium samples (TENX115 = "patient 2", TENX117 = "patient 1",
per methods/gen4_hest/assets/HEST_v1_1_0.csv lines 147/149). Splits = 2-fold leave-one-out,
each fold trains on one patient's sample and tests on the other -> patient-level split,
NO leakage (mapping argued from HEST metadata patient column + splits fold count 2 with
test size 1; one sample per patient).

## Structural facts fixed before prediction (hb_skcm_depth_pretable.csv + addendum)
- X dense integer counts, 541 targeted genes per sample, but DIFFERENT panels
  (Xenium skin panel vs skin panel + custom add-on): only 343 genes shared.
  Official var_50genes.json: all 50 present in BOTH samples.
- in_tissue == True for ALL spots in both adatas -> "in_tissue" spot set DEGENERATES to
  all_spots (in_tissue fraction = 100%). Second spot set therefore = OFFICIAL PATCH SUBSET
  (patches/*.h5 barcodes; the set the leaderboard actually scores):
  TENX115 1741/3886 spots, TENX117 1293/1830 spots.
- Depth structure sd(log max(lib,1)), lib = obs/total_counts (541-gene panel total):
  all_spots:  TENX115 4.216 (1478/3886 zero-lib bins), TENX117 4.116 (495/1830 zero-lib)
  official:   TENX115 1.642 (7 zero-lib),              TENX117 3.566 (198/1293 zero-lib!)
- Moran's I of log lib (k=6, all_spots): 0.969 / 0.933 — depth is almost purely spatial.
- Official caliber verified in source (gen4_hest/src/hest/bench/st_dataset.py:55-89 +
  benchmark.py:96): targets = log1p(RAW counts) of the 50-gene panel on patch barcodes;
  "normalize_adata" does NOT CP10K, only log1p. Metric = per-gene Pearson mean, then
  mean over the 2 folds.

## Pre-registered predictions (primary = orderings)
HER2ST linear fit r = 0.0296 + 0.2874*sd(loglib), fitted range sd in [0.389, 1.428].
LYMPH already FALSIFIED linear extrapolation at sd > 1 (expect saturation ~0.4+ under
cp10k; log1p_raw can go far higher). All four SKCM sd values are ABOVE the fitted range
-> point predictions below are extrapolations flagged as such; orderings are the real test.

Point predictions (linear fit, flagged extrapolated):
  all TENX115: 0.0296+0.2874*4.216 = 1.24  -> impossible; predict saturation, r_med >= 0.5
  all TENX117: 0.0296+0.2874*4.116 = 1.21  -> impossible; predict saturation, r_med >= 0.5
  off TENX115: 0.0296+0.2874*1.642 = 0.50  (mild extrapolation)
  off TENX117: 0.0296+0.2874*3.566 = 1.05  -> impossible; predict saturation, r_med >= 0.5

P1 (ordering, primary): H1 libnull r follows sd(loglib) within spot set x caliber:
   - official subset: TENX117 > TENX115 (STRONG prediction, delta sd = 1.92)
   - all_spots: TENX115 > TENX117 (WEAK, delta sd = 0.10; treat as coin-flip band)
P2: on same sample & spot set, log1p_raw >= cp10k_panel (depth dominates raw-log targets).
P3: H2 zero-biology sim >= real in the MAJORITY of conditions (per HER2ST + LYMPH precedent),
   at least under cp10k_panel; under log1p_raw also expected (global profile + real depth
   is nearly the whole signal).
P4 (leaderboard confrontation): official-caliber H1 (log1p_raw, hest_var50, official patch
   subset, train-side fit, mean-r per fold then averaged over 2 folds) lands in [0.30, 0.70].
   FALSIFICATION BAND: < 0.20 falsifies "depth artifact dominates SKCM leaderboard".
   Leaderboard column SKCM: max 0.6715 (GenBio-PathFM), min 0.4821 (ResNet50), 26 rows.
   Secondary prediction: libnull beats at least the ResNet50 row (0.4821) is PLAUSIBLE but
   not required; required is >= 0.30.
P5: H4 ceiling r_tech on official subset will be HIGH (median lib ~14-16k) -> ceiling is
   not the binding constraint there; on all_spots ceiling lower due to 38%/27% empty bins.
P6: H5 r_nbr(k=6) will be large (Moran I ~0.93-0.97 for lib), especially all_spots.

Deviations from LYMPH protocol, declared now:
 - "in_tissue" spot set replaced by "official" (patch-barcode) subset; in_tissue is
   degenerate (100% True).
 - trainHVG737 capped by eligibility: only 541 genes exist per sample; panel = top
   min(737, n_eligible) train-side HVGs -> effectively "all eligible train genes";
   genes absent from the test sample's var are dropped AT SCORING and the scored count
   is reported (train/test gene spaces differ; this mismatch is disclosed, not hidden).
 - Train side per fold = ONE sample (2-fold LOO), so "train-side fit" = fit on the single
   training sample.

Timestamp: see file mtime + line below (written before hb_skcm_libnull.csv exists).
Pre-registered: 2026-08-24 (UTC+8), before any H1/H2/H4/H5 computation on SKCM.
