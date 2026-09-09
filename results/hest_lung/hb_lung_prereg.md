# hb_lung PRE-REGISTRATION  (timestamp 2026-08-24 02:21:31)

Written BEFORE any H1/H2/H4/H5 output for LUNG. Cohort facts feeding it:
depth pre-table only (hb_lung_depth_pretable.csv, same commit).

## Cohort structure (established, not predicted)
- 2 samples, 2 patients (HEST_v1_1_0.csv: TENX141=patient 1, TENX118=patient 2),
  K=2 leave-one-patient-out splits -> NO split leakage.
- Both samples are Xenium (targeted panels, FFPE, LUAD). Panels differ:
  541 genes each, 259 shared; official var50 lies in the intersection.
- in_tissue == 100%% in both -> all_spots and in_tissue are THE SAME set;
  the two-spot-set protocol collapses to one (reported once, labeled).
- lib = obs/total_counts = 541-gene panel sum (no full transcriptome exists;
  'cp10k_full' here means full-sample-panel normalization, disclosed).

## Depth summary
- sd(log lib) [log(max(lib,1))]: TENX118 = 2.4639, TENX141 = 2.4025

## Predictions (H3 ordering is the primary prediction)
P1 ORDER: libnull median r (per test sample, official-analogue train-side fit)
   ranks by sd(log lib): TENX118 > TENX141.
P2 POINT (HER2ST linear fit r = 0.0296 + 0.2874*sd; LYMPH falsified the
   linear extrapolation at sd>1, expect saturation ~0.4 there):
   TENX118 pred 0.738, TENX141 pred 0.720 (band +/-0.10; if sd>1 treat
   min(pred, ~0.4) as the point).
P3 H2 zerobio: sim >= real median r in >=3 of 4 (sample x caliber) cells for
   hest_var50; i.e. the libnull signal is reproducible with ZERO biology
   (global train profile + real depths).
P4 Calibers: log1p_raw >= cp10k_panel on the same spot set (depth artifact
   dominates); Xenium counts are deep (lib median 2868/7549 on 541 genes),
   so cp10k gap may be smaller than Visium cohorts.
P5 H4 ceiling: deep targeted counts -> rtech median HIGH (>0.8) for
   hest_var50 under log1p_raw in both samples.
P6 Official-caliber zero-model mean-r vs leaderboard: LUNG leaderboard is
   high (max 0.5787, min 0.4949, 26 rows). Given only 2 samples and deep
   Xenium counts, PREDICT libnull mean-r (log1p_raw, var50) lands BELOW the
   leaderboard minimum (models beat depth-only null here), in 0.15-0.45.
   FALSIFICATION BAND: if zero-model mean-r >= 0.4949 the leaderboard is
   depth-dominated (LYMPH-style); if <= 0.15 depth explains little.

Primary registered readout: P1 ordering + P3 sim>=real + P6 band position.
