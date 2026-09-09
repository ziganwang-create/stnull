# hb_idc pre-registration — IDC (HEST-bench, 4x Xenium breast IDC)

Timestamp (UTC): 2026-08-23T16:23:10Z (written BEFORE any H1/H2/H4/H5 null-model output
for this cohort; only structure/depth pre-table `hb_idc_depth_pretable.csv` and panels
`hb_idc_panels.json` were computed first, per protocol).

## Cohort structure (facts, from inspect)
- Samples/folds (official splits, K=4, leave-one-out): fold0 test=TENX99, fold1 TENX95,
  fold2 NCBI785, fold3 NCBI783.
- Patient mapping (HEST_v1_1_0.csv + HESTData.py `create_benchmark_data` grouping by
  (dataset_title, patient)): TENX99=patient 1 (10x FFPE "Entire Sample Area"),
  TENX95=patient 2 (10x FFPE "Pre-designed Panel"), NCBI785=patient 4, NCBI783=patient 5
  (Janesick TME study; distinct receptor status ER+/HER2+ vs ER-/HER2+). 4 samples, 4
  distinct patient labels -> official splits do NOT cut a patient. Caveat: 10x demo
  datasets do not disclose donor identity; HEST's patient1-vs-patient2 distinction for
  TENX99/TENX95 is not independently verifiable offline. Adopted as-is; no leakage
  correction needed under the official mapping.
- Technology: Xenium binned to pseudo-visium; X dense uint16 raw counts; 541 probes per
  sample incl. 261/261/253/220 BLANK_/NegControl controls; common real genes = 280.
- in_tissue == True everywhere -> degenerate. Official eval spot set = patches h5
  barcodes ("bench_patches"): 20761/25080, 7760/11845, 4010/4180, 3005/3869.
  Spot-set tiers used here: all_spots / bench_patches (replaces in_tissue; disclosed).
- Zero-library spots (all_spots): TENX95 21.7%, TENX99 8.1%, NCBI785 0.1%, NCBI783 0%.
  Bench masking removes essentially all (max 1 spot remains).
- lib = obs/total_counts == X.sum(1) (panel total incl. controls; Xenium has no fuller
  transcriptome). sd(log lib) uses log(max(lib,1)).
- Official var50 reproduction: pooled ALL 4 samples (test included) -> 49/50 overlap.
  Confirms HEST panel-selection leakage is inherited by this task. Per-fold trainHVG50
  overlaps official var50 only 34-40/50.

## Depth pre-table (sd_loglib; from hb_idc_depth_pretable.csv)
all_spots:      TENX95 3.907 > TENX99 2.650 > NCBI783 2.051 > NCBI785 1.066
bench_patches:  NCBI783 1.377 > TENX99 1.072 > TENX95 0.841 ~= NCBI785 0.828
Moran's I(log lib, k=6): 0.60-0.96 everywhere (very strong spatial depth structure).

## Pre-registered predictions
P1 (PRIMARY, ordering): H1 library-null median r follows sd_loglib within each spot set
   x caliber. all_spots: TENX95 > TENX99 > NCBI783 > NCBI785.
   bench_patches: NCBI783 > TENX99 > {TENX95, NCBI785} — the last two differ by
   sd 0.013, declared a TIE (no prediction between them).
   Falsification band: any inversion between non-adjacent samples, or Spearman(sd,
   med r) < 0.8 on all_spots, falsifies the depth-structure account for this cohort.
P2 (point, secondary): HER2ST linear fit r = 0.0296 + 0.2874*sd (fit range sd
   0.389-1.428; corr 0.964). LYMPH already falsified linear extrapolation at sd>1 with
   saturation ~0.4. Bench-set points: NCBI783 0.425, TENX99 0.338, TENX95 0.271,
   NCBI785 0.268 (+-0.1 band; sd>1 entries expected to saturate near ~0.4 rather than
   follow the line). all_spots entries (sd 2-3.9) are far outside the fit range: the
   line predicts impossible r>0.6-1.15; we predict instead a band 0.4-0.8 under
   log1p_raw driven by the zero/nonzero bimodal structure, low confidence.
P3 (official-caliber headline): zero-pixel library null, official caliber (log1p_raw,
   bench_patches spots, var50 panel, per-gene Pearson mean then 4-fold mean) lands in
   [0.30, 0.50]. Leaderboard (gen4_hest README, IDC column, 26 models): best 0.6024
   (H-Optimus-1), worst 0.4739 (ResNet50). PREDICTION: unlike LYMPH_IDC, the null does
   NOT beat the leaderboard best; it may approach or exceed the ResNet50 floor.
   Falsification: null < 0.25 or > 0.60 would falsify our depth-based expectation.
P4 (H2 zero-biology): sim >= real (median r) in >= 6/8 sample x spot-set conditions
   under log1p_raw, and >= 5/8 under cp10k_panel — i.e., a no-biology simulation
   reproduces most of what the library null captures.
P5 (H4 ceiling): median r_tech under log1p_raw > 0.8 for the two 10x samples (deep
   libs); cp10k_panel ceilings materially lower (0.4-0.8 band).
P6 (H5 r_nbr, k=6): median r_nbr >= H1 libnull median r in most conditions (Moran's I
   of log lib 0.6-0.96), i.e., a pixel-free spatial-neighbor predictor is also
   competitive.
