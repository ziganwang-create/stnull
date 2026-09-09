# PAAD (HEST-bench) zero-pixel audit — PRE-REGISTRATION

Written AFTER structure inspection (hb_paad_depth_pretable.csv, hb_paad_struct.json),
BEFORE any H1/H2/H4/H5 null-model output. Timestamp = file mtime (see also shell `date`
echoed in run log hb_paad_prereg_stamp.txt).

## Cohort structure (facts, from inspection)
- 3 samples, all **Xenium** (not Visium): TENX116 (3090x538), TENX126 (2190x541),
  TENX140 (4513x541). X dense uint16, integer counts.
- **in_tissue = 100% in all 3 samples** -> the all_spots / in_tissue duality collapses;
  one spot set only (disclosed, not a protocol violation).
- Gene universes are 3 different Xenium panels; intersection = **159 genes** (union 919).
  Official var_50genes.json: all 50 in the intersection. mean_50genes.json: only 9/50 in
  intersection (unusable across folds; not used by us).
- lib = obs/total_counts = **panel-total** (Xenium has no full transcriptome; disclosed
  deviation from "full-transcriptome lib" rule — no full transcriptome exists here).
- TENX116 lib_p10 = 0: >=10% of its spots have ZERO panel counts while flagged in_tissue.
- sd(log max(lib,1)): TENX116 4.106, TENX126 2.316, TENX140 2.101 — ALL far beyond the
  HER2ST fit range [0.389, 1.428]; LYMPH already falsified linear extrapolation at sd>1.

## Patient mapping (argued, not assumed)
HEST_v1_1_0.csv rows: TENX116 = "patient 2" (Xenium Human Multi-Tissue & Cancer Panel,
2023-11-16), TENX126 = "patient1" (Xenium Multimodal Cell Segmentation, 2024-03-19),
TENX140 = "patient 3" (Xenium Immuno-Oncology Profiling, 2024-05-15). Three distinct
(dataset_title, patient) pairs -> HESTData.py:1249 grouping yields 3 groups; splits/ has
3 folds, each test = exactly 1 sample, train = other 2. **Official splits are
patient-level; NO leakage.** (Caveat: patient labels come from 10x public dataset pages;
distinct titles/panels/dates make same-donor overlap implausible.)

## Panel tiers (adapted; disclosed)
- hest_var50 (official), trainHVG50 (seurat-flavor HVG on pooled train samples,
  universe = 159-gene intersection, eligibility min_cells_pct=0.10 per train sample),
- trainHVGmax = ALL eligible intersection genes (<=159) — the 737 tier is impossible on
  a ~159-gene universe; disclosed substitute.

## Pre-registered predictions (primary = ordering)
P1 (ORDERING, primary): H1 library-size null per-sample median r, official caliber
   (log1p:raw, hest_var50): **TENX116 > TENX126 > TENX140**, i.e. ordering follows
   sd(log lib). Falsified if any pairwise inversion.
P2 (point, weak; HER2ST linear fit r = 0.0296 + 0.2874*sd, ALL THREE EXTRAPOLATED,
   linear form already falsified at sd>1 by LYMPH; expect saturation):
   raw linear values 1.210 / 0.695 / 0.634 (TENX116 >1 = self-evidently invalid).
   Falsification bands instead: TENX116 med r in [0.50, 0.95]; TENX126 in [0.35, 0.80];
   TENX140 in [0.35, 0.80]. Below band = depth-structure account fails; above = n/a.
P3: log1p:raw exceeds log1p-CP10K:panel on the same spot set in all 3 samples, and the
   gap is LARGER than LYMPH's (~0.10-0.20) given the extreme depth variance:
   predicted gap >= 0.20 in median r for hest_var50.
P4: H2 zero-biology simulation: sim >= real in >= 5/6 of (sample x caliber) cells under
   log1p:raw; under cp10k_panel sim may drop below real (biology re-enters after
   depth removal) — no directional prereg for cp10k beyond "gap shrinks".
P5 (leaderboard): official-analog H1 (log1p:raw, hest_var50, official 3 folds,
   per-gene Pearson mean then fold mean) predicted in **[0.45, 0.80]**, i.e. likely
   >= leaderboard best PAAD = 0.5159 (Virchow, gen4_hest README line 114). Falsified
   below 0.35.
P6: H4 corrected ceiling (split-half + SB + sqrt) under log1p:raw will be HIGH
   (>0.8 median) because depth spread dominates; under cp10k_panel materially lower.
P7: H5 r_nbr(k=6) high in all samples (Moran's I of log lib is 0.84-0.92):
   predicted med r_nbr >= 0.5 under log1p:raw.

Approx tolerance on all point numbers: +/-0.02-0.05. n=3 samples -> no p-values.
