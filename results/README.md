# stnull results -- 11 cohorts, 10 organs

Everything in this directory was produced by auditing published spatial-gene-
expression benchmarks with `stnull`. The audit asks one question of each
benchmark: **how much of the reported correlation survives a model that never
looks at the image?**

The zero-pixel baseline is a per-gene ordinary least squares of expression on
`[1, log(total_counts)]`. One number per spot goes in. No pixel is ever read.

## What is here

| path | contents |
|---|---|
| `hb_MASTER_TABLE.csv` | One row per cohort, 23 columns: platform, split structure, leakage verdict, depth dispersion, the null's score under the official convention, the best published model and its score, how many published models the null matches or beats, the technical ceiling, and the H2/H3 verdicts. The `sources` column names the file each cell came from. |
| `hb_CROSS_COHORT.md` | The written synthesis across all 11 cohorts (in Chinese). Start here for the argument; use the master table for the numbers. |
| `leaderboard_cpu_sweep.csv` | Fourteen pathology foundation-model backbones run through the same HER2ST protocol: `hest_official_pcc`, `strict_median_r_36sec` (our patient-level protocol), and `drg_obs_8sec` (the depth-and-composition residual gain). Parameter counts span 21.7M to 1.1B; the residual gain spans 0.017 to 0.033. |
| `leaderboard_triage.csv` | Which published leaderboard entries were reproducible enough to enter the comparison, and why the rest were not. |
| `hb_verify_cross_indep.py`, `hb_verify_cross_indep_results.json` | An independent recomputation of the cross-cohort claims, run to check the aggregation rather than trust it. |
| `her2st/` | The discovery cohort: 36 sections, 8 patients. Nulls, DRG ladder, ceiling, and the full protocol-lever price list. |
| `hest_<cohort>/` | The ten HEST-Benchmark cohorts, out-of-sample. Each carries H1 nulls, H2 zero-biology simulation, H3 depth-structure prediction, H4 ceiling, H5 `r_nbr`, the depth pre-table, the leaderboard comparison, and the pre-registration that was timestamped before the audit ran. |

Each subdirectory has its own `README.md` with the column glossary for its files.

## The cohorts

| directory | cohort | organ | platform | samples / patients |
|---|---|---|---|---|
| `her2st/` | HER2ST | breast | legacy ST | 36 sections / 8 |
| `hest_lymph_idc/` | LYMPH_IDC | lymph node | Visium | 4 / 4 |
| `hest_idc/` | IDC | breast | Xenium (pseudo-Visium binned) | 4 / 4 |
| `hest_read/` | READ | rectum | Visium | 4 / 2 |
| `hest_paad/` | PAAD | pancreas | Xenium | 3 / 3 |
| `hest_skcm/` | SKCM | skin | Xenium | 2 / 2 |
| `hest_lung/` | LUNG | lung | Xenium | 2 / 2 |
| `hest_hcc/` | HCC | liver | Visium | 2 / 2 |
| `hest_coad/` | COAD | colon | Xenium | 4 / 2 groups |
| `hest_prad/` | PRAD | prostate | Visium | 23 / 2 |
| `hest_ccrcc/` | CCRCC | kidney | Visium | 24 / 24 |

## Headline

Under the official metric, the official caliber, the official panel, the
official spot set, and the official folds, the zero-pixel depth read-out:

- **matches or beats every published model** in 7 of the 10 leaderboard cohorts
  (LYMPH_IDC, READ, PAAD, LUNG, PRAD, CCRCC, HCC);
- **ranks second** in COAD, 0.0066 behind the leader;
- is **cleared by the whole leaderboard** in IDC and SKCM, which are the
  positive controls: real image signal exists there, and the null still accounts
  for the bulk of the score (66 to 71 percent of the leader).

Across model-by-task cells: the null matches or beats **190 of 243**, about 78
percent. On HER2ST it wins 36 of 36 sections, 0.230 against 0.113 for a 2048-d
image ridge under the same patient-level protocol.

## What this does and does not say

It does **not** say the published models are worthless, and it does not say the
null has found biology. Two guards in this bundle argue the other way:

1. `*zerobio*.csv` re-simulates the counts with **no spatial biology at all**,
   keeping only the real library sizes. On most cohorts the null scores as high
   or higher on the simulation than on the real data. So the score is largely
   what `log1p` does to sparse counts in the presence of depth variation -- a
   property of the measurement convention, not of the tissue.
2. `*ceiling*.csv` puts a technical ceiling on every cohort. Where that ceiling
   is 0.51 (HER2ST) a reported 0.30 is a different achievement than where the
   ceiling is 0.96 (LUNG).

The useful conclusion is narrower and harder: a Pearson r on this task is not
interpretable on its own. It has to be reported against the floor of its own
readout rule and under its own ceiling, which is what `stnull` computes.

## Reading discipline

- Aggregated numbers are approximate: plus or minus 0.02 to 0.05.
- Per-cohort sample counts run from 2 to 24, so no p-values are reported.
- The zero-pixel null uses each test spot's **true** sequencing depth, a
  quantity an image model cannot have. It is an audit baseline, not a
  competitor. The claim is about what a score proves, not about who wins.
- HER2ST is the discovery cohort and is in-sample throughout. The ten HEST
  cohorts are the replication, with pre-registrations timestamped before each
  run.

Zigan Wang
