# stnull audit report

stnull 0.3.0 | 2026-09-08T09:30:42Z | space=`log1p_cp10k:panel` | seed=0

> This report evaluates THIS SET OF PREDICTIONS. It says nothing about the information content of H&E in general.

```
model median r = 0.1051
  vs strongest zero-pixel null (depth_comp_null) = 0.3169
  model beats that null in 0/3 (0%) of sections
------------------------------------------------------------------------
stnull 0.3.0 | space=log1p_cp10k:panel | 812 spots x 100 genes x 3 sections | seed=0
perm=perm_block x200 | boot x200 | null_fit=cv_within_section | depth_proxy=observed

[1] HEADLINE
    median-of-genes : 0.1051  [perm_block floor 0.0319]  CI[0.0311,0.1086]  n=812/100  (ci_kind=section; perm med=-0.0014 max=0.0549 nperm=200)
    mean-of-genes   : 0.1115  [perm_block floor 0.0316]  CI[0.0532,0.1139]  n=812/100  (mean over genes; ci_kind=section; perm med=0.0007 max=0.0594 nperm=200)
[2] DEPTH
    depth_null      : 0.3029  [perm_block floor 0.0686]  CI[0.1744,0.4135]  n=812/100  (zero pixels: one scalar (log lib) per spot; ci_kind=section; perm med=-0.0044 max=0.1270 nperm=200)
    depth_1d(img)   : n/a  [no floor]  CI[n/a]  n=812/0  <na>  (depth_proxy='observed': l_hat IS the measured log library size, so this readout is fixed by construction and says nothing about the model. Pass lib_pred= or depth_proxy='from_pred' to measure the depth the model itself carries.)
    DG (ctrl depth) : 0.0288  [perm_block floor 0.0223]  CI[-0.0294,0.0304]  n=812/100  (ci_kind=section; perm med=-0.0003 max=0.0487 nperm=200)
    share of r from depth : 0.7350
    corr(l_hat, log lib)  : n/a  [no floor]  CI[n/a]  n=812/0  <na>  (depth_proxy='observed': l_hat IS the measured log library size, so this readout is fixed by construction and says nothing about the model. Pass lib_pred= or depth_proxy='from_pred' to measure the depth the model itself carries.)
[3] COMPOSITION (7 classes)
    cross null      : 0.1229  [perm_block floor 0.0727]  CI[0.1018,0.2451]  n=812/100  (class means, cv_within_section; ci_kind=section; perm med=-0.0164 max=0.1325 nperm=200)
    oracle null     : 0.2397  [perm_block floor 0.0704]  CI[0.2288,0.3027]  n=812/100  (UPPER_BOUND: class means read off the scored section; not eligible to be the strongest null; ci_kind=section; perm med=-0.0180 max=0.1385 nperm=200) <UPPER_BOUND>
    CRG (ctrl comp) : 0.0435  [perm_block floor 0.0222]  CI[-0.0010,0.0863]  n=812/100  (ci_kind=section; perm med=0.0006 max=0.0390 nperm=200)
[4] SPATIAL
    r_nbr (true-y smoothing, no pixels) : 0.2020  [perm_block floor 0.0520]  CI[0.1359,0.3201]  n=812/100  (mean of the neighbours' TRUE y; the image is never used; ci_kind=section; perm med=-0.0241 max=0.1133 nperm=200)
    nbr beats model in 3/3 (100%) of sections
    lever: smoothing the ground truth   : 0.0205
    Moran's I of model residual         : 0.1221
[5] SELECTION AND LEVERS
    test-selected top-10   : 0.3491  [perm_selection floor 0.2336]  CI[0.2244,0.4224]  n=812/10  (genes ranked by r ON THE TEST SET; ci_kind=section; perm med=0.1544 max=0.2767 nperm=200)
    test-selected top-50   : 0.2021  [perm_selection floor 0.1124]  CI[0.1160,0.2623]  n=812/50  (genes ranked by r ON THE TEST SET; ci_kind=section; perm med=0.0587 max=0.1520 nperm=200)
[6] CEILING (split_half_full)
    r_tech          : 0.6649  [perm_free floor 0.1262]  CI[0.5386,0.6728]  n=812/100  (split_half_full; ci_kind=section; perm med=0.0221 max=0.1809 nperm=50)
    r / r_tech      : 0.1581
[7] ATTRIBUTION LADDER (across-section median)
    r_full  : 0.1051
    dg      : 0.0288
    crg     : 0.0435
    drg     : 0.0037
[9] DEGRADATION LEDGER
    - no is_train -> nulls are cross-fitted within section (null_fit=cv_within_section), which can be slightly optimistic relative to a train-only fit; only test-selected top-N readouts are shown

This report evaluates THESE PREDICTIONS, not the information content of H&E.
```

## 7. Attribution ladder

| section | patient | n_spots | r_full | r_full_floor | r_full_ci_lo | r_full_ci_hi | dg | dg_floor | dg_ci_lo | dg_ci_hi | crg | crg_floor | crg_ci_lo | crg_ci_hi | drg | drg_floor | drg_ci_lo | drg_ci_hi | drg_observed | drg_observed_floor | drg_observed_ci_lo | drg_observed_ci_hi | drg_above_floor | n_genes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A1 | A | 345 | 0.0311 | 0.0202 | 0.0216 | 0.0607 | 0.0304 | 0.0198 | 0.0094 | 0.0437 | -0.0010 | 0.0162 | -0.0157 | 0.0247 | 0.0037 | 0.0164 | -0.0135 | 0.0235 | 0.0037 | 0.0164 | n/a | n/a | False | 100 |
| B1 | B | 293 | 0.1051 | 0.1516 | 0.0660 | 0.1465 | -0.0294 | 0.0514 | -0.0539 | -0.0038 | 0.0435 | 0.0893 | 0.0037 | 0.0833 | -0.0028 | 0.0255 | -0.0185 | 0.0163 | -0.0028 | 0.0255 | n/a | n/a | False | 100 |
| C1 | C | 174 | 0.1086 | 0.0491 | 0.0711 | 0.1462 | 0.0288 | 0.0332 | 0.0003 | 0.0558 | 0.0863 | 0.0359 | 0.0633 | 0.1290 | 0.0513 | 0.0303 | 0.0202 | 0.0718 | 0.0513 | 0.0303 | n/a | n/a | False | 100 |

## 1. Headline

| section | patient | n_spots | n_eval | r_model | depth_null | depth_1d | cross_null | oracle_null | depth_comp_null | r_nbr | dg | crg | drg | drg_observed | moran | smooth_delta | sd_loglib | depth_r_of_pred |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A1 | A | 345 | 345 | 0.0311 | 0.1744 | n/a | 0.1229 | 0.2288 | 0.2208 | 0.1359 | 0.0304 | -0.0010 | 0.0037 | 0.0037 | 0.0819 | 0.0115 | 0.5252 | n/a |
| B1 | B | 293 | 293 | 0.1051 | 0.4135 | n/a | 0.2451 | 0.3027 | 0.4316 | 0.3201 | -0.0294 | 0.0435 | -0.0028 | -0.0028 | 0.1785 | 0.0896 | 1.2563 | n/a |
| C1 | C | 174 | 174 | 0.1086 | 0.3029 | n/a | 0.1018 | 0.2397 | 0.3169 | 0.2020 | 0.0288 | 0.0863 | 0.0513 | 0.0513 | 0.1221 | 0.0205 | 0.8201 | n/a |

## 8. What you may and may not say

- [OK] Across 3 sections the model's median per-gene r is 0.1051, BELOW the strongest zero-pixel null tried here (depth_comp_null = 0.3169). Model beats that null in 0/3 (0%) of sections.
- [OK] After controlling for sequencing depth, the predictions of this model retain r = 0.0288 (perm_block floor 0.0223, 95% CI [-0.0294,0.0304]).
- [OK] After controlling for depth and composition, THE PREDICTIONS OF THIS MODEL retain a median r = 0.0037 (0/3 sections have a bootstrap CI lower bound above their permutation floor). This is a property of these predictions, not an upper bound on the information content of H&E.
- [OUT OF SCOPE] H&E carries no information beyond depth and composition.
- [OUT OF SCOPE] DRG close to zero proves the task is impossible.
- [OUT OF SCOPE] DRG above its floor proves the model learned genuine morphology.
- [NO] Our r is higher than the number reported in paper X.

## 10. Reproducibility appendix

```
METHODS PARAGRAPH (paste and edit)
Predictions were audited with stnull 0.3.0. Expression targets were in the
log1p_cp10k:panel space; all correlations are per-gene Pearson r computed within a
section and aggregated first over genes (median and mean are both
reported) and then over sections. Every reported value is accompanied by
the null floor of the same readout rule, obtained by perm_block permutation of
the spot order within each section (200 draws, floor = 95th percentile of
the null distribution), and by a 200-draw bootstrap 95% confidence
interval. Three zero-pixel null models were fitted alongside the model: a
depth null (per-gene least squares on log library size only), a
composition null (class-mean lookup, cross-fitted within section, with an
oracle variant that reads class means off the scored section and is
reported as an upper bound), and a neighbour null (the mean of the true
expression of a spot's lattice neighbours). The attribution ladder
residualises both y and y_hat on X = [1], [1, l_hat], [1, pi] and
[1, l_hat, pi] via an SVD orthonormal basis (rank-deficiency safe),
giving r_full, DG, CRG and DRG respectively; l_hat is the sequencing
depth carried by the model's own predictions, obtained by cross-fitted
kernel ridge regression of log library size on y_hat within each section.
DRG is a property of the audited predictions and is not an upper bound on
the information content of H&E.

```