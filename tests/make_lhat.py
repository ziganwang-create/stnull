# -*- coding: utf-8 -*-
"""Rebuild l_hat (image-predicted log library size) for the HER2ST regression.

Usage:  python tests/make_lhat.py resnet50            -> tests/lhat_resnet50.npy
        python tests/make_lhat.py phikon_p50_cls      -> tests/lhat_phikon_p50_cls.npy
Read-only on prototype/; reuses prototype/metric_drg.py's fit_lhat so that the
regression compares stnull against the project's own frozen estimator.
"""
import os, sys
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[v] = "3"
import numpy as np, pandas as pd

PROTO = r"D:\project\STImage\prototype"
sys.path.insert(0, PROTO)
import metric_drg as M

bk = sys.argv[1] if len(sys.argv) > 1 else "resnet50"
out = {"resnet50": "lhat_resnet50.npy",
       "phikon_p50_cls": "lhat_phikon_p50.npy"}.get(bk, "lhat_%s.npy" % bk)
meta = pd.read_parquet(os.path.join(PROTO, "data", "meta.parquet"))
feats = np.load(os.path.join(PROTO, "data", "feats_%s.npy" % bk))
lh, alpha, dr = M.fit_lhat(meta, feats)
np.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), out), lh)
print("alphas", alpha)
print("depth_r", {k: round(v, 4) for k, v in dr.items()})
