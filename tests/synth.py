# -*- coding: utf-8 -*-
"""Synthetic ground-truth generator for stnull validation.

Every scenario has a KNOWN correct answer, computed analytically from the
generative parameters by residualising on the TRUE latent depth and the TRUE
class labels (the 'oracle ladder').  The tool's job is to reproduce it without
being told which latent is which.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "3")
import numpy as np

GRID = (14, 15)          # 210 spots per section
N_SEC = 3
N_GENE = 60
N_CLASS = 6


def _smooth_field(gx, gy, rng, scale=3.0):
    """Spatially autocorrelated standard field on a lattice (gaussian blur)."""
    n = len(gx)
    raw = rng.normal(size=n)
    d2 = (gx[:, None] - gx[None, :]) ** 2 + (gy[:, None] - gy[None, :]) ** 2
    K = np.exp(-d2 / (2.0 * scale ** 2))
    K = K / K.sum(1, keepdims=True)
    f = K @ raw
    f = (f - f.mean()) / (f.std() + 1e-12)
    return f


def _noisy(f, rho, rng):
    """A copy of f with corr(f, out) ~= rho."""
    e = rng.normal(size=len(f))
    e = (e - e.mean()) / (e.std() + 1e-12)
    out = rho * f + np.sqrt(max(1e-9, 1 - rho ** 2)) * e
    return (out - out.mean()) / (out.std() + 1e-12)


def make(scenario, seed, n_sec=N_SEC, n_gene=N_GENE, sigma=None, rho=0.85):
    """Return dict with y, pred, lib, labels, coords, section, and oracle latents."""
    rng = np.random.default_rng(seed)
    W = dict(depth=(1.0, 0.0, 0.0), signal=(0.0, 1.0, 0.0),
             mixed=(1.0, 1.0, 0.0), noise=(0.0, 1.0, 0.0),
             comp=(0.0, 0.0, 1.0), selection=(0.0, 1.0, 0.0))[scenario]
    wd, wu, wc = W
    if sigma is None:
        sigma = {"depth": 3.0, "signal": 3.0, "mixed": 4.2, "noise": 3.0,
                 "comp": 3.0, "selection": 3.0}[scenario]
    # gene loadings shared across sections (a real panel)
    b = rng.normal(size=n_gene)                       # depth loadings
    c = rng.normal(size=n_gene)                       # signal loadings
    M = rng.normal(size=(n_gene, N_CLASS))            # class means
    base = rng.normal(size=n_gene) * 2.0
    Y, P, LIB, LAB, XY, SEC, ZT, UT = [], [], [], [], [], [], [], []
    for s in range(n_sec):
        gx, gy = np.meshgrid(np.arange(GRID[0]), np.arange(GRID[1]), indexing="ij")
        gx = gx.ravel().astype(float); gy = gy.ravel().astype(float)
        n = len(gx)
        z = _smooth_field(gx, gy, rng, 3.0)           # latent depth
        u = _smooth_field(gx, gy, rng, 3.0)           # latent depth-free signal
        q = _smooth_field(gx, gy, rng, 2.5)           # latent that sets classes
        cls = np.clip(np.digitize(q, np.quantile(q, np.linspace(0, 1, N_CLASS + 1)[1:-1])),
                      0, N_CLASS - 1)
        onehot = np.eye(N_CLASS)[cls]
        loglib = 8.0 + 0.35 * z
        lib = np.exp(loglib)
        y = (base[None, :] + wd * np.outer(z, b) + wu * np.outer(u, c)
             + wc * (onehot @ M.T) + sigma * rng.normal(size=(n, n_gene)))
        if scenario in ("noise", "selection"):
            # predictor is an independent spatially smooth field, zero true signal
            v = _smooth_field(gx, gy, rng, 3.0)
            cc = rng.normal(size=n_gene)
            p = np.outer(v, cc) + 0.5 * rng.normal(size=(n, n_gene))
        else:
            zh = _noisy(z, rho, rng)
            uh = _noisy(u, rho, rng)
            # class read out with 15% label error (an imperfect morphology reader)
            cls_h = cls.copy()
            flip = rng.random(n) < 0.15
            cls_h[flip] = rng.integers(0, N_CLASS, flip.sum())
            oh_h = np.eye(N_CLASS)[cls_h]
            p = (wd * np.outer(zh, b) + wu * np.outer(uh, c)
                 + wc * (oh_h @ M.T))
            p = p + 0.1 * rng.normal(size=(n, n_gene))
        Y.append(y); P.append(p); LIB.append(lib); LAB.append(cls.astype(str))
        XY.append(np.c_[gx, gy]); SEC.append(np.full(n, "S%d" % s)); ZT.append(z); UT.append(u)
    return dict(y=np.vstack(Y), pred=np.vstack(P), lib=np.concatenate(LIB),
                labels=np.concatenate(LAB), coords=np.vstack(XY),
                section=np.concatenate(SEC), z_true=np.concatenate(ZT),
                u_true=np.concatenate(UT))


# ------------------------------------------------------------ oracle ladder
def _resid(Mx, X):
    Q, _ = np.linalg.qr(X)
    return Mx - Q @ (Q.T @ Mx)


def _corr(A, B):
    A = A - A.mean(0, keepdims=True); B = B - B.mean(0, keepdims=True)
    den = np.sqrt((A * A).sum(0)) * np.sqrt((B * B).sum(0))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (A * B).sum(0) / den
    return np.where(den <= 1e-12, np.nan, r)


def oracle(d):
    """Ground-truth ladder using the TRUE latent depth and TRUE labels."""
    out = {}
    secs = sorted(set(d["section"].tolist()))
    rows = {"r_full": [], "dg": [], "crg": [], "drg": []}
    for s in secs:
        m = d["section"] == s
        y, p = d["y"][m], d["pred"][m]
        z = d["z_true"][m].reshape(-1, 1)
        cl = d["labels"][m]
        oh = np.eye(len(set(cl.tolist())))[
            np.searchsorted(np.array(sorted(set(cl.tolist()))), cl)]
        one = np.ones((m.sum(), 1))
        rows["r_full"].append(np.nanmedian(_corr(y, p)))
        for k, X in (("dg", np.hstack([one, z])),
                     ("crg", np.hstack([one, oh])),
                     ("drg", np.hstack([one, z, oh]))):
            rows[k].append(np.nanmedian(_corr(_resid(y, X), _resid(p, X))))
    for k, v in rows.items():
        out[k] = float(np.nanmedian(v))
    return out
