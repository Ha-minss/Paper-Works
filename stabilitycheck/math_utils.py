from __future__ import annotations
import numpy as np
from scipy import stats

def winsorize(y: np.ndarray, p):
    if p is None:
        return y
    p = float(p)
    if p <= 0:
        return y
    lo = np.quantile(y, p)
    hi = np.quantile(y, 1-p)
    return np.clip(y, lo, hi)

def oneway_demean(v: np.ndarray, code: np.ndarray, nG: int) -> np.ndarray:
    code = code.astype(int)
    cnt = np.bincount(code, minlength=nG).astype(float)
    s   = np.bincount(code, weights=v, minlength=nG).astype(float)
    m = np.zeros(nG)
    mask = cnt > 0
    m[mask] = s[mask] / cnt[mask]
    return v - m[code]

def twoway_demean(v: np.ndarray, unit_code: np.ndarray, time_code: np.ndarray, nU: int, nT: int) -> np.ndarray:
    unit_code = unit_code.astype(int); time_code = time_code.astype(int)
    overall = v.mean()

    cnt_u = np.bincount(unit_code, minlength=nU).astype(float)
    sum_u = np.bincount(unit_code, weights=v, minlength=nU).astype(float)
    mean_u = np.zeros(nU)
    mu = cnt_u > 0
    mean_u[mu] = sum_u[mu] / cnt_u[mu]

    cnt_t = np.bincount(time_code, minlength=nT).astype(float)
    sum_t = np.bincount(time_code, weights=v, minlength=nT).astype(float)
    mean_t = np.zeros(nT)
    mt = cnt_t > 0
    mean_t[mt] = sum_t[mt] / cnt_t[mt]

    return v - mean_u[unit_code] - mean_t[time_code] + overall

def fit_ols_cluster(y: np.ndarray, X: np.ndarray, cluster_ids: np.ndarray):
    n, k = X.shape
    XtX = X.T @ X
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(XtX)

    beta = XtX_inv @ (X.T @ y)
    u = y - X @ beta

    clusters = np.unique(cluster_ids)
    G = len(clusters)
    meat = np.zeros((k,k))
    for g in clusters:
        idx = (cluster_ids == g)
        Xg = X[idx]
        ug = u[idx][:,None]
        meat += (Xg.T @ ug) @ (ug.T @ Xg)

    df_c = (G/(G-1)) * ((n-1)/(n-k)) if (G>1 and n>k) else 1.0
    V = df_c * (XtX_inv @ meat @ XtX_inv)

    se = np.sqrt(np.maximum(np.diag(V), 1e-12))
    tstat = beta / se
    dof = max(G-1, 1)
    pval = 2*(1 - stats.t.cdf(np.abs(tstat), df=dof))
    return beta, se, pval, V

def cluster_sandwich_mle(score: np.ndarray, hess: np.ndarray, cluster_ids: np.ndarray):
    k = hess.shape[0]
    try:
        bread = np.linalg.inv(hess)
    except np.linalg.LinAlgError:
        bread = np.linalg.pinv(hess)

    clusters = np.unique(cluster_ids)
    meat = np.zeros((k,k))
    for g in clusters:
        idx = (cluster_ids == g)
        sg = score[idx].sum(axis=0).reshape(-1,1)
        meat += sg @ sg.T

    V = bread @ meat @ bread
    se = np.sqrt(np.maximum(np.diag(V), 1e-12))
    return V, se
