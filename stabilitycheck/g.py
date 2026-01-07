from __future__ import annotations
from typing import List, Optional
import numpy as np

def ensure_cov(U: np.ndarray) -> np.ndarray:
    """
    Ensure U is a covariance matrix (k,k).
    If U is 1D (se vector), return diag(U^2).
    If U is scalar, interpret as se and return [[U^2]].
    """
    U = np.asarray(U, dtype=float)
    if U.ndim == 0:
        se = float(max(U, 1e-12))
        return np.array([[se*se]], dtype=float)
    if U.ndim == 1:
        se = np.maximum(U, 1e-12)
        return np.diag(se*se)
    if U.ndim == 2:
        return U
    raise ValueError("U must be scalar, 1D, or 2D.")

def mahalanobis_like(diff: np.ndarray, cov: np.ndarray) -> float:
    cov = ensure_cov(cov)
    k = cov.shape[0]
    diff = np.asarray(diff, dtype=float).reshape(-1)
    if diff.size != k:
        scale = float(np.sqrt(np.maximum(np.mean(np.diag(cov)), 1e-12)))
        return float(np.linalg.norm(diff) / scale)
    try:
        inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(cov)
    return float(np.sqrt(np.maximum(diff.T @ inv @ diff, 0.0)))

def g_vector(
    psi_d: np.ndarray,
    p_d: Optional[np.ndarray],
    psi_a: np.ndarray,
    U_a: np.ndarray,
    alpha: float,
    sign_index: int = 0,
) -> np.ndarray:
    psi_d = np.asarray(psi_d, dtype=float).reshape(-1)
    psi_a = np.asarray(psi_a, dtype=float).reshape(-1)
    U_a = ensure_cov(U_a)

    diff = psi_d - psi_a
    g_mag = mahalanobis_like(diff, U_a)

    si = int(sign_index) if sign_index is not None else 0
    si = max(0, min(si, psi_d.size-1))
    s_d = float(np.sign(psi_d[si]))
    s_a = float(np.sign(psi_a[si]))
    g_sign = 1.0 if s_d != s_a else 0.0

    p0 = 1.0
    if p_d is not None:
        p_d = np.asarray(p_d, dtype=float).reshape(-1)
        if p_d.size > 0:
            p0 = float(p_d[si if si < p_d.size else 0])
    g_inf = max(0.0, p0 - float(alpha))

    return np.array([g_mag, g_sign, g_inf], dtype=float)

def aggregate_anchors(G_list: List[np.ndarray], how: str) -> np.ndarray:
    G = np.vstack(G_list)
    how = str(how).lower()
    if how == "max":
        return G.max(axis=0)
    if how == "mean":
        return G.mean(axis=0)
    if how == "min":
        return G.min(axis=0)
    raise ValueError("Unknown anchor aggregation")

def scalarize(g_vec: np.ndarray, pnorm, w2: float, w3: float) -> float:
    g_vec = np.asarray(g_vec, dtype=float).reshape(-1)
    v = np.array([g_vec[0], float(w2)*g_vec[1], float(w3)*g_vec[2]], dtype=float)
    if pnorm == 1:
        return float(np.sum(np.abs(v)))
    if pnorm == 2:
        return float(np.sqrt(np.sum(v*v)))
    if pnorm == np.inf:
        return float(np.max(np.abs(v)))
    return float(np.linalg.norm(v, ord=pnorm))
