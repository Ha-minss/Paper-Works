from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats

from .base import BaseAdapter, EstimationResult
from ..math_utils import winsorize


def _kernel_weights(u: np.ndarray, kernel: str) -> np.ndarray:
    """Kernel weights for u = |x-c|/h (clipped to [0,1])."""
    k = str(kernel).lower()
    u = np.clip(u, 0.0, 1.0)
    if k in ("triangular", "triangle", "tri"):
        return np.clip(1.0 - u, 0.0, 1.0)
    if k in ("uniform", "rect", "rectangle"):
        return np.ones_like(u)
    if k in ("epanechnikov", "epa"):
        return 0.75 * np.clip(1.0 - u * u, 0.0, 1.0)
    raise ValueError(f"Unknown kernel: {kernel}")


def _transform_y(y: np.ndarray, how: str) -> np.ndarray:
    how = str(how).lower()
    if how in ("level", "none"):
        return y
    if how == "log":
        # require positive
        return np.log(np.maximum(y, 1e-12))
    if how == "log1p":
        return np.log1p(np.maximum(y, 0.0))
    raise ValueError(f"Unknown y_transform: {how}")


def _fit_wls_robust(
    y: np.ndarray,
    X: np.ndarray,
    w: np.ndarray,
    cluster: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Weighted least squares with HC1 or cluster-robust covariance.

    Uses X'WX and weighted residuals u_w = sqrt(w) * u.
    """
    n, k = X.shape
    sw = np.sqrt(np.clip(w, 1e-12, np.inf))
    Xw = X * sw[:, None]
    yw = y * sw

    XtX = Xw.T @ Xw
    XtX_inv = np.linalg.pinv(XtX)

    beta = XtX_inv @ (Xw.T @ yw)
    resid = y - (X @ beta)
    uw = resid * sw

    if cluster is None:
        meat = Xw.T @ (Xw * (uw * uw)[:, None])
        dof = max(n - k, 1)
        df_c = (n / dof) if dof > 0 else 1.0
    else:
        clusters = pd.unique(cluster)
        G = len(clusters)
        meat = np.zeros((k, k), dtype=float)
        for g in clusters:
            idx = cluster == g
            Sg = Xw[idx].T @ uw[idx]
            meat += np.outer(Sg, Sg)
        df_c = 1.0
        if G > 1 and n > k:
            df_c = (G / (G - 1.0)) * ((n - 1.0) / (n - k))
        dof = max(G - 1, 1)

    V = df_c * (XtX_inv @ meat @ XtX_inv)
    se = np.sqrt(np.maximum(np.diag(V), 1e-18))
    t = beta / se
    p = 2.0 * (1.0 - stats.t.cdf(np.abs(t), df=dof))

    return {
        "beta": beta,
        "se": se,
        "p": p,
        "V": V,
        "resid": resid,
    }


def _fit_wls_2sls(
    y: np.ndarray,
    X: np.ndarray,
    Z: np.ndarray,
    w: np.ndarray,
    cluster: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Weighted 2SLS with (optional) cluster-robust covariance.

    Formula:
      beta = (X'W Pz X)^{-1} X'W Pz y
      Pz = Z (Z'WZ)^{-1} Z'W

    Robust variance (cluster version):
      V = A^{-1} (X'WZ (Z'WZ)^{-1} (Z'W Ω W Z) (Z'WZ)^{-1} Z'WX) A^{-1}
      where A = X'W Pz X.

    Here Ω is based on u^2; cluster robust uses sums of Z'u within cluster.
    """
    n, k = X.shape
    sw = np.sqrt(np.clip(w, 1e-12, np.inf))
    Xw = X * sw[:, None]
    Zw = Z * sw[:, None]
    yw = y * sw

    ZtZ = Zw.T @ Zw
    ZtZ_inv = np.linalg.pinv(ZtZ)

    ZtX = Zw.T @ Xw
    Zty = Zw.T @ yw

    A = ZtX.T @ ZtZ_inv @ ZtX
    A_inv = np.linalg.pinv(A)

    beta = A_inv @ (ZtX.T @ ZtZ_inv @ Zty)

    resid = y - (X @ beta)
    uw = resid * sw

    if cluster is None:
        # Meat = Z'W diag(u^2) W Z = Zw' diag(uw^2) Zw
        Meat = Zw.T @ (Zw * (uw * uw)[:, None])
        dof = max(n - k, 1)
        df_c = (n / dof) if dof > 0 else 1.0
    else:
        clusters = pd.unique(cluster)
        G = len(clusters)
        q = Zw.shape[1]
        Meat = np.zeros((q, q), dtype=float)
        for g in clusters:
            idx = cluster == g
            Sg = Zw[idx].T @ uw[idx]
            Meat += np.outer(Sg, Sg)
        df_c = 1.0
        if G > 1 and n > k:
            df_c = (G / (G - 1.0)) * ((n - 1.0) / (n - k))
        dof = max(G - 1, 1)

    middle = ZtX.T @ ZtZ_inv @ Meat @ ZtZ_inv @ ZtX
    V = df_c * (A_inv @ middle @ A_inv)

    se = np.sqrt(np.maximum(np.diag(V), 1e-18))
    t = beta / se
    p = 2.0 * (1.0 - stats.t.cdf(np.abs(t), df=dof))

    return {
        "beta": beta,
        "se": se,
        "p": p,
        "V": V,
        "resid": resid,
    }


@dataclass
class RDColMap:
    y: str = "y"
    x: str = "x"
    # fuzzy RD (optional)
    d: str = "d"


class RDAdapter(BaseAdapter):
    """Regression Discontinuity (Sharp or Fuzzy), local polynomial WLS.

    Expected columns
    ----------------
    - y: outcome
    - x: running variable
    - (optional) d: realized treatment (for fuzzy RD)

    Design spec (dict)
    ------------------
    - p:
        - y_transform: {'level','log','log1p'}
        - winsor_p: Optional[float]
    - s:
        - h or bandwidth: Optional[float] (bandwidth)
        - min_n: int (minimum sample size)
    - e:
        - cutoff: float (default 0.0)
        - order: int (default 1)
        - kernel: {'triangular','uniform','epanechnikov'}
        - fuzzy: bool (default False)
    - i:
        - cluster_col: Optional[str]

    Output
    ------
    psi_hat: [tau]
    U:       1x1 covariance matrix
    """

    family = "RD"

    def __init__(self, colmap: Optional[Dict[str, str]] = None):
        cm = RDColMap()
        if colmap:
            for k, v in colmap.items():
                if hasattr(cm, k):
                    setattr(cm, k, v)
        self.colmap = cm

    def fit(self, data: Any, design_spec: Dict[str, Any]) -> EstimationResult:
        try:
            df = pd.DataFrame(data).copy()
            if self.colmap.y not in df.columns or self.colmap.x not in df.columns:
                return EstimationResult(False, np.array([]), np.array([[]]), msg="Missing y/x columns")

            p = design_spec.get("p", {})
            s = design_spec.get("s", {})
            e = design_spec.get("e", {})
            i = design_spec.get("i", {})

            y = pd.to_numeric(df[self.colmap.y], errors="coerce").to_numpy()
            x = pd.to_numeric(df[self.colmap.x], errors="coerce").to_numpy()

            cutoff = float(e.get("cutoff", 0.0))
            xc = x - cutoff

            winsor_p = p.get("winsor_p", None)
            if winsor_p is not None:
                y = winsorize(y, float(winsor_p))

            y_transform = p.get("y_transform", "level")
            y = _transform_y(y, y_transform)

            # bandwidth
            h = s.get("h", s.get("bandwidth", None))
            if h is None:
                # heuristic: 35th percentile distance to cutoff (stable, avoids too tiny window)
                h = float(np.quantile(np.abs(xc), 0.35))
                if not np.isfinite(h) or h <= 0:
                    h = float(np.std(xc))
            h = float(max(h, 1e-8))

            min_n = int(s.get("min_n", 50))

            # kernel weights
            u = np.abs(xc) / h
            w = _kernel_weights(u, e.get("kernel", "triangular"))

            keep = (u <= 1.0) & np.isfinite(y) & np.isfinite(xc)
            if keep.sum() < min_n:
                return EstimationResult(False, np.array([]), np.array([[]]), msg=f"Too few obs in bandwidth (n={keep.sum()})")

            yk = y[keep]
            xk = xc[keep]
            wk = w[keep]

            # cluster
            cluster_col = i.get("cluster_col", None)
            cluster = None
            if cluster_col is not None and cluster_col in df.columns:
                cluster = pd.to_numeric(df.loc[keep, cluster_col], errors="coerce").fillna(-1).to_numpy()

            order = int(e.get("order", 1))
            order = max(1, min(order, 5))

            z = (xk >= 0).astype(float)  # cutoff indicator

            fuzzy = bool(e.get("fuzzy", False))

            if not fuzzy:
                # Sharp RD: y = a + tau*z + sum_j b_j x^j + sum_j c_j z*x^j + eps
                cols = [np.ones_like(xk), z]
                for j in range(1, order + 1):
                    xj = xk ** j
                    cols.append(xj)
                    cols.append(z * xj)
                X = np.column_stack(cols)

                res = _fit_wls_robust(yk, X, wk, cluster=cluster)
                tau = float(res["beta"][1])
                se = float(res["se"][1])
                pv = float(res["p"][1])
                V = np.array([[float(res["V"][1, 1])]])

                Q = {
                    "family": self.family,
                    "cutoff": cutoff,
                    "h": h,
                    "kernel": str(e.get("kernel", "triangular")),
                    "order": order,
                    "fuzzy": False,
                    "n_used": int(len(yk)),
                }

                return EstimationResult(True, np.array([tau]), V, p=np.array([pv]), Q=Q, n=int(len(yk)))

            # Fuzzy RD: endog d, instrument z
            if self.colmap.d not in df.columns:
                return EstimationResult(False, np.array([]), np.array([[]]), msg="Missing d column for fuzzy RD")

            d = pd.to_numeric(df[self.colmap.d], errors="coerce").to_numpy()[keep]
            if not np.isfinite(d).all():
                return EstimationResult(False, np.array([]), np.array([[]]), msg="NaNs in d within bandwidth")

            # Regressors X: [1, d, x^j, z*x^j]
            X_cols = [np.ones_like(xk), d]
            Z_cols = [np.ones_like(xk), z]  # instruments include z for d
            for j in range(1, order + 1):
                xj = xk ** j
                zxj = z * xj
                # exog controls
                X_cols.append(xj)
                X_cols.append(zxj)
                # instruments: include xj, zxj too
                Z_cols.append(xj)
                Z_cols.append(zxj)
            X = np.column_stack(X_cols)
            Zm = np.column_stack(Z_cols)

            res = _fit_wls_2sls(yk, X, Zm, wk, cluster=cluster)
            tau = float(res["beta"][1])
            se = float(res["se"][1])
            pv = float(res["p"][1])
            V = np.array([[float(res["V"][1, 1])]])

            Q = {
                "family": self.family,
                "cutoff": cutoff,
                "h": h,
                "kernel": str(e.get("kernel", "triangular")),
                "order": order,
                "fuzzy": True,
                "n_used": int(len(yk)),
            }

            return EstimationResult(True, np.array([tau]), V, p=np.array([pv]), Q=Q, n=int(len(yk)))

        except Exception as ex:
            return EstimationResult(False, np.array([]), np.array([[]]), msg=f"RDAdapter error: {ex}")
