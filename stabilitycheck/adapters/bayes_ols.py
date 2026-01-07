from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats

from .base import BaseAdapter, EstimationResult
from ..math_utils import fit_ols_cluster, oneway_demean, twoway_demean, winsorize


def _transform_y(y: np.ndarray, how: str) -> np.ndarray:
    how = str(how).lower()
    if how in ("level", "identity", "none"):
        return y
    if how in ("log",):
        return np.log(np.clip(y, 1e-12, None))
    if how in ("log1p",):
        return np.log1p(np.clip(y, 0.0, None))
    raise ValueError(f"Unknown y_transform: {how}")


class BayesOLSAdapter(BaseAdapter):
    """Conjugate Bayesian linear regression (Normal-Inverse-Gamma).

    This adapter is intentionally lightweight and deterministic.

    Model:
      y = X b + eps, eps ~ N(0, sigma^2)
      b | sigma^2 ~ N(b0, sigma^2 * V0)
      sigma^2 ~ InvGamma(a0, b0)

    Output:
      - psi_hat: posterior mean for coefficient-of-interest
      - U: posterior covariance matrix for psi_hat (1x1)
      - p: a two-sided posterior tail probability ("Bayesian p") for sign=0

    Notes:
      - If you want a full Bayesian workflow (hierarchical, nonconjugate, MCMC),
        it is better to implement an external adapter. This provides a solid baseline.
    """

    def __init__(
        self,
        colmap: Optional[Dict[str, str]] = None,
    ):
        self.colmap = colmap or {"y": "y", "X_prefix": "x", "unit": "unit", "time": "time"}

    def fit(self, df: pd.DataFrame, spec: Dict[str, Any]) -> EstimationResult:
        try:
            p = spec.get("p", {})
            e = spec.get("e", {})
            i = spec.get("i", {})

            y_col = e.get("y_col", self.colmap["y"])
            x_cols = e.get("x_cols")
            x_prefix = str(e.get("x_prefix", self.colmap.get("X_prefix", "x")))
            if x_cols is None:
                x_cols = [c for c in df.columns if c.startswith(x_prefix)]

            if len(x_cols) == 0:
                raise ValueError("BayesOLS requires x_cols (or x_prefix with matching columns)")

            winsor_p = p.get("winsor_p", None)
            y_transform = p.get("y_transform", "level")

            y_raw = df[y_col].to_numpy(dtype=float)
            y_w = winsorize(y_raw, winsor_p)
            y = _transform_y(y_w, y_transform)

            X = df[x_cols].to_numpy(dtype=float)

            # optional FE residualization
            fe = str(e.get("fe", "none")).lower()
            if fe in ("twoway", "2way", "two-way"):
                uc = df[e.get("unit_col", self.colmap.get("unit", "unit"))].astype(int).to_numpy()
                tc = df[e.get("time_col", self.colmap.get("time", "time"))].astype(int).to_numpy()
                nU = int(uc.max()) + 1
                nT = int(tc.max()) + 1
                y = twoway_demean(y, uc, tc, nU, nT)
                X = np.column_stack([twoway_demean(X[:, j], uc, tc, nU, nT) for j in range(X.shape[1])])
                add_const = False
            elif fe in ("unit", "u"):
                uc = df[e.get("unit_col", self.colmap.get("unit", "unit"))].astype(int).to_numpy()
                nU = int(uc.max()) + 1
                y = oneway_demean(y, uc, nU)
                X = np.column_stack([oneway_demean(X[:, j], uc, nU) for j in range(X.shape[1])])
                add_const = True
            elif fe in ("time", "t"):
                tc = df[e.get("time_col", self.colmap.get("time", "time"))].astype(int).to_numpy()
                nT = int(tc.max()) + 1
                y = oneway_demean(y, tc, nT)
                X = np.column_stack([oneway_demean(X[:, j], tc, nT) for j in range(X.shape[1])])
                add_const = True
            else:
                add_const = True

            if add_const:
                X = np.column_stack([np.ones(len(y)), X])
                names = ["const"] + list(x_cols)
            else:
                names = list(x_cols)

            # OLS baseline (for posterior updates)
            cluster_col = i.get("cluster_col")
            clusters = None
            if cluster_col is not None and cluster_col in df.columns:
                clusters = df[cluster_col].to_numpy()
            beta, se, pvals, V = fit_ols_cluster(y, X, clusters)

            # conjugate prior
            # prior on b: N(b0, sigma^2 V0)
            # default: weakly informative ridge-like prior
            k = X.shape[1]
            b0 = np.array(e.get("b0", [0.0] * k), dtype=float)
            if b0.shape[0] != k:
                b0 = np.resize(b0, k)

            v0_scale = float(e.get("v0_scale", 10.0))
            V0 = np.eye(k) * v0_scale

            a0 = float(e.get("a0", 2.0))
            b0_ig = float(e.get("b0_ig", 1.0))

            XtX = X.T @ X
            XtY = X.T @ y

            V0_inv = np.linalg.inv(V0)
            Vn_inv = V0_inv + XtX
            Vn = np.linalg.inv(Vn_inv)
            bn = Vn @ (V0_inv @ b0 + XtY)

            # posterior IG params
            yTy = float(y.T @ y)
            b0V0invb0 = float(b0.T @ V0_inv @ b0)
            bnVninvbn = float(bn.T @ Vn_inv @ bn)

            an = a0 + len(y) / 2.0
            bn_ig = b0_ig + 0.5 * (yTy + b0V0invb0 - bnVninvbn)

            # coefficient-of-interest index
            poi_name = str(e.get("poi", names[1] if "const" in names and len(names) > 1 else names[0]))
            if poi_name not in names:
                raise ValueError(f"poi '{poi_name}' not in regressors: {names}")
            j = names.index(poi_name)

            # marginal posterior for b_j is Student-t
            # mean = bn[j], scale^2 = (bn_ig/an) * Vn[j,j]
            scale2 = (bn_ig / an) * float(Vn[j, j])
            scale = float(np.sqrt(max(scale2, 1e-18)))
            df_t = 2.0 * an

            # two-sided posterior tail prob around 0
            t0 = (0.0 - bn[j]) / scale
            # P(b <= 0) under t
            p_left = float(stats.t.cdf(t0, df=df_t))
            p_two = float(2.0 * min(p_left, 1.0 - p_left))

            psi_hat = np.array([float(bn[j])], dtype=float)
            U = np.array([[scale2]], dtype=float)

            Q = {
                "family": "bayes_ols",
                "poi": poi_name,
                "fe": fe,
                "n": int(len(y)),
                "prior": {"v0_scale": v0_scale, "a0": a0, "b0_ig": b0_ig},
                "posterior": {"df": df_t},
            }
            return EstimationResult(ok=True, psi_hat=psi_hat, U=U, p=np.array([p_two]), Q=Q, n=int(len(y)))
        except Exception as ex:
            return EstimationResult(ok=False, psi_hat=np.array([]), U=np.array([[]]), msg=str(ex))
