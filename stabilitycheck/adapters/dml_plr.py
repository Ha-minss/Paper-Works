from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import KFold

from .base import BaseAdapter, EstimationResult
from ..math_utils import oneway_demean, twoway_demean, winsorize


def _transform_y(y: np.ndarray, how: str) -> np.ndarray:
    how = str(how).lower()
    if how in ("level", "identity", "none"):
        return y
    if how in ("log",):
        return np.log(np.clip(y, 1e-12, None))
    if how in ("log1p",):
        return np.log1p(np.clip(y, 0.0, None))
    raise ValueError(f"Unknown y_transform: {how}")


def _make_learner(kind: str, seed: int):
    k = str(kind).lower()
    if k in ("rf", "random_forest"):
        return RandomForestRegressor(
            n_estimators=400,
            random_state=seed,
            min_samples_leaf=5,
            n_jobs=-1,
        )
    if k in ("lasso",):
        return Lasso(alpha=0.001, random_state=seed, max_iter=20000)
    if k in ("ridge",):
        return Ridge(alpha=1.0, random_state=seed)
    raise ValueError(f"Unknown learner: {kind}")


def _cluster_robust_var(psi: np.ndarray, v: np.ndarray, clusters: np.ndarray) -> float:
    """Asymptotic var for theta in PLR DML using cluster-robust aggregation.

    theta_hat = sum(v*y_res)/sum(v^2)
    psi_i = v_i * (y_res_i - theta_hat*v_i)
    var(theta) approx = Var(sum_g psi_g / n) / (E[v^2])^2
    """
    n = len(psi)
    denom = float(np.mean(v * v))
    if denom <= 0:
        return float("nan")

    ucl = np.unique(clusters)
    G = len(ucl)
    meat = 0.0
    for g in ucl:
        idx = clusters == g
        s = float(np.sum(psi[idx]))
        meat += s * s

    dfc = (G / (G - 1)) if G > 1 else 1.0
    var_sum = dfc * (meat / (n * n))
    return float(var_sum / (denom * denom))


def _robust_var(psi: np.ndarray, v: np.ndarray) -> float:
    n = len(psi)
    denom = float(np.mean(v * v))
    if denom <= 0:
        return float("nan")
    var_sum = float(np.mean(psi * psi) / n)
    return float(var_sum / (denom * denom))


class DMLPLRAdapter(BaseAdapter):
    """Double Machine Learning (Partial Linear Regression).

    Requires:
      - outcome y
      - treatment d (continuous)
      - covariates X (either explicit list or prefix-based selection)

    Supports optional FE residualization before ML (unit/time/twoway).
    """

    def __init__(
        self,
        colmap: Optional[Dict[str, str]] = None,
        seed: int = 7,
    ):
        self.colmap = colmap or {
            "y": "y",
            "d": "d",
            "unit": "unit",
            "time": "time",
        }
        self.seed = int(seed)

    def fit(self, df: pd.DataFrame, spec: Dict[str, Any]) -> EstimationResult:
        try:
            p = spec.get("p", {})
            e = spec.get("e", {})
            i = spec.get("i", {})

            y_col = e.get("y_col", self.colmap["y"])
            d_col = e.get("d_col", self.colmap["d"])

            # X columns
            x_cols: Optional[List[str]] = e.get("x_cols")
            x_prefix: str = str(e.get("x_prefix", "x"))
            if x_cols is None:
                x_cols = [c for c in df.columns if c.startswith(x_prefix)]
            if len(x_cols) == 0:
                raise ValueError("DML needs x_cols (or x_prefix with matching columns)")

            # preprocessing
            winsor_p = p.get("winsor_p", None)
            y_transform = p.get("y_transform", "level")

            y_raw = df[y_col].to_numpy(dtype=float)
            y_w = winsorize(y_raw, winsor_p)
            y = _transform_y(y_w, y_transform)

            d = df[d_col].to_numpy(dtype=float)
            X = df[x_cols].to_numpy(dtype=float)

            # optional FE residualization
            fe = str(e.get("fe", "none")).lower()  # none/unit/time/twoway
            if fe != "none":
                if fe in ("unit", "twoway"):
                    unit_col = e.get("unit_col", self.colmap.get("unit", "unit"))
                    ucode = df[unit_col].astype(int).to_numpy()
                    nU = int(ucode.max()) + 1
                if fe in ("time", "twoway"):
                    time_col = e.get("time_col", self.colmap.get("time", "time"))
                    tcode = df[time_col].astype(int).to_numpy()
                    nT = int(tcode.max()) + 1

                if fe == "unit":
                    y = oneway_demean(y, ucode, nU)
                    d = oneway_demean(d, ucode, nU)
                    X = np.column_stack([oneway_demean(X[:, j], ucode, nU) for j in range(X.shape[1])])
                elif fe == "time":
                    y = oneway_demean(y, tcode, nT)
                    d = oneway_demean(d, tcode, nT)
                    X = np.column_stack([oneway_demean(X[:, j], tcode, nT) for j in range(X.shape[1])])
                elif fe == "twoway":
                    y = twoway_demean(y, ucode, tcode, nU, nT)
                    d = twoway_demean(d, ucode, tcode, nU, nT)
                    X = np.column_stack([twoway_demean(X[:, j], ucode, tcode, nU, nT) for j in range(X.shape[1])])
                else:
                    raise ValueError(f"Unknown fe: {fe}")

            # ML nuisance
            learner_g = _make_learner(e.get("ml_g", "rf"), self.seed)
            learner_m = _make_learner(e.get("ml_m", "rf"), self.seed + 1)

            n_splits = int(e.get("n_splits", 5))
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.seed)

            y_hat = np.zeros_like(y, dtype=float)
            d_hat = np.zeros_like(d, dtype=float)

            for train, test in kf.split(X):
                g = _make_learner(e.get("ml_g", "rf"), self.seed)
                m = _make_learner(e.get("ml_m", "rf"), self.seed + 1)
                g.fit(X[train], y[train])
                m.fit(X[train], d[train])
                y_hat[test] = g.predict(X[test])
                d_hat[test] = m.predict(X[test])

            y_res = y - y_hat
            d_res = d - d_hat

            denom = float(np.sum(d_res * d_res))
            if denom <= 1e-12:
                raise ValueError("Near-zero residualized treatment variance")
            theta = float(np.sum(d_res * y_res) / denom)

            # influence
            psi = d_res * (y_res - theta * d_res)

            # variance
            cluster_col = i.get("cluster_col")
            if cluster_col is None and "unit" in self.colmap and self.colmap["unit"] in df.columns:
                cluster_col = self.colmap["unit"]

            if cluster_col is not None:
                clusters = df[cluster_col].to_numpy()
                var = _cluster_robust_var(psi, d_res, clusters)
                dof = max(len(np.unique(clusters)) - 1, 1)
            else:
                var = _robust_var(psi, d_res)
                dof = max(len(y) - 1, 1)

            se = float(np.sqrt(max(var, 1e-18)))
            t = theta / se if se > 0 else 0.0
            pval = float(2 * (1 - stats.t.cdf(abs(t), df=dof)))

            Q = {
                "fe": fe,
                "x_cols": x_cols,
                "n_splits": n_splits,
                "ml_g": str(e.get("ml_g", "rf")),
                "ml_m": str(e.get("ml_m", "rf")),
                "cluster": cluster_col,
            }

            return EstimationResult(
                ok=True,
                psi_hat=np.array([theta], dtype=float),
                U=np.array([[se * se]], dtype=float),
                p=np.array([pval], dtype=float),
                Q=Q,
                n=int(len(y)),
                msg="",
            )
        except Exception as ex:
            return EstimationResult(
                ok=False,
                psi_hat=np.array([]),
                U=np.array([[]]),
                p=None,
                Q={},
                n=0,
                msg=str(ex),
            )
