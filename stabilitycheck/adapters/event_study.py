from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .base import BaseAdapter, EstimationResult
from ..math_utils import (
    fit_ols_cluster,
    oneway_demean,
    twoway_demean,
    winsorize,
)


def _transform_y(y: np.ndarray, how: str) -> np.ndarray:
    how = str(how).lower()
    if how in ("level", "none"):
        return y
    if how == "log":
        return np.log(np.maximum(y, 1e-12))
    if how == "log1p":
        return np.log1p(np.maximum(y, 0.0))
    raise ValueError(f"Unknown y_transform: {how}")


def _build_event_dummies(
    time: np.ndarray,
    treat: np.ndarray,
    policy_t: int,
    k_min: int,
    k_max: int,
    ref_k: int,
) -> Tuple[np.ndarray, List[int]]:
    rel = time.astype(int) - int(policy_t)
    ks = [k for k in range(int(k_min), int(k_max) + 1) if k != int(ref_k)]
    X = np.column_stack([(treat * (rel == k)).astype(float) for k in ks])
    return X, ks


class EventStudyAdapter(BaseAdapter):
    """Canonical two-way-FE event study for a single adoption date.

    Expects (by default) columns:
      - y: outcome
      - unit: panel entity id
      - time: integer time index
      - treat: treated-group indicator (time-invariant)

    The model (twoway FE) is:
      y_it = sum_{k in [k_min,k_max], k!=ref} beta_k * (treat_i * 1[t - policy_t = k])
             + controls_it + unitFE + timeFE + error

    Returns psi_hat = [beta_{k1}, beta_{k2}, ...] in increasing k order.
    """

    def __init__(self, colmap: Optional[Dict[str, str]] = None):
        self.colmap = colmap or {
            "y": "y",
            "unit": "unit",
            "time": "time",
            "treat": "treat",
        }

    def fit(self, data: pd.DataFrame, design_spec: Dict[str, Any]) -> EstimationResult:
        try:
            df = data.copy()

            y_col = self.colmap["y"]
            unit_col = self.colmap["unit"]
            time_col = self.colmap["time"]
            treat_col = self.colmap["treat"]

            e = design_spec.get("e", {})
            p = design_spec.get("p", {})
            i = design_spec.get("i", {})

            policy_t = int(e.get("policy_t", e.get("policy_time", 0)))
            k_min = int(e.get("k_min", -12))
            k_max = int(e.get("k_max", 12))
            ref_k = int(e.get("ref_k", -1))
            fe = str(e.get("fe", "twoway")).lower()  # twoway / unit / time / none

            controls: List[str] = list(e.get("controls", []))

            # sample: optional drop tail periods
            drop_last_k = int(design_spec.get("s", {}).get("drop_last_k", 0))
            if drop_last_k > 0:
                tmax = int(df[time_col].max())
                df = df[df[time_col] <= (tmax - drop_last_k)].copy()

            # y preprocessing
            y_raw = df[y_col].to_numpy(dtype=float)
            y_w = winsorize(y_raw, p.get("winsor_p", None))
            y = _transform_y(y_w, p.get("y_transform", "level"))

            unit = df[unit_col].astype(int).to_numpy()
            time = df[time_col].astype(int).to_numpy()
            treat = df[treat_col].to_numpy(dtype=float)

            X_es, ks = _build_event_dummies(time, treat, policy_t, k_min, k_max, ref_k)

            # controls
            X_ctrl = None
            if controls:
                X_ctrl = df[controls].to_numpy(dtype=float)

            if X_ctrl is None:
                X = X_es
                names = [f"k_{k}" for k in ks]
            else:
                X = np.column_stack([X_es, X_ctrl])
                names = [f"k_{k}" for k in ks] + controls

            # FE transformations
            if fe == "twoway":
                # within transform, no intercept
                y_t = twoway_demean(y, unit, time)
                X_t = np.column_stack([twoway_demean(X[:, j], unit, time) for j in range(X.shape[1])])
                X_use = X_t
                y_use = y_t
                add_intercept = False
            elif fe == "unit":
                y_t = oneway_demean(y, unit)
                X_t = np.column_stack([oneway_demean(X[:, j], unit) for j in range(X.shape[1])])
                X_use = np.column_stack([np.ones(len(y_t)), X_t])
                y_use = y_t
                add_intercept = True
                names = ["const"] + names
            elif fe == "time":
                y_t = oneway_demean(y, time)
                X_t = np.column_stack([oneway_demean(X[:, j], time) for j in range(X.shape[1])])
                X_use = np.column_stack([np.ones(len(y_t)), X_t])
                y_use = y_t
                add_intercept = True
                names = ["const"] + names
            elif fe == "none":
                X_use = np.column_stack([np.ones(len(y)), X])
                y_use = y
                add_intercept = True
                names = ["const"] + names
            else:
                raise ValueError(f"Unknown fe={fe}")

            cluster_col = i.get("cluster_col", unit_col)
            clusters = df[cluster_col].astype(int).to_numpy() if cluster_col else unit

            beta, V = fit_ols_cluster(y_use, X_use, clusters)

            # identify ES block indices (skip intercept if any)
            start = 1 if add_intercept else 0
            end = start + len(ks)
            psi = beta[start:end]
            cov = V[start:end, start:end]
            se = np.sqrt(np.maximum(np.diag(cov), 1e-18))
            tstat = psi / se
            dof = max(len(pd.unique(clusters)) - 1, 1)
            pvals = 2.0 * (1.0 - stats.t.cdf(np.abs(tstat), df=dof))

            Q = {
                "family": "event_study",
                "fe": fe,
                "policy_t": policy_t,
                "k_min": k_min,
                "k_max": k_max,
                "ref_k": ref_k,
                "k_list": ks,
                "controls": controls,
                "n": int(len(y_use)),
                "cluster": cluster_col,
            }

            return EstimationResult(
                ok=True,
                psi_hat=psi.astype(float),
                U=cov.astype(float),
                p=pvals.astype(float),
                Q=Q,
                n=int(len(y_use)),
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
