from __future__ import annotations
from typing import Any, Dict, List
import numpy as np
import pandas as pd

from .base import BaseAdapter, EstimationResult
from ..math_utils import winsorize, oneway_demean, twoway_demean, fit_ols_cluster

class OLSAdapter(BaseAdapter):
    family = "OLS"

    def fit(self, df: pd.DataFrame, spec: Dict[str, Any]) -> EstimationResult:
        if not isinstance(df, pd.DataFrame):
            return EstimationResult(False, np.array([]), np.array([]), msg="data must be DataFrame")

        p = spec.get("p", {}) or {}
        s = spec.get("s", {}) or {}
        e = spec.get("e", {}) or {}
        i = spec.get("i", {}) or {}

        ycol = str(e.get("y","y"))
        xcols = list(e.get("x", []))
        unit_col = str(e.get("unit_col","unit"))
        time_col = str(e.get("time_col","time"))
        uFE = int(e.get("uFE", 0))
        tFE = int(e.get("tFE", 0))

        for c in [ycol, unit_col, time_col] + xcols:
            if c not in df.columns:
                return EstimationResult(False, np.array([]), np.array([]), msg=f"missing column: {c}")

        d = df.copy()
        drop_k = int(s.get("drop_last_k", 0))
        if drop_k > 0:
            tmax = int(d[time_col].max())
            d = d[d[time_col] <= (tmax - drop_k)].copy()

        y_raw = d[ycol].to_numpy(dtype=float)
        y_w = winsorize(y_raw, p.get("winsor_p", None))
        tr = str(p.get("transform","level"))
        if tr == "level":
            y = y_w
        elif tr == "log1p":
            y = np.log1p(np.maximum(y_w, 0.0))
        elif tr == "log":
            y = np.log(np.maximum(y_w, 1e-12))
        else:
            return EstimationResult(False, np.array([]), np.array([]), msg=f"unknown transform={tr}")

        X = np.column_stack([d[c].to_numpy(dtype=float) for c in xcols]).astype(float)
        names = list(xcols)

        unit_code = d[unit_col].astype(int).to_numpy()
        time_code = d[time_col].astype(int).to_numpy()
        nU = int(df[unit_col].max()) + 1
        nT = int(df[time_col].max()) + 1

        if uFE == 1 and tFE == 1:
            y_use = twoway_demean(y, unit_code, time_code, nU, nT)
            X_use = np.column_stack([twoway_demean(X[:,j], unit_code, time_code, nU, nT) for j in range(X.shape[1])])
            names_use = names
        elif uFE == 1 and tFE == 0:
            y_t = oneway_demean(y, unit_code, nU)
            X_t = np.column_stack([oneway_demean(X[:,j], unit_code, nU) for j in range(X.shape[1])])
            y_use = y_t
            X_use = np.column_stack([np.ones(len(y_t)), X_t])
            names_use = ["const"] + names
        elif uFE == 0 and tFE == 1:
            y_t = oneway_demean(y, time_code, nT)
            X_t = np.column_stack([oneway_demean(X[:,j], time_code, nT) for j in range(X.shape[1])])
            y_use = y_t
            X_use = np.column_stack([np.ones(len(y_t)), X_t])
            names_use = ["const"] + names
        else:
            y_use = y
            X_use = np.column_stack([np.ones(len(y)), X])
            names_use = ["const"] + names

        cluster = str(i.get("cluster","none")).lower()
        if cluster == "unit":
            cluster_ids = unit_code
        elif cluster == "time":
            cluster_ids = time_code
        else:
            cluster_ids = np.arange(len(y_use), dtype=int)

        beta, se, pval, V = fit_ols_cluster(y_use, X_use, cluster_ids)

        psi_indices = i.get("psi_indices", None)
        if psi_indices is None:
            psi_indices = [names_use.index(xcols[0])] if xcols else [0]
        psi_indices = [int(j) for j in psi_indices]

        psi_hat = beta[psi_indices].astype(float)
        U = V[np.ix_(psi_indices, psi_indices)].astype(float)
        p_out = pval[psi_indices].astype(float)
        Q = {"names": names_use, "transform": tr, "uFE": uFE, "tFE": tFE}
        return EstimationResult(True, psi_hat, U, p=p_out, Q=Q, n=int(len(y_use)))
