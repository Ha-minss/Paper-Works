from __future__ import annotations
from typing import Any, Dict
import numpy as np
import pandas as pd
from scipy import stats

from .base import BaseAdapter, EstimationResult
from ..math_utils import winsorize, oneway_demean, twoway_demean

class IV2SLSAdapter(BaseAdapter):
    family = "IV"

    def fit(self, df: pd.DataFrame, spec: Dict[str, Any]) -> EstimationResult:
        if not isinstance(df, pd.DataFrame):
            return EstimationResult(False, np.array([]), np.array([]), msg="data must be DataFrame")

        p = spec.get("p", {}) or {}
        s = spec.get("s", {}) or {}
        e = spec.get("e", {}) or {}
        i = spec.get("i", {}) or {}

        ycol = str(e.get("y","y"))
        endog = list(e.get("endog", []))
        exog  = list(e.get("exog", []))
        instr = list(e.get("instr", []))
        unit_col = str(e.get("unit_col","unit"))
        time_col = str(e.get("time_col","time"))
        uFE = int(e.get("uFE", 0))
        tFE = int(e.get("tFE", 0))

        req = [ycol, unit_col, time_col] + endog + exog + instr
        for c in req:
            if c not in df.columns:
                return EstimationResult(False, np.array([]), np.array([]), msg=f"missing column: {c}")
        if len(endog) == 0 or len(instr) == 0:
            return EstimationResult(False, np.array([]), np.array([]), msg="need endog and instr")

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

        X_end = np.column_stack([d[c].to_numpy(dtype=float) for c in endog])
        X_ex  = np.column_stack([d[c].to_numpy(dtype=float) for c in exog]) if len(exog)>0 else np.zeros((len(d),0))
        Z     = np.column_stack([d[c].to_numpy(dtype=float) for c in (instr + exog)])
        X     = np.column_stack([X_end, X_ex])

        unit_code = d[unit_col].astype(int).to_numpy()
        time_code = d[time_col].astype(int).to_numpy()
        nU = int(df[unit_col].max()) + 1
        nT = int(df[time_col].max()) + 1

        def demean_mat(M: np.ndarray):
            if uFE == 1 and tFE == 1:
                return np.column_stack([twoway_demean(M[:,j], unit_code, time_code, nU, nT) for j in range(M.shape[1])])
            if uFE == 1 and tFE == 0:
                return np.column_stack([oneway_demean(M[:,j], unit_code, nU) for j in range(M.shape[1])])
            if uFE == 0 and tFE == 1:
                return np.column_stack([oneway_demean(M[:,j], time_code, nT) for j in range(M.shape[1])])
            return M

        if uFE == 1 and tFE == 1:
            y_use = twoway_demean(y, unit_code, time_code, nU, nT)
            X_use = demean_mat(X); Z_use = demean_mat(Z)
            add_const = False
        elif uFE == 1 and tFE == 0:
            y_use = oneway_demean(y, unit_code, nU)
            X_use = demean_mat(X); Z_use = demean_mat(Z)
            add_const = True
        elif uFE == 0 and tFE == 1:
            y_use = oneway_demean(y, time_code, nT)
            X_use = demean_mat(X); Z_use = demean_mat(Z)
            add_const = True
        else:
            y_use = y; X_use = X; Z_use = Z
            add_const = True

        if add_const:
            X_use = np.column_stack([np.ones(len(y_use)), X_use])
            Z_use = np.column_stack([np.ones(len(y_use)), Z_use])

        # 2SLS
        try:
            ZtZ_inv = np.linalg.inv(Z_use.T @ Z_use)
        except np.linalg.LinAlgError:
            ZtZ_inv = np.linalg.pinv(Z_use.T @ Z_use)

        PZX = Z_use @ (ZtZ_inv @ (Z_use.T @ X_use))
        A = X_use.T @ PZX
        try:
            A_inv = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            A_inv = np.linalg.pinv(A)

        beta = A_inv @ (X_use.T @ (Z_use @ (ZtZ_inv @ (Z_use.T @ y_use))))
        u = y_use - X_use @ beta

        cluster = str(i.get("cluster","unit")).lower()
        if cluster == "unit":
            cl = unit_code
        elif cluster == "time":
            cl = time_code
        else:
            cl = np.arange(len(y_use), dtype=int)
        clusters = np.unique(cl)

        S = np.zeros((Z_use.shape[1], Z_use.shape[1]))
        for g in clusters:
            idx = (cl == g)
            Zg = Z_use[idx]
            ug = u[idx].reshape(-1,1)
            S += (Zg.T @ ug) @ (ug.T @ Zg)

        B = (X_use.T @ Z_use) @ (ZtZ_inv @ (S @ (ZtZ_inv @ (Z_use.T @ X_use))))
        V = A_inv @ B @ A_inv
        se = np.sqrt(np.maximum(np.diag(V), 1e-12))

        G = len(clusters)
        dof = max(G-1, 1)
        tstat = beta / np.maximum(se, 1e-12)
        pval = 2*(1 - stats.t.cdf(np.abs(tstat), df=dof))

        psi_indices = i.get("psi_indices", None)
        if psi_indices is None:
            psi_indices = [1] if add_const else [0]
        psi_indices = [int(j) for j in psi_indices]

        psi_hat = beta[psi_indices].astype(float)
        U = V[np.ix_(psi_indices, psi_indices)].astype(float)
        p_out = pval[psi_indices].astype(float)
        Q = {"transform": tr, "uFE": uFE, "tFE": tFE}
        return EstimationResult(True, psi_hat, U, p=p_out, Q=Q, n=int(len(y_use)))
