from __future__ import annotations
from typing import Any, Dict, List
import numpy as np
import pandas as pd
from scipy import stats

from .base import BaseAdapter, EstimationResult
from ..math_utils import winsorize, cluster_sandwich_mle

class GLMLogLinkAdapter(BaseAdapter):
    family = "GLM"

    def _build_X(self, d: pd.DataFrame, xcols: List[str], unit_col: str, time_col: str, uFE: int, tFE: int):
        X_list = [np.ones(len(d), dtype=float)]
        names = ["const"]
        for c in xcols:
            X_list.append(d[c].to_numpy(dtype=float))
            names.append(c)
        if uFE == 1:
            du = pd.get_dummies(d[unit_col].astype(int), prefix="u", drop_first=True)
            X_list.append(du.to_numpy(dtype=float))
            names.extend(list(du.columns))
        if tFE == 1:
            dt = pd.get_dummies(d[time_col].astype(int), prefix="t", drop_first=True)
            X_list.append(dt.to_numpy(dtype=float))
            names.extend(list(dt.columns))
        return np.column_stack(X_list).astype(float), names

    def _irls_poisson(self, y: np.ndarray, X: np.ndarray, max_iter: int=80, tol: float=1e-8):
        n, k = X.shape
        beta = np.zeros(k, dtype=float)
        for _ in range(max_iter):
            eta = X @ beta
            mu = np.exp(np.clip(eta, -30, 30))
            W = mu
            z = eta + (y - mu) / np.maximum(mu, 1e-12)
            Xw = X * np.sqrt(W)[:,None]
            zw = z * np.sqrt(W)
            XtX = Xw.T @ Xw
            try:
                beta_new = np.linalg.solve(XtX, Xw.T @ zw)
            except np.linalg.LinAlgError:
                beta_new = np.linalg.pinv(XtX) @ (Xw.T @ zw)
            if np.max(np.abs(beta_new - beta)) < tol:
                beta = beta_new
                break
            beta = beta_new
        mu = np.exp(np.clip(X @ beta, -30, 30))
        return beta, mu

    def _irls_gamma_log(self, y: np.ndarray, X: np.ndarray, max_iter: int=80, tol: float=1e-8):
        n, k = X.shape
        beta = np.zeros(k, dtype=float)
        y = np.maximum(y, 1e-12)
        for _ in range(max_iter):
            eta = X @ beta
            mu = np.exp(np.clip(eta, -30, 30))
            W = np.ones_like(mu)
            z = eta + (y - mu) / np.maximum(mu, 1e-12)
            Xw = X * np.sqrt(W)[:,None]
            zw = z * np.sqrt(W)
            XtX = Xw.T @ Xw
            try:
                beta_new = np.linalg.solve(XtX, Xw.T @ zw)
            except np.linalg.LinAlgError:
                beta_new = np.linalg.pinv(XtX) @ (Xw.T @ zw)
            if np.max(np.abs(beta_new - beta)) < tol:
                beta = beta_new
                break
            beta = beta_new
        mu = np.exp(np.clip(X @ beta, -30, 30))
        return beta, mu

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
        fam = str(e.get("family","poisson")).lower()

        for c in [ycol, unit_col, time_col] + xcols:
            if c not in df.columns:
                return EstimationResult(False, np.array([]), np.array([]), msg=f"missing column: {c}")

        d = df.copy()
        drop_k = int(s.get("drop_last_k", 0))
        if drop_k > 0:
            tmax = int(d[time_col].max())
            d = d[d[time_col] <= (tmax - drop_k)].copy()

        y_raw = d[ycol].to_numpy(dtype=float)
        y = winsorize(y_raw, p.get("winsor_p", None))

        y_zero = str(p.get("y_zero","keep")).lower()
        if fam == "poisson":
            if np.any(y < 0):
                return EstimationResult(False, np.array([]), np.array([]), msg="poisson requires y>=0")
        if fam == "gamma":
            if y_zero == "drop":
                keep = y > 0
                d = d.loc[keep].copy()
                y = y[keep]
            elif y_zero == "add_eps":
                y = np.maximum(y, 1e-8)
            else:
                if np.any(y <= 0):
                    return EstimationResult(False, np.array([]), np.array([]), msg="gamma requires y>0 (set y_zero=drop|add_eps)")

        X, names = self._build_X(d, xcols, unit_col, time_col, uFE, tFE)
        cluster = str(i.get("cluster","unit")).lower()
        if cluster == "unit":
            cl = d[unit_col].astype(int).to_numpy()
        elif cluster == "time":
            cl = d[time_col].astype(int).to_numpy()
        else:
            cl = np.arange(len(d), dtype=int)

        try:
            if fam == "poisson":
                beta, mu = self._irls_poisson(y, X)
                hess = X.T @ (X * mu[:,None])
                score = X * (y - mu)[:,None]
                V, se = cluster_sandwich_mle(score, hess, cl)
            elif fam == "gamma":
                beta, mu = self._irls_gamma_log(y, X)
                resid = (y - mu) / np.maximum(mu, 1e-12)
                df_res = max(len(y) - X.shape[1], 1)
                phi = float(np.sum(resid*resid) / df_res)
                hess = (X.T @ X) / max(phi, 1e-12)
                score = X * ((y - mu) / (max(phi,1e-12) * np.maximum(mu,1e-12)))[:,None]
                V, se = cluster_sandwich_mle(score, hess, cl)
            else:
                return EstimationResult(False, np.array([]), np.array([]), msg=f"unknown family={fam}")
        except Exception as ex:
            return EstimationResult(False, np.array([]), np.array([]), msg=f"GLM failed: {ex}")

        G = len(np.unique(cl))
        dof = max(G-1, 1)
        tstat = beta / np.maximum(se, 1e-12)
        pval = 2*(1 - stats.t.cdf(np.abs(tstat), df=dof))

        psi_indices = i.get("psi_indices", None)
        if psi_indices is None:
            psi_indices = [names.index(xcols[0])] if xcols else [0]
        psi_indices = [int(j) for j in psi_indices]

        psi_hat = beta[psi_indices].astype(float)
        U = V[np.ix_(psi_indices, psi_indices)].astype(float)
        p_out = pval[psi_indices].astype(float)
        Q = {"names": names, "family": fam, "uFE": uFE, "tFE": tFE}
        return EstimationResult(True, psi_hat, U, p=p_out, Q=Q, n=int(len(y)))
