from __future__ import annotations
import numpy as np
import pandas as pd

from .base import BaseAdapter, EstimationResult
from ..configs import SchemaConfig, DesignSpec
from ..math_utils import winsorize, twoway_demean, oneway_demean, fit_ols_cluster

class DIDAdapter(BaseAdapter):
    family = "DID"

    def __init__(self, schema: SchemaConfig):
        self.schema = schema

    def fit(self, df: pd.DataFrame, spec: DesignSpec) -> EstimationResult:
        if not isinstance(df, pd.DataFrame):
            return EstimationResult(False, np.array([]), np.array([]), msg="data must be DataFrame")

        req = ["unit","time","treat","post","y"]
        for c in req:
            if c not in df.columns:
                return EstimationResult(False, np.array([]), np.array([]), msg=f"missing column: {c}")

        p = spec.get("p", {}) or {}
        s = spec.get("s", {}) or {}
        e = spec.get("e", {}) or {}
        i = spec.get("i", {}) or {}

        transform = str(p.get("transform","log1p"))
        winsor_p = p.get("winsor_p", None)
        drop_k = int(s.get("drop_last_k", 0))
        uFE = int(e.get("uFE", 1))
        tFE = int(e.get("tFE", 1))
        ctrl_L = int(e.get("control_L", 0))

        d = df.copy()
        if drop_k > 0:
            tmax = int(d["time"].max())
            d = d[d["time"] <= (tmax - drop_k)].copy()
            if len(d) < 50:
                return EstimationResult(False, np.array([]), np.array([]), msg="too small after drop_last_k")

        y_raw = d["y"].to_numpy(dtype=float)
        y_w = winsorize(y_raw, winsor_p)

        if transform == "log1p":
            y = np.log1p(np.maximum(y_w, 0.0))
        elif transform == "level":
            y = y_w
        elif transform == "log":
            y = np.log(np.maximum(y_w, 1e-12))
        else:
            return EstimationResult(False, np.array([]), np.array([]), msg=f"unknown transform={transform}")

        treat = d["treat"].to_numpy(dtype=float)
        post  = d["post"].to_numpy(dtype=float)
        did   = treat * post

        Xcols = []
        names = []

        if uFE == 0:
            Xcols.append(treat); names.append("treat")
        if tFE == 0:
            Xcols.append(post); names.append("post")

        Xcols.append(did); names.append("did")

        if ctrl_L > 0:
            for c in self.schema.controls_ranked[:ctrl_L]:
                if c not in d.columns:
                    return EstimationResult(False, np.array([]), np.array([]), msg=f"missing control: {c}")
                Xcols.append(d[c].to_numpy(dtype=float)); names.append(c)

        X = np.column_stack(Xcols).astype(float)

        unit_code = d["unit"].astype(int).to_numpy()
        time_code = d["time"].astype(int).to_numpy()
        nU = int(df["unit"].max()) + 1
        nT = int(df["time"].max()) + 1

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

        cluster = str(i.get("cluster","unit")).lower()
        if cluster == "unit":
            cluster_ids = unit_code
        elif cluster == "time":
            cluster_ids = time_code
        else:
            cluster_ids = unit_code

        try:
            beta, se, pval, V = fit_ols_cluster(y_use, X_use, cluster_ids)
        except Exception as ex:
            return EstimationResult(False, np.array([]), np.array([]), msg=f"OLS failed: {ex}")

        if "did" not in names_use:
            return EstimationResult(False, np.array([]), np.array([]), msg="did not found")
        j = names_use.index("did")
        tau = float(beta[j]); se_tau = float(se[j]); p_tau = float(pval[j])

        psi_hat = np.array([tau], dtype=float)
        U = np.array([[se_tau*se_tau]], dtype=float)
        p_out = np.array([p_tau], dtype=float)
        Q = {"names": names_use, "transform": transform, "winsor_p": winsor_p, "uFE": uFE, "tFE": tFE, "control_L": ctrl_L, "pretrend_max_abs_t": 0.0}
        return EstimationResult(True, psi_hat, U, p=p_out, Q=Q, n=int(len(y_use)))
