from __future__ import annotations

import numpy as np
import pandas as pd

from .base import BaseAdapter, EstimationResult
from stabilitycheck.structural.ddc import estimate_binary_logit_ddc


class StructuralDDCAdapter(BaseAdapter):
    """Adapter for a simple binary logit dynamic discrete choice (DDC) model.

    The structural solver lives in stabilitycheck.structural.ddc.

    Spec convention:
      spec = {
        "m": "structural_ddc",
        "i": {"psi_index": 0},
        "e": {
           "id_col": "id",
           "time_col": "t",
           "state_col": "state",
           "choice_col": "choice",
           "beta": 0.9,
           "start": [0.5, -0.1],
           "max_iter": 500,
           "tol": 1e-10,
        }
      }
    """

    family = "structural_ddc"

    def fit(self, df: pd.DataFrame, spec: dict) -> EstimationResult:
        e = (spec or {}).get("e", {}) or {}

        id_col = str(e.get("id_col", "id"))
        time_col = str(e.get("time_col", "t"))
        state_col = str(e.get("state_col", "state"))
        choice_col = str(e.get("choice_col", "choice"))

        beta = float(e.get("beta", 0.95))
        start = e.get("start", (-1.0, 0.05))
        try:
            start_theta = tuple(float(x) for x in start)
        except Exception:
            start_theta = (-1.0, 0.05)

        max_iter = int(e.get("max_iter", 400))
        tol = float(e.get("tol", 1e-6))

        # Basic column checks (fail-soft through EstimationResult)
        for c in [id_col, time_col, state_col, choice_col]:
            if c not in df.columns:
                return EstimationResult(False, np.full(2, np.nan), np.full((2, 2), np.nan), msg=f"missing column: {c}")

        try:
            out = estimate_binary_logit_ddc(
                df,
                id_col=id_col,
                time_col=time_col,
                state_col=state_col,
                choice_col=choice_col,
                beta=beta,
                start_theta=start_theta,
                max_iter=max_iter,
                tol=tol,
            )
        except Exception as ex:
            return EstimationResult(False, np.full(2, np.nan), np.full((2, 2), np.nan), msg=str(ex))

        theta_hat = np.asarray(out.get("theta_hat", out.get("theta", np.full(2, np.nan))), dtype=float).reshape(-1)
        if theta_hat.size != 2:
            theta_hat = np.full(2, np.nan)

        cov = out.get("cov", None)
        if cov is None:
            U = np.full((2, 2), np.nan)
        else:
            U = np.asarray(cov, dtype=float)
            if U.shape != (2, 2):
                U = np.full((2, 2), np.nan)

        Q = {
            "converged": bool(out.get("ok", False)),
            "loglike": float(out.get("loglike", np.nan)),
            "discount": float(out.get("discount", beta)),
        }
        opt = out.get("opt", {}) or {}
        if "nit" in opt:
            Q["nit"] = int(opt.get("nit"))
        if "nfev" in opt:
            Q["nfev"] = int(opt.get("nfev"))

        return EstimationResult(bool(out.get("ok", False)), theta_hat, U, Q=Q)
