"""Contract validators and sanitizers.

The project is intentionally adapter-agnostic: adapters may fail on some designs.
A core stability requirement is that failures must *not* crash the full pipeline.

This module centralizes "fail-soft" validation for:
- EstimationResult objects returned by adapters
- the design_table DataFrame produced by the engine

These utilities are used by the runner to enforce a consistent schema early,
preventing brittle KeyError/TypeError failures downstream.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import json
import numpy as np
import pandas as pd


REQUIRED_DESIGN_TABLE_COLS = [
    "family",
    "tier",
    "design_id",
    "ok",
    "psi_hat",
    "U",
    "p",
    "Q",
    "Q_json",
    "msg",
    "spec",
]


def _as_1d_float_array(x: Any, k_fallback: int = 1) -> np.ndarray:
    try:
        arr = np.asarray(x, dtype=float).reshape(-1)
        if arr.size == 0:
            return np.full(int(k_fallback), np.nan, dtype=float)
        return arr
    except Exception:
        return np.full(int(k_fallback), np.nan, dtype=float)


def _as_cov(x: Any, k: int) -> np.ndarray:
    """Coerce to a (k,k) covariance matrix; if impossible, return NaNs."""
    try:
        arr = np.asarray(x, dtype=float)
        if arr.ndim == 0:
            se = float(arr)
            return np.eye(k, dtype=float) * (se * se)
        if arr.ndim == 1:
            se = arr.reshape(-1)
            if se.size != k:
                se = np.full(k, np.nan, dtype=float)
            return np.diag(se * se)
        if arr.ndim == 2:
            if arr.shape != (k, k):
                return np.full((k, k), np.nan, dtype=float)
            return arr
    except Exception:
        pass
    return np.full((k, k), np.nan, dtype=float)


def validate_estimation_like(res: Any) -> Dict[str, Any]:
    """Validate/coerce an adapter result-like object.

    We accept either:
      - stabilitycheck.adapters.base.EstimationResult
      - or a dict with similar keys.

    Returns a normalized dict with keys:
      ok, psi_hat, U, p, Q, msg
    """

    # Pull fields permissively
    ok = False
    psi_hat = None
    U = None
    p = None
    Q: Dict[str, Any] = {}
    msg = ""

    if isinstance(res, dict):
        ok = bool(res.get("ok", False))
        psi_hat = res.get("psi_hat", res.get("theta_hat", None))
        U = res.get("U", res.get("cov", None))
        p = res.get("p", None)
        Q = dict(res.get("Q", {})) if res.get("Q", {}) is not None else {}
        msg = str(res.get("msg", ""))
    else:
        ok = bool(getattr(res, "ok", False))
        psi_hat = getattr(res, "psi_hat", None)
        U = getattr(res, "U", None)
        p = getattr(res, "p", None)
        Q = dict(getattr(res, "Q", {}) or {})
        msg = str(getattr(res, "msg", ""))

    psi_hat_arr = _as_1d_float_array(psi_hat, k_fallback=1)
    k = int(psi_hat_arr.size) if psi_hat_arr.size > 0 else 1
    U_arr = _as_cov(U, k=k)

    # p allowed None or 1d
    if p is None:
        p_arr = None
    else:
        p_arr = _as_1d_float_array(p, k_fallback=k)

    # If coercion produced all-NaN psi_hat, treat as failure
    if not np.isfinite(psi_hat_arr).any():
        ok = False

    return {
        "ok": bool(ok),
        "psi_hat": psi_hat_arr,
        "U": U_arr,
        "p": p_arr,
        "Q": Q,
        "msg": msg,
    }


def validate_design_table(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the design_table schema exists and is safe to save as CSV."""
    if df is None or not isinstance(df, pd.DataFrame):
        df = pd.DataFrame()

    out = df.copy()

    # Ensure required columns exist
    for c in REQUIRED_DESIGN_TABLE_COLS:
        if c not in out.columns:
            if c == "ok":
                out[c] = False
            elif c in {"Q", "spec"}:
                out[c] = [{} for _ in range(len(out))]
            else:
                out[c] = ""

    # Standardize psi_hat and U types
    psi_list = []
    U_list = []
    p_list = []
    Q_list = []
    msg_list = []
    ok_list = []

    for _, r in out.iterrows():
        norm = validate_estimation_like(
            {
                "ok": r.get("ok", False),
                "psi_hat": r.get("psi_hat", None),
                "U": r.get("U", None),
                "p": r.get("p", None),
                "Q": r.get("Q", {}),
                "msg": r.get("msg", ""),
            }
        )
        ok_list.append(bool(norm["ok"]))
        psi_list.append(norm["psi_hat"])
        U_list.append(norm["U"])
        p_list.append(norm["p"])
        Q_list.append(norm["Q"])
        msg_list.append(str(norm.get("msg", "")))

    out["ok"] = ok_list
    out["psi_hat"] = psi_list
    out["U"] = U_list
    out["p"] = p_list
    out["Q"] = Q_list
    out["msg"] = msg_list

    # JSON-safe Q
    def _to_json(q: Any) -> str:
        try:
            return json.dumps(q if q is not None else {}, ensure_ascii=False, default=str)
        except Exception:
            return "{}"

    out["Q_json"] = [_to_json(q) for q in out["Q"].tolist()]

    # Enforce psi_hat name (ban legacy 'psi') by removing if present
    if "psi" in out.columns:
        out = out.drop(columns=["psi"])

    return out
