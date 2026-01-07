from __future__ import annotations
import numpy as np

def critical_value(S_null: np.ndarray, alpha: float) -> float:
    return float(np.quantile(S_null, 1 - float(alpha)))

def mc_pvalue(S_null: np.ndarray, S_obs: float) -> float:
    return float((np.sum(S_null >= S_obs) + 1) / (len(S_null) + 1))
