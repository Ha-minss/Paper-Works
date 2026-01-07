"""Assumption-indexed stability (δ-knob).

δ is *not* an estimator change. It is a filter over the design set:

  D(δ) = { d : A_value(d) <= δ }

where A_value is a family-specific assumption-violation metric extracted from each design's diagnostics Q.

This module keeps the interface minimal and universal so any adapter can opt-in by adding a single metric to its Q.

For the demo schema (DID), we use Q["pretrend_max_abs_t"] if available; otherwise we fall back to NaN.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .softmax import stage1_score


@dataclass(frozen=True)
class AssumptionMetric:
    A_name: str
    direction: str = "leq"  # currently only support "leq" (smaller is better)
    extractor: Optional[Callable[[str, Dict[str, Any], Dict[str, Any]], float]] = None

    def extract(self, family: str, Q: Dict[str, Any], spec: Dict[str, Any]) -> float:
        if self.extractor is None:
            return float("nan")
        try:
            v = self.extractor(str(family), Q or {}, spec or {})
            return float(v)
        except Exception:
            return float("nan")


def _extract_default(family: str, Q: Dict[str, Any], spec: Dict[str, Any]) -> float:
    fam = str(family).strip().lower()

    # Prefer explicit Q metrics.
    if fam in {"did", "event_study", "es"}:
        v = Q.get("pretrend_max_abs_t", np.nan)
        try:
            return float(v)
        except Exception:
            return float("nan")

    if fam in {"rd"}:
        v = Q.get("density_p", np.nan)
        # For RD, higher p-value is better; but our convention is "smaller is better".
        # Use 1-p so that smaller means "less evidence of manipulation".
        try:
            return float(1.0 - float(v))
        except Exception:
            return float("nan")

    if fam in {"panel_iv_2sls", "iv", "iv2sls"}:
        v = Q.get("first_stage_F", np.nan)
        # Larger F is better -> invert.
        try:
            fv = float(v)
            return float(1.0 / max(fv, 1e-12))
        except Exception:
            return float("nan")

    # fallback: missing
    return float("nan")


DEFAULT_METRIC = AssumptionMetric(A_name="A", direction="leq", extractor=_extract_default)


def delta_grid_for_family(family: str) -> List[float]:
    fam = str(family).strip().lower()
    if fam in {"did", "event_study", "es"}:
        # t-stat max abs pretrend. Typical thresholds: 1, 1.64, 1.96, 2.58
        return [0.0, 0.5, 1.0, 1.64, 1.96, 2.58, 3.0]
    if fam in {"rd"}:
        # 1-p: 0..1
        return [0.0, 0.01, 0.05, 0.10, 0.20, 0.50, 1.0]
    if fam in {"panel_iv_2sls", "iv", "iv2sls"}:
        # 1/F: smaller is better. Use coarse grid.
        return [0.0, 0.05, 0.10, 0.20, 0.50, 1.0]
    return [0.0, 1.0]


# Backward/runner compatibility: earlier drafts used this name.
def default_delta_grid_for_family(family: str) -> List[float]:
    return delta_grid_for_family(family)


def attach_A_columns(design_table: pd.DataFrame, *, metric: AssumptionMetric = DEFAULT_METRIC) -> pd.DataFrame:
    out = design_table.copy()

    A_name = metric.A_name
    A_vals: List[float] = []

    for _, r in out.iterrows():
        family = str(r.get("family", r.get("m", ""))).strip().lower()
        Q = r.get("Q", {}) or {}
        spec = r.get("spec", {}) or {}
        A_vals.append(metric.extract(family, Q, spec))

    out["A_name"] = A_name
    out["A_value"] = np.asarray(A_vals, dtype=float)
    return out


def filter_by_delta(design_table: pd.DataFrame, delta: float) -> pd.DataFrame:
    if "A_value" not in design_table.columns:
        return design_table.iloc[0:0].copy()

    v = np.asarray(design_table["A_value"].to_numpy(dtype=float), dtype=float)
    okA = np.isfinite(v) & (v <= float(delta))
    return design_table.loc[okA].copy()


def compute_delta_curve(
    design_table: pd.DataFrame,
    g: np.ndarray,
    *,
    K_grid: List[int],
    delta_grid: List[float],
    family: str,
    tier: str,
    A_name: str,
) -> pd.DataFrame:
    """Compute S_wc(K;δ) for a fixed tier.

    Parameters
    ----------
    design_table : DataFrame
        Must include A_value.
    g : ndarray
        g-scores aligned with rows of design_table.
    """

    g = np.asarray(g, dtype=float).reshape(-1)
    if len(g) != len(design_table):
        raise ValueError("g must align with design_table rows")

    rows = []
    A_vals = np.asarray(design_table.get("A_value", np.full(len(design_table), np.nan)), dtype=float)
    ok_design = np.asarray(design_table.get("ok", np.ones(len(design_table), dtype=bool)), dtype=bool)

    for delta in delta_grid:
        sel = ok_design & np.isfinite(g) & np.isfinite(A_vals) & (A_vals <= float(delta))
        g_sub = g[sel]
        n_sub = int(g_sub.size)
        max_g = float(np.nanmax(g_sub)) if n_sub > 0 else float("nan")

        for K in K_grid:
            if n_sub <= 0:
                S = float("nan")
            else:
                K_eff = int(min(max(1, int(K)), n_sub))
                S = float(stage1_score(g_sub, K_eff)["S_wc"])
            rows.append(
                {
                    "family": str(family),
                    "tier": str(tier),
                    "A_name": str(A_name),
                    "delta": float(delta),
                    "K": int(K),
                    "S_wc": float(S),
                    "max_g": float(max_g),
                    "n_designs": int(n_sub),
                }
            )

    return pd.DataFrame(rows)


def plot_assumption_curve_to(path: str, curve_df: pd.DataFrame, *, K: int):
    """Save a simple δ-curve plot at fixed K."""
    import matplotlib.pyplot as plt

    df = curve_df[curve_df["K"] == int(K)].copy()
    df = df.sort_values("delta")

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    if len(df) > 0:
        ax.plot(df["delta"].values, df["S_wc"].values, marker="o")
    ax.set_xlabel("delta")
    ax.set_ylabel("S_wc")
    ax.set_title(f"Assumption curve (K={K})")
    ax.grid(True, alpha=0.25)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
