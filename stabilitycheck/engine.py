from __future__ import annotations

"""Core engine computations.

Engine responsibilities:
- compute a design table by running adapters over design specs
- compute anchors
- compute g-scores for designs relative to anchors

Runner responsibilities:
- orchestration, looping over tiers, saving outputs, plots, testification

A core stability requirement is fail-soft behavior: individual design failures
must be absorbed as rows with ok=False rather than crashing the pipeline.
"""

import math
import traceback
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .contract import validate_design_table, validate_estimation_like
from .configs import EngineConfig
from .schema import make_designs_for_tier, spec_to_features
from .softmax import stage1_score
import stabilitycheck.g as gmod


def _tb1(ex: BaseException) -> str:
    s = "".join(traceback.format_exception_only(type(ex), ex)).strip()
    return s.splitlines()[0] if s else repr(ex)


def compute_design_table(
    cfg: EngineConfig,
    schema: Any,
    adapter_registry: Any,
    *,
    df: pd.DataFrame,
    tier_name: str,
    designs: Optional[List[dict]] = None,
) -> pd.DataFrame:
    """Run all (or a subset of) designs for a tier and return a design_table."""

    if designs is None:
        # schema may be a dataclass (configs.SchemaConfig) rather than an object
        # exposing methods. Use the helper to keep the public runner API stable.
        designs = make_designs_for_tier(schema, tier_name)

    rows: List[Dict[str, Any]] = []

    for j, spec in enumerate(designs):
        family = str((spec or {}).get("m", "")).strip().lower()
        design_id = f"{tier_name}_{j:04d}"

        try:
            adapter = adapter_registry.make(family)
            est = adapter.fit(df, spec)
            norm = validate_estimation_like(est)
        except Exception as ex:
            # Fail-soft: keep a row and continue
            norm = validate_estimation_like({"ok": False, "psi_hat": None, "U": None, "p": None, "Q": {}, "msg": _tb1(ex)})

        # Attach features (used by stage2/visualization; safe if schema doesn't know)
        try:
            features = spec_to_features(spec)
        except Exception:
            features = {}

        row = {
            "family": family,
            "tier": str(tier_name),
            "design_id": design_id,
            "ok": bool(norm["ok"]),
            "psi_hat": norm["psi_hat"],
            "U": norm["U"],
            "p": norm["p"],
            "Q": norm["Q"],
            "msg": str(norm.get("msg", "")),
            "spec": spec,
            "features": features,
        }
        rows.append(row)

    return validate_design_table(pd.DataFrame(rows))


def compute_anchors(
    cfg: EngineConfig,
    *,
    df: pd.DataFrame,
    adapter_registry: Any,
    schema: Any | None = None,
    tier_name: str | None = None,
) -> Dict[str, Dict[str, Any]]:
    """Compute anchors, always returning a dict with keys main/cons/alt.

    If anchors are not provided in cfg, and schema+tier_name are available,
    we pick the first few designs from that tier as fallbacks.
    """

    def _fallback_design(i: int) -> dict | None:
        if schema is None or tier_name is None:
            return None
        try:
            ds = make_designs_for_tier(schema, tier_name)
            return ds[i] if i < len(ds) else None
        except Exception:
            return None

    anchor_specs = {
        "main": cfg.anchor_main or _fallback_design(0),
        "cons": cfg.anchor_cons or _fallback_design(1),
        "alt": cfg.anchor_alt or _fallback_design(2),
    }

    out: Dict[str, Dict[str, Any]] = {}
    for name, spec in anchor_specs.items():
        if spec is None:
            # Still return a placeholder
            out[name] = {"ok": False, "psi_hat": np.full(1, np.nan), "U": np.full((1, 1), np.nan), "p": None, "Q": {}, "spec": None, "msg": "anchor missing"}
            continue
        family = str((spec or {}).get("m", "")).strip().lower()
        try:
            adapter = adapter_registry.make(family)
            est = adapter.fit(df, spec)
            norm = validate_estimation_like(est)
            out[name] = {**norm, "spec": spec}
        except Exception as ex:
            norm = validate_estimation_like({"ok": False, "psi_hat": None, "U": None, "p": None, "Q": {}, "msg": _tb1(ex)})
            out[name] = {**norm, "spec": spec}

    return out


def make_k_grid(cfg: EngineConfig, D_size: int) -> List[int]:
    D_size = int(D_size)
    if D_size <= 0:
        return [1]
    K_rule = min(D_size, max(cfg.k_min, int(math.ceil(cfg.rho * D_size))))
    K_05 = min(D_size, max(cfg.k_min, int(math.ceil(0.05 * D_size))))
    K_10 = min(D_size, max(cfg.k_min, int(math.ceil(0.10 * D_size))))
    base = [k for k in cfg.k_grid_base if k <= D_size]
    grid = sorted(set(base + [1, K_rule, K_05, K_10]))
    return [k for k in grid if 1 <= k <= D_size]


def compute_g_scalar(
    cfg: EngineConfig,
    D: pd.DataFrame,
    anchors: Dict[str, Dict[str, Any]],
    *,
    pnorm: int = 2,
    w2: float = 1.0,
    w3: float = 1.0,
) -> np.ndarray:
    """Compute scalar g(d) for each design row.

    If a design failed (ok=False) or anchors are missing, returns NaN for that row.
    """

    if D is None or len(D) == 0:
        return np.asarray([], dtype=float)

    # anchor list: only use those with finite psi_hat
    anchor_list = []
    for a in (anchors or {}).values():
        if a is None:
            continue
        psi_a = np.asarray(a.get("psi_hat", np.nan), dtype=float).reshape(-1)
        U_a = np.asarray(a.get("U", np.nan), dtype=float)
        if psi_a.size > 0 and np.isfinite(psi_a).any() and np.isfinite(U_a).any():
            anchor_list.append({"psi_hat": psi_a, "U": U_a, "p": a.get("p", None)})

    g_list: List[float] = []

    for _, r in D.iterrows():
        if not bool(r.get("ok", True)):
            g_list.append(float("nan"))
            continue
        psi_d = np.asarray(r.get("psi_hat", np.nan), dtype=float).reshape(-1)
        U_d = np.asarray(r.get("U", np.nan), dtype=float)
        p_d = r.get("p", None)

        if psi_d.size == 0 or not np.isfinite(psi_d).any() or not np.isfinite(U_d).any() or len(anchor_list) == 0:
            g_list.append(float("nan"))
            continue

        spec = r.get("spec", {}) or {}
        try:
            sign_index = int((spec.get("i", {}) or {}).get("psi_index", 0))
        except Exception:
            sign_index = 0

        Gs = []
        for a in anchor_list:
            Gs.append(
                gmod.g_vector(
                    psi_d=psi_d,
                    p_d=p_d,
                    psi_a=a["psi_hat"],
                    U_a=a["U"],
                    alpha=cfg.alpha_g,
                    sign_index=sign_index,
                )
            )
        g_agg = gmod.aggregate_anchors(Gs, cfg.anchor_agg)
        g_list.append(float(gmod.scalarize(g_agg, pnorm, w2, w3)))

    return np.asarray(g_list, dtype=float)


def stage1_diagnostics(cfg: EngineConfig, g: np.ndarray) -> Dict[str, Any]:
    g = np.asarray(g, dtype=float).reshape(-1)
    g = g[np.isfinite(g)]
    if g.size < 1:
        return {"K_rule": 1, "S_wc": float("nan")}
    K_rule = min(int(g.size), max(cfg.k_min, int(math.ceil(cfg.rho * int(g.size)))))
    return {"K_rule": int(K_rule), **stage1_score(g, K_rule)}
