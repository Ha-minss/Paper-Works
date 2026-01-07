from __future__ import annotations

"""Large-|D| sampling utilities.

This module provides simple, reproducible approximations when enumerating
all designs is infeasible. It is intentionally light-weight (NumPy only).

Functions here are designed to be used by `stabilitycheck.runner`:
  - `space_filling_sample`: initial broad coverage sample
  - `refine_high_risk`: add samples near high-risk seeds
  - `spike_sweep`: coordinate-wise sweep to catch discontinuities
  - `coverage_summary`: kNN-like coverage diagnostics
  - `largeD_tighten`: one-shot wrapper used by the DoD runner
"""

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np


def _stable_hash01(s: str) -> float:
    """Deterministic hash -> [0,1) float (stable across runs)."""
    import hashlib

    h = hashlib.md5(s.encode("utf-8"), usedforsecurity=False).hexdigest()
    return (int(h[:8], 16) % 10_000_000) / 10_000_000.0


def _flatten_design(design: dict) -> Dict[str, float]:
    """Flatten a nested design spec dict into numeric features."""

    out: Dict[str, float] = {}

    def rec(prefix: str, obj):
        if isinstance(obj, dict):
            for k in sorted(obj.keys()):
                rec(f"{prefix}.{k}" if prefix else str(k), obj[k])
            return
        if obj is None:
            out[prefix] = 0.0
            return
        if isinstance(obj, bool):
            out[prefix] = float(int(obj))
            return
        if isinstance(obj, (int, float)):
            out[prefix] = float(obj)
            return
        # Strings / others -> stable hash
        out[prefix] = float(_stable_hash01(str(obj)))

    rec("", design)
    return out


def largeD_tighten(
    designs_all: List[dict],
    *,
    seed: int = 0,
    method: str = "lhs",
    M0: int = 256,
    M1: int = 128,
    top_frac: float = 0.10,
    M_sweep: int = 128,
) -> Dict[str, object]:
    """One-shot large-|D| tightening wrapper.

    This is a light-weight approximation for the paper pipeline.
    It produces:
      - a sampled index set (initial + refinement + sweep)
      - a simple coverage summary in a numeric feature space
      - a CSV-friendly table describing what was sampled
    """

    n = len(designs_all)
    rng = np.random.default_rng(int(seed))

    if n <= 0:
        import pandas as pd

        return {
            "sampling_plan": {"method": method, "M0": 0, "M1": 0, "M_sweep": 0, "n_all": 0},
            "tightening_df": pd.DataFrame(),
            "coverage": {},
        }

    # Build a numeric feature matrix (deterministic) for coverage/nearest-neighbor ops.
    feats0 = _flatten_design(designs_all[0])
    keys = sorted(feats0.keys())
    X = np.empty((n, len(keys)), dtype=float)
    for i, d in enumerate(designs_all):
        fi = _flatten_design(d)
        X[i, :] = np.array([fi.get(k, 0.0) for k in keys], dtype=float)

    chosen0 = space_filling_sample(designs_all, M0=int(M0), method=str(method), seed=int(seed))

    # Dummy risk scores: a reproducible noisy linear combination of features.
    w = rng.normal(size=X.shape[1])
    risk_all = X @ w + 0.05 * rng.normal(size=n)

    add1 = refine_high_risk(X, risk_all, chosen0, top_frac=float(top_frac), M1=int(M1), seed=int(seed) + 11)
    chosen1 = sorted(set(chosen0) | set(add1))

    add2 = spike_sweep(designs_all, chosen1, M_sweep=int(M_sweep), seed=int(seed) + 23)
    chosen = sorted(set(chosen1) | set(add2))

    cov = coverage_summary(X, chosen)

    import pandas as pd

    rows = []
    for i in chosen0:
        rows.append({"idx": int(i), "stage": "M0", "risk": float(risk_all[i])})
    for i in add1:
        rows.append({"idx": int(i), "stage": "M1_refine", "risk": float(risk_all[i])})
    for i in add2:
        rows.append({"idx": int(i), "stage": "sweep", "risk": float(risk_all[i])})
    tightening_df = pd.DataFrame(rows).sort_values(["stage", "risk"], ascending=[True, False]).reset_index(drop=True)

    return {
        "sampling_plan": {
            "method": str(method),
            "M0": int(min(M0, n)),
            "M1": int(min(M1, n)),
            "top_frac": float(top_frac),
            "M_sweep": int(min(M_sweep, n)),
            "n_all": int(n),
            "n_chosen": int(len(chosen)),
        },
        "tightening_df": tightening_df,
        "coverage": cov,
    }


def space_filling_sample(
    designs_all: List[dict],
    M0: int,
    method: str = "lhs",
    seed: int = 0,
) -> List[int]:
    """Pick M0 indices from designs_all in a reproducible, space-filling manner.

    Current implementation:
      - If the design list is not huge, simple stratified random sampling.
      - `method` is kept for API stability; currently supports "lhs" and "random".
    """
    n = len(designs_all)
    M0 = int(max(1, min(M0, n)))
    rng = np.random.default_rng(seed)
    if method not in {"lhs", "random"}:
        method = "lhs"
    # Stratified sample across list index (acts like a crude space-filling proxy)
    if method == "lhs" and M0 < n:
        edges = np.linspace(0, n, M0 + 1, dtype=int)
        idx = []
        for i in range(M0):
            lo, hi = edges[i], max(edges[i + 1], edges[i] + 1)
            idx.append(int(rng.integers(lo, hi)))
        return sorted(set(idx))
    # Fallback
    return sorted(rng.choice(n, size=M0, replace=False).tolist())


def _pairwise_min_dist(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """For each row in X, compute min Euclidean distance to any row in Y."""
    if len(Y) == 0:
        return np.full(len(X), np.inf, dtype=float)
    # O(n*m) brute force; acceptable for demo-sized problems.
    dmins = np.empty(len(X), dtype=float)
    for i, x in enumerate(X):
        diff = Y - x
        d2 = np.sum(diff * diff, axis=1)
        dmins[i] = float(np.sqrt(np.min(d2)))
    return dmins


def coverage_summary(
    X_all: np.ndarray,
    chosen_idx: Sequence[int],
    quantiles: Tuple[float, float] = (0.5, 0.9),
) -> Dict[str, float]:
    """Compute a simple coverage diagnostic.

    We measure how well the chosen sample covers the (candidate) design feature space.
    For each point in X_all, compute distance to nearest sampled point.
    Return requested quantiles of that distance distribution.
    """
    chosen_idx = list(dict.fromkeys(int(i) for i in chosen_idx))
    Xs = X_all[chosen_idx] if len(chosen_idx) else np.empty((0, X_all.shape[1]))
    dmin = _pairwise_min_dist(X_all, Xs)
    q50, q90 = quantiles
    return {
        "coverage_p50": float(np.quantile(dmin, q50)),
        "coverage_p90": float(np.quantile(dmin, q90)),
        "coverage_max": float(np.max(dmin)),
        "n_all": int(X_all.shape[0]),
        "n_sample": int(len(chosen_idx)),
    }


def refine_high_risk(
    X_all: np.ndarray,
    risk_scores: np.ndarray,
    chosen_idx: Sequence[int],
    top_frac: float,
    M1: int,
    seed: int = 0,
) -> List[int]:
    """Add M1 indices near high-risk seeds.

    Strategy:
      1) Select seed set = top `top_frac` by `risk_scores` among chosen_idx.
      2) For each seed, pull nearest neighbors (in feature space) not yet chosen.
      3) Return newly added indices (unique).
    """
    n = X_all.shape[0]
    M1 = int(max(0, min(M1, n)))
    if M1 == 0:
        return []

    chosen = set(int(i) for i in chosen_idx)
    rng = np.random.default_rng(seed)

    chosen_list = np.array(sorted(chosen), dtype=int)
    if len(chosen_list) == 0:
        return []

    rs = risk_scores[chosen_list]
    k_seed = int(max(1, np.ceil(top_frac * len(chosen_list))))
    seed_idx = chosen_list[np.argsort(-rs)[:k_seed]]

    # Candidate pool
    pool = np.array([i for i in range(n) if i not in chosen], dtype=int)
    if len(pool) == 0:
        return []

    # Compute distance from pool to each seed and pick closest.
    # For speed, we random-subset if pool is very large.
    if len(pool) > 5000:
        pool = rng.choice(pool, size=5000, replace=False)

    X_pool = X_all[pool]
    add: List[int] = []
    # Round-robin neighbors per seed
    per_seed = int(max(1, np.ceil(M1 / len(seed_idx))))
    for s in seed_idx:
        x = X_all[s]
        diff = X_pool - x
        d2 = np.sum(diff * diff, axis=1)
        nn = pool[np.argsort(d2)[:per_seed]]
        for j in nn:
            if j not in chosen:
                add.append(int(j))
                chosen.add(int(j))
                if len(add) >= M1:
                    return add
    return add[:M1]


def spike_sweep(
    designs_all: List[dict],
    chosen_idx: Sequence[int],
    M_sweep: int,
    seed: int = 0,
) -> List[int]:
    """Coordinate-wise sweep to catch spikes.

    A very simple implementation:
      - Start from a few chosen indices (highest-risk),
      - For each, randomly perturb a single knob by selecting a nearby design
        in the list index space.

    Returns additional indices to evaluate.
    """
    n = len(designs_all)
    M_sweep = int(max(0, min(M_sweep, n)))
    if M_sweep == 0:
        return []

    rng = np.random.default_rng(seed)
    chosen = set(int(i) for i in chosen_idx)
    base = np.array(sorted(chosen), dtype=int)
    if len(base) == 0:
        base = np.array([int(rng.integers(0, n))], dtype=int)

    add: List[int] = []
    while len(add) < M_sweep:
        i = int(rng.choice(base))
        # "nearby" in index space; cheap proxy for a local move in knob space
        step = int(rng.integers(-25, 26))
        j = int(np.clip(i + step, 0, n - 1))
        if j in chosen:
            continue
        add.append(j)
        chosen.add(j)
    return add


# -----------------------------------------------------------------------------
# Runner wrapper
# -----------------------------------------------------------------------------

def _stable_str_to_float(s: str) -> float:
    """Deterministically map a string to a float in [0, 1)."""
    import hashlib

    h = hashlib.md5(s.encode("utf-8")).hexdigest()[:8]
    return int(h, 16) / float(16**8)


def _flatten_design(des: dict, prefix: str = "") -> Dict[str, object]:
    out: Dict[str, object] = {}
    for k, v in (des or {}).items():
        key = f"{prefix}{k}" if prefix == "" else f"{prefix}.{k}"
        if isinstance(v, dict):
            out.update(_flatten_design(v, prefix=key))
        else:
            out[key] = v
    return out


def largeD_tighten(
    designs_all: List[dict],
    *,
    seed: int = 0,
    method: str = "lhs",
    M0: int = 256,
    M1: int = 128,
    top_frac: float = 0.10,
    M_sweep: int = 128,
) -> Dict[str, object]:
    """One-shot large-|D| tightening plan.

    This is a light-weight wrapper that produces:
      - a reproducible sample plan (initial + refinement + spike sweep)
      - basic coverage diagnostics
      - a CSV-ready table of selected indices

    It intentionally does **not** evaluate the model; that happens elsewhere.
    """
    import pandas as pd

    n = int(len(designs_all))
    if n <= 0:
        return {
            "sampling_plan": {"note": "empty design list", "n_all": 0},
            "tightening_df": pd.DataFrame(),
            "coverage": {},
        }

    rng = np.random.default_rng(int(seed))

    # Build a simple feature matrix from flattened spec (deterministic encoding).
    flat0 = _flatten_design(designs_all[0])
    keys = sorted(flat0.keys())
    X = np.zeros((n, len(keys)), dtype=float)
    for i, d in enumerate(designs_all):
        flat = _flatten_design(d)
        for j, k in enumerate(keys):
            v = flat.get(k, 0.0)
            if isinstance(v, (int, float, np.integer, np.floating)):
                X[i, j] = float(v)
            elif isinstance(v, bool):
                X[i, j] = float(int(v))
            elif v is None:
                X[i, j] = 0.0
            else:
                X[i, j] = _stable_str_to_float(str(v))

    # 1) Initial sample
    idx0 = space_filling_sample(designs_all, int(M0), method=str(method), seed=int(seed))

    # Dummy "risk" scores for refinement (runner may replace with true scores later).
    # Use a stable random draw so outputs are deterministic given seed.
    risk = rng.normal(size=n)

    # 2) Local refinement
    add1 = refine_high_risk(X, risk, idx0, top_frac=float(top_frac), M1=int(M1), seed=int(seed) + 1)

    # 3) Spike sweep
    idx01 = list(dict.fromkeys(list(idx0) + list(add1)))
    add2 = spike_sweep(designs_all, idx01, int(M_sweep), seed=int(seed) + 2)

    chosen = list(dict.fromkeys(list(idx0) + list(add1) + list(add2)))

    cov = coverage_summary(X, chosen)

    rows = []
    for i in idx0:
        rows.append({"idx": int(i), "stage": "initial", "risk_proxy": float(risk[int(i)])})
    for i in add1:
        rows.append({"idx": int(i), "stage": "refine", "risk_proxy": float(risk[int(i)])})
    for i in add2:
        rows.append({"idx": int(i), "stage": "spike_sweep", "risk_proxy": float(risk[int(i)])})

    df = pd.DataFrame(rows).drop_duplicates(subset=["idx"], keep="first")
    df = df.sort_values(["stage", "idx"]).reset_index(drop=True)

    return {
        "sampling_plan": {
            "method": str(method),
            "M0": int(M0),
            "M1": int(M1),
            "top_frac": float(top_frac),
            "M_sweep": int(M_sweep),
            "n_all": int(n),
            "n_chosen": int(len(chosen)),
        },
        "tightening_df": df,
        "coverage": cov,
    }
