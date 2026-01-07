"""Stage1 testification helpers.

A "score" is not a test by itself. This module provides:
- critical value estimation from a null distribution
- Monte Carlo p-value
- PASS/FAIL decision

The design is deliberately adapter-agnostic:
You provide a `score_fn(df)->float` and a `null_generator(df,rng)->df`.

Also provides a small helper to build simple size/power tables using DGP templates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class Stage1TestResult:
    score_obs: float
    critical_value: float
    p_value_mc: float
    alpha: float
    decision: str
    B_used: int
    null_mode: str
    null_scores: np.ndarray


def monte_carlo_p_value(score_obs: float, null_scores: np.ndarray) -> float:
    """One-sided p-value: P(null >= obs). Uses +1 smoothing."""
    null_scores = np.asarray(null_scores, dtype=float)
    B = len(null_scores)
    return float((np.sum(null_scores >= score_obs) + 1.0) / (B + 1.0))


def critical_value(null_scores: np.ndarray, alpha: float) -> float:
    """Right-tail critical value at level alpha."""
    null_scores = np.asarray(null_scores, dtype=float)
    q = float(np.quantile(null_scores, 1.0 - alpha))
    return q


def run_stage1_testification(
    df: pd.DataFrame,
    score_fn: Callable[[pd.DataFrame], float],
    null_generator: Callable[[pd.DataFrame, np.random.Generator], pd.DataFrame],
    B: int = 200,
    alpha: float = 0.05,
    seed: int = 0,
    null_mode: str = "custom",
) -> Stage1TestResult:
    rng = np.random.default_rng(seed)

    score_obs = float(score_fn(df))
    null_scores = np.empty(int(B), dtype=float)
    for b in range(int(B)):
        df_b = null_generator(df, rng)
        null_scores[b] = float(score_fn(df_b))

    c = critical_value(null_scores, alpha=alpha)
    p = monte_carlo_p_value(score_obs, null_scores)
    decision = "PASS" if score_obs <= c else "FAIL"

    return Stage1TestResult(
        score_obs=score_obs,
        critical_value=c,
        p_value_mc=p,
        alpha=float(alpha),
        decision=decision,
        B_used=int(B),
        null_mode=str(null_mode),
        null_scores=null_scores,
    )


def make_test_report_row(
    tier: str,
    K: int,
    test: Stage1TestResult,
    Neff: float | None = None,
    lam: float | None = None,
) -> Dict[str, object]:
    """Build the fixed-schema CSV row for Stage1 testification.

    Required columns (DoD): tier, K, Neff, lambda, S_obs, critical_value, p_value_mc, alpha, B_used, decision
    """
    import math
    neff_v = float("nan") if Neff is None or (isinstance(Neff, float) and math.isnan(Neff)) else float(Neff)
    lam_v = float("nan") if lam is None or (isinstance(lam, float) and math.isnan(lam)) else float(lam)
    return {
        "tier": str(tier),
        "K": int(K),
        "Neff": neff_v,
        "lambda": lam_v,
        "S_obs": float(test.score_obs),
        "critical_value": float(test.critical_value),
        "p_value_mc": float(test.p_value_mc),
        "alpha": float(test.alpha),
        "B_used": int(test.B_used),
        "decision": str(test.decision),
        "null_mode": str(test.null_mode),
    }


def size_power_table(
    scenarios: Dict[str, Callable[[np.random.Generator], pd.DataFrame]],
    score_fn_factory: Callable[[pd.DataFrame], Callable[[pd.DataFrame], float]],
    null_generator_factory: Callable[[pd.DataFrame], Callable[[pd.DataFrame, np.random.Generator], pd.DataFrame]],
    alpha: float = 0.05,
    B: int = 200,
    R: int = 30,
    seed: int = 0,
) -> pd.DataFrame:
    """Compute a minimal size/power table.

    - scenarios: name -> generator(rng) -> df
    - score_fn_factory: creates a score_fn bound to the *design set* for each df
    - null_generator_factory: creates a null generator appropriate for each df
    """
    rng = np.random.default_rng(seed)
    rows = []

    for name, gen in scenarios.items():
        rejects = 0
        for r in range(int(R)):
            df_r = gen(rng)
            score_fn = score_fn_factory(df_r)
            null_gen = null_generator_factory(df_r)
            t = run_stage1_testification(
                df_r,
                score_fn=score_fn,
                null_generator=null_gen,
                B=int(B),
                alpha=float(alpha),
                seed=int(rng.integers(0, 2**31 - 1)),
                null_mode=name,
            )
            if t.decision == "FAIL":
                rejects += 1

        rows.append(
            {
                "scenario": name,
                "alpha": float(alpha),
                "rejection_rate": float(rejects / float(R)),
                "B": int(B),
                "R": int(R),
            }
        )

    return pd.DataFrame(rows)


def benchmark_size_power(*, seed: int = 7, alpha: float = 0.10, B: int = 200, R: int = 30, n_designs: int = 200) -> pd.DataFrame:
    """Convenience benchmark used by the CLI demo.

    The full econometric benchmark (re-running adapters on synthetic panels)
    can be computationally heavy. For a fast, always-available smoke test,
    this function benchmarks the Stage1 *scoring + testification* mechanism
    directly on synthetic design-score vectors g.

    Scenarios:
      - stable_null: iid N(0,1)
      - local_violation: mean-shifted N(0.4,1)
      - regime_shift: mixture with a small mass of very large scores
    """

    from .softmax import stage1_score

    rng = np.random.default_rng(int(seed))

    def _K_rule(n: int) -> int:
        # Mimic EngineConfig defaults without importing EngineConfig here.
        k_min = 10
        rho = 0.05
        return min(n, max(k_min, int(np.ceil(rho * n))))

    def _null_draw(rng_: np.random.Generator) -> np.ndarray:
        return rng_.normal(loc=0.0, scale=1.0, size=int(n_designs)).astype(float)

    def _crit_val(rng_: np.random.Generator, K: int) -> float:
        vals = []
        for _ in range(int(B)):
            g = _null_draw(rng_)
            vals.append(stage1_score(g, K)["S_wc"])
        return float(np.quantile(np.asarray(vals, dtype=float), 1.0 - float(alpha)))

    scenarios = {
        "Stable null (size)": lambda rng_: rng_.normal(0.0, 1.0, int(n_designs)).astype(float),
        "Local violation (power)": lambda rng_: rng_.normal(0.4, 1.0, int(n_designs)).astype(float),
        "Regime shift (power)": lambda rng_: (
            rng_.normal(0.0, 1.0, int(n_designs)).astype(float)
            + (rng_.random(int(n_designs)) < 0.05) * rng_.normal(3.0, 0.5, int(n_designs)).astype(float)
        ),
    }

    rows = []
    K = _K_rule(int(n_designs))

    # Precompute critical value once (null reference)
    c_alpha = _crit_val(rng, K)

    for name, gen in scenarios.items():
        rejects = 0
        for _ in range(int(R)):
            g_obs = gen(rng)
            S_obs = float(stage1_score(g_obs, K)["S_wc"])
            if S_obs >= c_alpha:
                rejects += 1
        rows.append(
            {
                "scenario": name,
                "alpha": float(alpha),
                "rejection_rate": float(rejects / float(R)),
                "B": int(B),
                "R": int(R),
                "notes": f"n_designs={int(n_designs)}, K={int(K)}",
            }
        )

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Backward-compatible wrappers used by stabilitycheck.runner
# -----------------------------------------------------------------------------


def mc_stage1_test(
    g: np.ndarray,
    K: int,
    *,
    alpha: float = 0.05,
    B: int = 200,
    rng: np.random.Generator | None = None,
    null_mode: str = "bootstrap",
) -> Dict[str, object]:
    """Minimal Monte Carlo testification for Stage 1.

    The current codebase's core testification API is generic (operates on a
    score function and null generator). The CLI/runner historically expected a
    convenience wrapper that operates directly on a score vector `g`.
    """

    from .softmax import stage1_score, solve_lambda_for_neff

    g = np.asarray(g, dtype=float)
    g = g[np.isfinite(g)]
    K = int(max(1, min(int(K), len(g) if len(g) else 1)))

    if rng is None:
        rng = np.random.default_rng(0)

    # Observed score
    S_obs = float(stage1_score(g, K)["S_wc"])

    # Null scores: bootstrap resample g (fast, always defined)
    null_scores = np.empty(int(B), dtype=float)
    for b in range(int(B)):
        gb = rng.choice(g, size=len(g), replace=True)
        null_scores[b] = float(stage1_score(gb, K)["S_wc"])

    cv = float(np.quantile(null_scores, 1.0 - float(alpha)))
    p = float((np.sum(null_scores >= S_obs) + 1.0) / (len(null_scores) + 1.0))
    decision = "FAIL" if S_obs >= cv else "PASS"

    # Softmax QC (temperature λ solving Neff=K)
    sol = solve_lambda_for_neff(g, K)

    return {
        "S_obs": S_obs,
        "critical_value": cv,
        "p_value_mc": p,
        "alpha": float(alpha),
        "B_used": int(B),
        "decision": decision,
        "null_mode": str(null_mode),
        "null_scores": null_scores,
        "lambda": float(sol.get("lambda", np.nan)),
        "Neff": float(sol.get("neff", sol.get("Neff", np.nan))),
        "converged": bool(sol.get("converged", False)),
        "rel_err": float(sol.get("rel_err", np.nan)),
        "iters": int(sol.get("iters", 0)),
    }


def size_and_power_table(
    g: np.ndarray,
    K: int,
    *,
    alpha: float = 0.05,
    B: int = 200,
    R: int = 30,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Small, fast size/power sanity check table.

    This is intentionally lightweight and does not re-run full adapters.
    It benchmarks the Stage1 score/testification mechanism on synthetic `g`.
    """

    from .softmax import stage1_score

    g = np.asarray(g, dtype=float)
    g = g[np.isfinite(g)]
    n = int(len(g))
    if n <= 0:
        return pd.DataFrame([{"scenario": "empty", "rejection_rate": np.nan}])

    if rng is None:
        rng = np.random.default_rng(0)

    K = int(max(1, min(int(K), n)))

    # Critical value from null
    null_scores = []
    for _ in range(int(B)):
        g0 = rng.normal(0.0, 1.0, n)
        null_scores.append(float(stage1_score(g0, K)["S_wc"]))
    cv = float(np.quantile(np.asarray(null_scores, dtype=float), 1.0 - float(alpha)))

    def _rej_rate(gen) -> float:
        rej = 0
        for _ in range(int(R)):
            gg = gen(rng)
            S = float(stage1_score(gg, K)["S_wc"])
            if S >= cv:
                rej += 1
        return float(rej / float(R))

    scenarios = {
        "Stable null (size)": lambda r: r.normal(0.0, 1.0, n),
        "Local violation (power)": lambda r: r.normal(0.4, 1.0, n),
        "Regime shift (power)": lambda r: (
            r.normal(0.0, 1.0, n) + (r.random(n) < 0.05) * r.normal(3.0, 0.5, n)
        ),
    }

    rows = []
    for name, gen in scenarios.items():
        rows.append(
            {
                "scenario": name,
                "alpha": float(alpha),
                "rejection_rate": _rej_rate(gen),
                "B": int(B),
                "R": int(R),
                "n_designs": int(n),
                "K": int(K),
            }
        )

    return pd.DataFrame(rows)
