from __future__ import annotations

"""Softmax adversary utilities.

The unit tests for this repo define **lambda** as a *temperature* parameter:

    mu_d(lambda) ∝ exp(g_d / lambda)

so **larger** lambda makes weights *flatter* (larger effective sample size), and
**smaller** lambda concentrates mass on high-penalty designs.

We solve for lambda such that the effective sample size

    Neff(mu) = 1 / sum_d mu_d^2

is approximately a user-chosen target K.

Design goals:
  - Robust for degenerate inputs (NaN, inf, flat g)
  - Never raises in normal use
  - API matches tests: softmax_weights(), neff(), solve_lambda_for_neff(..., tol_rel=..., max_iter=...)
"""

from typing import Dict, Tuple

import numpy as np


def softmax_weights(g: np.ndarray, lam: float) -> np.ndarray:
    """Softmax weights with temperature lambda.

    mu ∝ exp(g / lam)
    """
    g = np.asarray(g, dtype=float).reshape(-1)
    n = int(g.size)
    if n <= 0:
        return np.array([], dtype=float)

    # If g is all non-finite, return uniform.
    if not np.isfinite(g).any():
        return np.ones(n, dtype=float) / n

    # Replace non-finite with minimum finite value (so they never dominate).
    gmin = float(np.nanmin(g[np.isfinite(g)]))
    gf = np.where(np.isfinite(g), g, gmin)

    # Temperature handling: lam -> 0 concentrates on max(g).
    if not np.isfinite(lam):
        # lam = +inf => uniform
        return np.ones(n, dtype=float) / n
    lam = float(lam)
    if lam <= 0:
        # Hard-max
        mu = np.zeros(n, dtype=float)
        mu[int(np.nanargmax(gf))] = 1.0
        return mu

    z = gf / lam
    z = z - float(np.max(z))  # stability
    w = np.exp(z)
    s = float(np.sum(w))
    if not np.isfinite(s) or s <= 0:
        return np.ones(n, dtype=float) / n
    return w / s


def neff(mu: np.ndarray) -> float:
    """Effective sample size Neff = 1 / sum(mu^2)."""
    mu = np.asarray(mu, dtype=float).reshape(-1)
    if mu.size == 0:
        return 0.0
    s2 = float(np.sum(mu * mu))
    if s2 <= 0 or not np.isfinite(s2):
        return float("nan")
    return float(1.0 / s2)


def solve_lambda_for_neff(
    g: np.ndarray,
    K: int,
    *,
    tol_rel: float = 1e-6,
    max_iter: int = 60,
    bracket_lo: float | None = None,
    bracket_hi: float | None = None,
) -> Dict[str, object]:
    """Find temperature lambda such that Neff(softmax_weights(g, lambda)) ~= K.

    Returns a dict with at least keys: lambda, mu, neff, converged, rel_err, iters.
    (Also includes legacy alias Neff for compatibility.)
    """
    g = np.asarray(g, dtype=float).reshape(-1)
    n = int(g.size)
    K = int(K)

    if n <= 0:
        mu = np.array([], dtype=float)
        return {
            "lambda": 0.0,
            "mu": mu,
            "neff": 0.0,
            "Neff": 0.0,
            "converged": True,
            "rel_err": 0.0,
            "iters": 0,
        }

    # Flat / degenerate g: Neff is always n.
    if (not np.isfinite(g).any()) or float(np.nanmax(g) - np.nanmin(g)) == 0.0:
        mu = np.ones(n, dtype=float) / n
        return {
            "lambda": 1.0,
            "mu": mu,
            "neff": float(n),
            "Neff": float(n),
            "converged": False if K != n else True,
            "rel_err": abs(float(n) - float(K)) / max(float(K), 1.0),
            "iters": 0,
        }

    # K bounds
    if K <= 1:
        mu = softmax_weights(g, 0.0)
        return {
            "lambda": 0.0,
            "mu": mu,
            "neff": neff(mu),
            "Neff": neff(mu),
            "converged": True,
            "rel_err": 0.0,
            "iters": 0,
        }
    if K >= n:
        mu = np.ones(n, dtype=float) / n
        return {
            "lambda": float("inf"),
            "mu": mu,
            "neff": float(n),
            "Neff": float(n),
            "converged": True,
            "rel_err": abs(float(n) - float(K)) / max(float(K), 1.0),
            "iters": 0,
        }

    target = float(K)

    # Bracket selection. Neff is monotone increasing in lambda.
    lo = float(bracket_lo) if bracket_lo is not None else 1e-6
    hi = float(bracket_hi) if bracket_hi is not None else 1.0

    mu_lo = softmax_weights(g, lo)
    ne_lo = neff(mu_lo)
    # Ensure lo is "small enough" so Neff(lo) <= K (concentrated)
    shrink = 0
    while np.isfinite(ne_lo) and ne_lo > target and shrink < 60:
        lo *= 0.5
        if lo < 1e-18:
            break
        mu_lo = softmax_weights(g, lo)
        ne_lo = neff(mu_lo)
        shrink += 1

    mu_hi = softmax_weights(g, hi)
    ne_hi = neff(mu_hi)
    expand = 0
    while np.isfinite(ne_hi) and ne_hi < target and expand < 60:
        hi *= 2.0
        if hi > 1e18:
            break
        mu_hi = softmax_weights(g, hi)
        ne_hi = neff(mu_hi)
        expand += 1

    # If we failed to bracket, return best effort (still fail-soft).
    if not (np.isfinite(ne_lo) and np.isfinite(ne_hi)):
        mu = softmax_weights(g, 1.0)
        ne_val = neff(mu)
        return {
            "lambda": 1.0,
            "mu": mu,
            "neff": float(ne_val),
            "Neff": float(ne_val),
            "converged": False,
            "rel_err": float("inf"),
            "iters": 0,
        }

    if ne_lo > target and ne_hi > target:
        # Can't get Neff as small as K even at very small lambda.
        rel_err = abs(ne_lo - target) / max(target, 1e-12)
        return {
            "lambda": float(lo),
            "mu": mu_lo,
            "neff": float(ne_lo),
            "Neff": float(ne_lo),
            "converged": False,
            "rel_err": float(rel_err),
            "iters": 0,
        }
    if ne_lo < target and ne_hi < target:
        # Can't get Neff as large as K.
        rel_err = abs(ne_hi - target) / max(target, 1e-12)
        return {
            "lambda": float(hi),
            "mu": mu_hi,
            "neff": float(ne_hi),
            "Neff": float(ne_hi),
            "converged": False,
            "rel_err": float(rel_err),
            "iters": 0,
        }

    lam = hi
    mu = mu_hi
    ne_val = ne_hi
    for it in range(int(max_iter)):
        mid = 0.5 * (lo + hi)
        mu_mid = softmax_weights(g, mid)
        ne_mid = neff(mu_mid)
        if not np.isfinite(ne_mid):
            break
        # Neff increases with lambda
        if ne_mid < target:
            lo = mid
            mu_lo, ne_lo = mu_mid, ne_mid
        else:
            hi = mid
            mu_hi, ne_hi = mu_mid, ne_mid
        lam = mid
        mu = mu_mid
        ne_val = ne_mid
        rel_err = abs(ne_val - target) / max(target, 1e-12)
        if rel_err <= float(tol_rel):
            return {
                "lambda": float(lam),
                "mu": mu,
                "neff": float(ne_val),
                "Neff": float(ne_val),
                "converged": True,
                "rel_err": float(rel_err),
                "iters": it + 1,
                "bracket_lo": float(lo),
                "bracket_hi": float(hi),
            }

    rel_err = abs(ne_val - target) / max(target, 1e-12) if np.isfinite(ne_val) else float("inf")
    return {
        "lambda": float(lam),
        "mu": mu,
        "neff": float(ne_val) if np.isfinite(ne_val) else float("nan"),
        "Neff": float(ne_val) if np.isfinite(ne_val) else float("nan"),
        "converged": False,
        "rel_err": float(rel_err),
        "iters": int(max_iter),
        "bracket_lo": float(lo),
        "bracket_hi": float(hi),
    }


def stage1_weights_for_K(g: np.ndarray, K: int) -> Tuple[np.ndarray, Dict[str, object]]:
    rep = solve_lambda_for_neff(g, K)
    return np.asarray(rep["mu"], dtype=float), rep


def stage1_score(g: np.ndarray, K: int) -> Dict[str, float]:
    """Worst-case stability score S_wc(K): mean of the K largest g values."""
    g = np.asarray(g, dtype=float).reshape(-1)
    if g.size == 0:
        return {"S_wc": float("nan")}
    K = int(max(1, min(int(K), g.size)))
    order = np.argsort(-g)
    top = g[order][:K]
    return {"S_wc": float(np.mean(top))}
