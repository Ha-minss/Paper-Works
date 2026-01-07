from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import optimize


def _logsumexp(a: np.ndarray, axis: int = -1) -> np.ndarray:
    a_max = np.max(a, axis=axis, keepdims=True)
    out = a_max + np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)


def _estimate_transitions(
    df: pd.DataFrame,
    id_col: str,
    state_col: str,
    choice_col: str,
    n_states: int,
    smoothing: float = 1e-3,
) -> np.ndarray:
    """Estimate P(s'|s,a) from data by simple frequencies.

    Returns
    -------
    P : ndarray, shape (A, S, S)
        P[a, s, s'] = Pr(s'|s,a)
    """
    states = df[state_col].to_numpy(dtype=int)
    choices = df[choice_col].to_numpy(dtype=int)
    ids = df[id_col].to_numpy()

    A = int(choices.max()) + 1
    P = np.zeros((A, n_states, n_states), dtype=float)

    # next-state pairs within id
    order = np.argsort(ids, kind="mergesort")
    ids_s = ids[order]
    st_s = states[order]
    ch_s = choices[order]

    # find consecutive within id
    same = ids_s[1:] == ids_s[:-1]
    s0 = st_s[:-1][same]
    a0 = ch_s[:-1][same]
    s1 = st_s[1:][same]

    for a in range(A):
        mask = a0 == a
        if mask.sum() == 0:
            P[a] = np.eye(n_states)
            continue
        counts = np.zeros((n_states, n_states), dtype=float)
        np.add.at(counts, (s0[mask], s1[mask]), 1.0)
        counts = counts + smoothing
        P[a] = counts / counts.sum(axis=1, keepdims=True)

    return P


def _vfi_logit_binary(
    theta: np.ndarray,
    beta: float,
    states_val: np.ndarray,
    P0: np.ndarray,
    P1: np.ndarray,
    max_iter: int = 2000,
    tol: float = 1e-10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Value function iteration for a binary logit DDC.

    Utility normalization:
    - action 0: u0(s) = 0
    - action 1: u1(s) = theta0 + theta1 * s_value

    Parameters
    ----------
    theta : array, (2,)
    beta : discount factor in (0,1)
    states_val : array, (S,) numeric state values (e.g., mileage bin centers)
    P0, P1 : transition matrices, shape (S,S) for each action

    Returns
    -------
    V : array, (S,) inclusive value
    CCP1 : array, (S,) Pr(action=1|state)
    """
    th0, th1 = float(theta[0]), float(theta[1])
    u0 = np.zeros_like(states_val, dtype=float)
    u1 = th0 + th1 * states_val

    S = len(states_val)
    V = np.zeros(S, dtype=float)

    for _ in range(max_iter):
        EV0 = P0 @ V
        EV1 = P1 @ V
        Q = np.stack([u0 + beta * EV0, u1 + beta * EV1], axis=1)  # (S,2)
        V_new = _logsumexp(Q, axis=1)
        if np.max(np.abs(V_new - V)) < tol:
            V = V_new
            break
        V = V_new

    # CCP
    EV0 = P0 @ V
    EV1 = P1 @ V
    Q = np.stack([u0 + beta * EV0, u1 + beta * EV1], axis=1)
    Q = Q - Q.max(axis=1, keepdims=True)
    expQ = np.exp(Q)
    denom = expQ.sum(axis=1)
    CCP1 = expQ[:, 1] / denom

    return V, CCP1


def _neg_loglike(theta: np.ndarray, beta: float, CCP1: np.ndarray, s_idx: np.ndarray, a: np.ndarray) -> float:
    p1 = np.clip(CCP1[s_idx], 1e-12, 1 - 1e-12)
    ll = a * np.log(p1) + (1 - a) * np.log(1 - p1)
    return float(-np.sum(ll))


def estimate_binary_logit_ddc(
    df: pd.DataFrame,
    id_col: str = "id",
    time_col: str = "t",
    state_col: str = "state",
    choice_col: str = "choice",
    next_state_col: str = "state_next",
    discount: float = 0.95,
    beta: Optional[float] = None,
    state_value_map: Optional[Dict[int, float]] = None,
    start_theta: Tuple[float, float] = (-1.0, 0.05),
    max_iter: int = 400,
    tol: float = 1e-6,
    max_iter_vfi: int = 1500,
) -> Dict[str, object]:
    """Estimate a simple binary logit Dynamic Discrete Choice model via NFXP.

    This is not meant to replace specialized structural packages, but it is
    a complete, dependency-light baseline that can live *inside* stabilitycheck.

    Returns a dict with theta_hat, cov, loglike, transitions, and diagnostics.
    """
    if beta is not None:
        discount = float(beta)

    if not (0.0 < float(discount) < 1.0):
        raise ValueError("discount must be in (0,1)")

    # map state to 0..S-1
    s_raw = df[state_col].to_numpy()
    uniq = np.unique(s_raw)
    state_map = {int(v): i for i, v in enumerate(uniq)}
    s_idx = np.array([state_map[int(v)] for v in s_raw], dtype=int)

    # binary choice
    a_raw = df[choice_col].to_numpy()
    uniq_a = np.unique(a_raw)
    if set(map(int, uniq_a)) - {0, 1}:
        raise ValueError("This baseline solver expects binary choice coded as 0/1.")
    a = a_raw.astype(int)

    S = len(uniq)
    if state_value_map is None:
        # use raw state value as numeric feature (scaled)
        sval = uniq.astype(float)
        if np.std(sval) > 0:
            sval = (sval - np.mean(sval)) / np.std(sval)
    else:
        sval = np.array([state_value_map[int(v)] for v in uniq], dtype=float)
        if np.std(sval) > 0:
            sval = (sval - np.mean(sval)) / np.std(sval)

    # transitions P(s'|s,a)
    tmp = df.copy()
    tmp["_s_idx"] = s_idx
    tmp["_a"] = a
    P = _estimate_transitions(tmp, id_col=id_col, state_col="_s_idx", choice_col="_a", n_states=S)
    P0, P1 = P[0], P[1]

    def obj(th: np.ndarray) -> float:
        _, CCP1 = _vfi_logit_binary(th, discount, sval, P0, P1, max_iter=max_iter_vfi)
        return _neg_loglike(th, discount, CCP1, s_idx, a)

    res = optimize.minimize(
        obj,
        x0=np.array(start_theta, dtype=float),
        method="BFGS",
        options={"gtol": float(tol), "maxiter": int(max_iter)},
    )

    theta_hat = res.x.astype(float)
    # final V, CCP
    V, CCP1 = _vfi_logit_binary(theta_hat, discount, sval, P0, P1, max_iter=max_iter_vfi)

    # covariance from (approx) inverse Hessian
    cov = None
    if hasattr(res, "hess_inv"):
        try:
            Hinv = np.array(res.hess_inv)
            cov = Hinv
        except Exception:
            cov = None

    out = {
        "ok": bool(res.success),
        "theta_hat": theta_hat,
        "cov": cov,
        "loglike": float(-res.fun),
        "discount": float(discount),
        "state_values": sval,
        "P0": P0,
        "P1": P1,
        "V": V,
        "CCP1": CCP1,
        "opt": {"message": str(res.message), "nit": int(res.nit), "nfev": int(res.nfev)},
        "state_map": state_map,
    }
    return out


def simulate_binary_logit_ddc(
    n_ids: int = 200,
    t_periods: int | None = None,
    T: int = 50,
    beta: float | None = None,
    discount: float = 0.95,
    theta: Tuple[float, float] = (-1.0, 0.8),
    seed: int = 7,
) -> pd.DataFrame:
    """Simulate a toy binary logit DDC dataset.

    Parameters (compat):
      - tests use t_periods and beta; we map to T and discount.

    Returns a DataFrame with columns: id, t, state, choice, state_next.
    We also include a duplicate column `time` for backward compatibility.
    """

    if t_periods is not None:
        T = int(t_periods)
    if beta is not None:
        discount = float(beta)

    rng = np.random.default_rng(seed)

    S = 10
    states_val = np.linspace(0, 1, S)

    # simple transitions: action 0 drifts up; action 1 tends to reset down
    P0 = np.zeros((S, S), dtype=float)
    P1 = np.zeros((S, S), dtype=float)
    for s in range(S):
        for sp in range(S):
            P0[s, sp] = np.exp(-3.0 * abs(sp - min(S - 1, s + 1)))
        P0[s] /= P0[s].sum()
        for sp in range(S):
            P1[s, sp] = np.exp(-3.0 * abs(sp - max(0, s - 2)))
        P1[s] /= P1[s].sum()

    # compute CCP from true theta
    _, CCP1 = _vfi_logit_binary(np.array(theta, dtype=float), float(discount), states_val, P0, P1)

    rows = []
    for i in range(int(n_ids)):
        s = int(rng.integers(0, S))
        for t in range(int(T)):
            p1 = float(CCP1[s])
            a = int(rng.random() < p1)
            P = P1 if a == 1 else P0
            sp = int(rng.choice(np.arange(S), p=P[s]))
            rows.append((i, t, s, a, sp))
            s = sp

    df = pd.DataFrame(rows, columns=["id", "t", "state", "choice", "state_next"])
    df["time"] = df["t"]
    return df
