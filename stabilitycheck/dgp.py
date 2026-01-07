from __future__ import annotations
import numpy as np
import pandas as pd

def simulate_panel(seed: int = 7, N_units: int = 70, T_periods: int = 72, treat_share: float = 0.5,
                   policy_t: int = 36, break_t: int = 48, tau_pre: float = -0.18, tau_post: float = -0.02):
    rng = np.random.default_rng(int(seed))
    treat_units = set(rng.choice(np.arange(N_units), size=int(N_units*treat_share), replace=False))
    rows = []
    for u in range(N_units):
        a_u = rng.normal(0, 0.8)
        for t in range(T_periods):
            g_t = 0.02*t + 0.35*np.sin(2*np.pi*t/12)
            post = 1 if t >= policy_t else 0
            treat = 1 if u in treat_units else 0
            did = treat*post
            x1, x2, x3 = rng.normal(size=3)
            tau = tau_pre if t < break_t else tau_post
            eps = rng.normal(0, 1.0*(1.0+0.4*post))
            y_lat = 1.3 + a_u + g_t + 0.35*x1 - 0.20*x2 + 0.12*x3 + tau*did + eps
            y = np.exp(y_lat/3.0)
            rows.append((u,t,treat,post,did,x1,x2,x3,y))
    return pd.DataFrame(rows, columns=["unit","time","treat","post","did","x1","x2","x3","y"])
