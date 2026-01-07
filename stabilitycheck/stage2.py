from __future__ import annotations
import numpy as np
import pandas as pd
from .engine import compute_design_table, compute_anchors, compute_g_scalar
from .softmax import stage1_score

def blocked_time_splits(df: pd.DataFrame, time_col: str, n_blocks: int):
    times = np.sort(df[time_col].unique())
    return list(np.array_split(times, int(max(2, n_blocks))))

def stage2_blocked(cfg, schema, registry, df: pd.DataFrame, tier_name: str, time_col: str = "time"):
    anchors = compute_anchors(cfg, registry, df)
    D_full = compute_design_table(cfg, schema, registry, df, tier_name)
    g_full = compute_g_scalar(cfg, D_full, anchors, pnorm=2, w2=2.0, w3=0.5)
    K_main = min(len(g_full), max(cfg.k_min, int(np.ceil(cfg.rho * len(g_full)))))

    blocks = blocked_time_splits(df, time_col, cfg.block_splits)
    out = []
    for bi, bt in enumerate(blocks, start=1):
        df_b = df[df[time_col].isin(bt)].copy()
        D_b = compute_design_table(cfg, schema, registry, df_b, tier_name)
        g_b = compute_g_scalar(cfg, D_b, anchors, pnorm=2, w2=2.0, w3=0.5)
        if len(g_b) < 10:
            continue
        res = stage1_score(g_b, min(K_main, len(g_b)))
        out.append({"block": bi, "n_times": int(len(bt)), "S_wc": res["S_wc"], "lambda": res["lambda"], "Neff": res["Neff"]})
    return pd.DataFrame(out), int(K_main)
