from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from .schema import spec_to_features, feature_columns

def rf_explain(D: pd.DataFrame, g: np.ndarray, seed: int, n_trees: int, n_boot: int):
    feats_list = [spec_to_features(s) for s in D["spec"]]
    cols = feature_columns(feats_list)
    X = np.column_stack([[f.get(c,0.0) for f in feats_list] for c in cols]).astype(float)

    rf = RandomForestRegressor(n_estimators=int(n_trees), random_state=int(seed), min_samples_leaf=2, n_jobs=-1)
    rf.fit(X, g)
    imp = rf.feature_importances_

    rng = np.random.default_rng(int(seed))
    imps = []
    n = len(X)
    for b in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        rf_b = RandomForestRegressor(n_estimators=max(200, int(n_trees)//2), random_state=int(seed+1000+b), min_samples_leaf=2, n_jobs=-1)
        rf_b.fit(X[idx], g[idx])
        imps.append(rf_b.feature_importances_)
    imps = np.asarray(imps)
    lo = np.quantile(imps, 0.05, axis=0)
    hi = np.quantile(imps, 0.95, axis=0)
    return {"cols": cols, "X": X, "rf": rf, "imp": imp, "imp_lo": lo, "imp_hi": hi}

def pdp_ice(rf, X: np.ndarray, feat_index: int, grid: np.ndarray, ice_n: int, seed: int):
    X_base = X.copy()
    pdp = []
    for gv in grid:
        Xt = X_base.copy()
        Xt[:, feat_index] = gv
        pdp.append(np.mean(rf.predict(Xt)))
    pdp = np.asarray(pdp)

    rng = np.random.default_rng(int(seed))
    n_ice = int(min(ice_n, len(X)))
    pick = rng.choice(len(X), size=n_ice, replace=False)
    ice_lines = []
    for idx in pick:
        yline = []
        for gv in grid:
            Xt = X[idx:idx+1].copy()
            Xt[:, feat_index] = gv
            yline.append(float(rf.predict(Xt)[0]))
        ice_lines.append(yline)
    return pdp, ice_lines
