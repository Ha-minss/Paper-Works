"""Split/cross-fit Stage1 utilities.

Purpose
- Separate *search/training* (learn adversarial weights) from *evaluation* (score).
- Reduce post-selection bias criticism in Stage1.

Rule
- Train fold: estimate lambda and mu (adversary weights) from g_train.
- Test fold: keep mu fixed; NEVER re-estimate lambda or mu.

Outputs
- Fold-level summary at K_main (recommended K_rule)
- Split K-curve (mean + SE across folds)
- Weight concentration diagnostic (top-share of weights)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import engine
from .schema import SchemaConfig
from .configs import EngineConfig
from .softmax import stage1_score


@dataclass
class SplitStage1Artifacts:
    split_summary_df: pd.DataFrame
    kcurve_split_df: pd.DataFrame
    weights_train_topshare_df: pd.DataFrame


def _kfold_groups(groups: np.ndarray, n_folds: int, rng: np.random.Generator) -> List[np.ndarray]:
    perm = rng.permutation(groups)
    folds = np.array_split(perm, n_folds)
    return [np.array(f, dtype=groups.dtype) for f in folds]


def run_stage1_split(
    df: pd.DataFrame,
    designs: List[Dict[str, Any]],
    registry: Any,
    schema: SchemaConfig,
    cfg: EngineConfig,
    K_grid: List[int],
    K_main: int,
    n_folds: int = 2,
    seed: int = 0,
    group_col: Optional[str] = "unit",
) -> SplitStage1Artifacts:
    """Run split/cross-fit Stage1.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    designs : list of specs
        Design list for the tier.
    registry : AdapterRegistry
        Adapter registry.
    schema : SchemaConfig
        Schema configuration.
    cfg : EngineConfig
        Engine configuration.
    K_grid : list[int]
        K values for the split K-curve.
    K_main : int
        The main K to report fold summary (typically K_rule).
    n_folds : int
        Number of folds.
    seed : int
        RNG seed.
    group_col : str or None
        If not None and exists, do group-wise split (recommended for panel/unit data).

    Returns
    -------
    SplitStage1Artifacts
    """

    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")

    rng = np.random.default_rng(seed)

    # Choose split keys.
    use_group = bool(group_col) and (group_col in df.columns)
    if use_group:
        groups = df[group_col].dropna().unique()
        fold_groups = _kfold_groups(groups, n_folds=n_folds, rng=rng)
        fold_masks = [df[group_col].isin(g) for g in fold_groups]
    else:
        idx = np.arange(len(df))
        fold_idx = np.array_split(rng.permutation(idx), n_folds)
        fold_masks = [df.index.isin(i) for i in fold_idx]

    # Per-fold records
    fold_rows: List[Dict[str, Any]] = []
    top_share_rows: List[Dict[str, Any]] = []
    kcurve_rows: List[Dict[str, Any]] = []

    for fold_id, test_mask in enumerate(fold_masks):
        test_df = df.loc[test_mask].copy()
        train_df = df.loc[~test_mask].copy()

        # Anchors learned on TRAIN only and fixed for test.
        anchors_train = engine.compute_anchors(train_df, cfg, schema, registry)

        D_train = engine.compute_design_table(train_df, designs, registry, schema, cfg)
        D_test = engine.compute_design_table(test_df, designs, registry, schema, cfg)

        if D_train.empty or D_test.empty:
            # Still record fold summary with NaNs to avoid silent pass.
            for K in K_grid:
                kcurve_rows.append(
                    {
                        "fold": fold_id,
                        "K": int(K),
                        "S_wc_test": np.nan,
                        "max_g_test": np.nan,
                        "lambda_train": np.nan,
                        "Neff_train": np.nan,
                    }
                )
            fold_rows.append(
                {
                    "fold": fold_id,
                    "n_train": int(len(train_df)),
                    "n_test": int(len(test_df)),
                    "K": int(K_main),
                    "lambda_train": np.nan,
                    "Neff_train": np.nan,
                    "S_wc_test": np.nan,
                    "max_g_test": np.nan,
                }
            )
            continue

        # Align designs by spec_key (intersection) to keep mu/g compatible.
        common = pd.merge(
            D_train[["spec_key"]].drop_duplicates(),
            D_test[["spec_key"]].drop_duplicates(),
            on="spec_key",
            how="inner",
        )
        if common.empty:
            for K in K_grid:
                kcurve_rows.append(
                    {
                        "fold": fold_id,
                        "K": int(K),
                        "S_wc_test": np.nan,
                        "max_g_test": np.nan,
                        "lambda_train": np.nan,
                        "Neff_train": np.nan,
                    }
                )
            fold_rows.append(
                {
                    "fold": fold_id,
                    "n_train": int(len(train_df)),
                    "n_test": int(len(test_df)),
                    "K": int(K_main),
                    "lambda_train": np.nan,
                    "Neff_train": np.nan,
                    "S_wc_test": np.nan,
                    "max_g_test": np.nan,
                }
            )
            continue

        Dtr = D_train.merge(common, on="spec_key", how="inner").sort_values("spec_key").reset_index(drop=True)
        Dte = D_test.merge(common, on="spec_key", how="inner").sort_values("spec_key").reset_index(drop=True)

        g_train = engine.compute_g_scalar(Dtr, anchors_train, cfg)
        g_test = engine.compute_g_scalar(Dte, anchors_train, cfg)

        # For each K: learn mu on train and evaluate on test.
        for K in K_grid:
            sc_train = stage1_score(g_train, K)
            mu_train = sc_train["mu"]

            S_test = float(np.sum(mu_train * g_test))
            kcurve_rows.append(
                {
                    "fold": fold_id,
                    "K": int(K),
                    "S_wc_test": S_test,
                    "max_g_test": float(np.max(g_test)) if len(g_test) else np.nan,
                    "lambda_train": float(sc_train["lambda"]),
                    "Neff_train": float(sc_train["Neff"]),
                }
            )

            if int(K) == int(K_main):
                # Weight concentration diagnostics at K_main
                w = np.sort(mu_train)[::-1]
                for top_frac in (0.01, 0.05, 0.10):
                    k = max(1, int(np.ceil(top_frac * len(w))))
                    top_share_rows.append(
                        {
                            "fold": fold_id,
                            "top_frac": float(top_frac),
                            "weight_share": float(np.sum(w[:k])) if len(w) else np.nan,
                        }
                    )

        # Fold summary row at K_main
        sc_train_main = stage1_score(g_train, int(K_main))
        mu_main = sc_train_main["mu"]
        fold_rows.append(
            {
                "fold": fold_id,
                "n_train": int(len(train_df)),
                "n_test": int(len(test_df)),
                "K": int(K_main),
                "lambda_train": float(sc_train_main["lambda"]),
                "Neff_train": float(sc_train_main["Neff"]),
                "S_wc_test": float(np.sum(mu_main * g_test)),
                "max_g_test": float(np.max(g_test)) if len(g_test) else np.nan,
            }
        )

    # Aggregate split K-curve
    kcurve_long = pd.DataFrame(kcurve_rows)
    kcurve_split = (
        kcurve_long.groupby("K")
        .agg(
            S_wc_split_mean=("S_wc_test", "mean"),
            S_wc_split_se=("S_wc_test", lambda x: float(np.nanstd(x, ddof=1) / np.sqrt(np.sum(~np.isnan(x)))))
            if np.sum(~np.isnan(x)) > 1
            else np.nan,
            max_g_split_mean=("max_g_test", "mean"),
        )
        .reset_index()
        .sort_values("K")
    )

    return SplitStage1Artifacts(
        split_summary_df=pd.DataFrame(fold_rows),
        kcurve_split_df=kcurve_split,
        weights_train_topshare_df=pd.DataFrame(top_share_rows),
    )
