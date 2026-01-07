"""Null generators for testification.

These utilities implement generic permutation/shuffle nulls for the canonical
column names used by adapters (especially DID/EventStudy adapters):
  unit, time, treat, post, y

If your adapter uses different column names, write a custom null generator and
pass it to `testification.run_stage1_testification(..., null_generator=...)`.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd


def permute_unit_treat(
    df: pd.DataFrame,
    rng: np.random.Generator,
    unit_col: str = "unit",
    treat_col: str = "treat",
) -> pd.DataFrame:
    """Permutation null: randomly permute treatment assignment across units."""
    if unit_col not in df.columns or treat_col not in df.columns:
        return df.copy()

    out = df.copy()
    unit_vals = out[[unit_col, treat_col]].drop_duplicates().sort_values(unit_col)
    units = unit_vals[unit_col].to_numpy()
    treats = unit_vals[treat_col].to_numpy()
    perm_treats = rng.permutation(treats)
    map_df = pd.DataFrame({unit_col: units, treat_col: perm_treats})
    out = out.drop(columns=[treat_col]).merge(map_df, on=unit_col, how="left")
    return out


def shuffle_y_within_unit(
    df: pd.DataFrame,
    rng: np.random.Generator,
    unit_col: str = "unit",
    y_col: str = "y",
) -> pd.DataFrame:
    """Permutation null: shuffle outcomes within each unit (keeps marginal distribution)."""
    if unit_col not in df.columns or y_col not in df.columns:
        return df.copy()

    out = df.copy()

    def _shuf(g: pd.DataFrame) -> pd.DataFrame:
        y = g[y_col].to_numpy()
        g[y_col] = rng.permutation(y)
        return g

    out = out.groupby(unit_col, group_keys=False).apply(_shuf)
    return out


def placebo_post_by_time_shift(
    df: pd.DataFrame,
    rng: np.random.Generator,
    time_col: str = "time",
    post_col: str = "post",
    min_shift: int = 1,
    max_shift: Optional[int] = None,
) -> pd.DataFrame:
    """Placebo null: circularly shift the `post` indicator along time.

    This is a simple workaround when a strict policy date placebo is desired but
    time encoding is generic. It preserves the fraction of post periods.
    """
    if time_col not in df.columns or post_col not in df.columns:
        return df.copy()

    out = df.copy()
    times = np.sort(out[time_col].dropna().unique())
    if len(times) <= 2:
        return out

    if max_shift is None:
        max_shift = max(1, len(times) - 2)

    shift = int(rng.integers(min_shift, max_shift + 1))

    # Build mapping time -> shifted time
    shifted = np.roll(times, shift)
    map_df = pd.DataFrame({time_col: times, "__time_shifted": shifted})

    out = out.merge(map_df, on=time_col, how="left")
    # re-assign post by reading it from shifted time index
    post_by_time = out[[time_col, post_col]].drop_duplicates().rename(columns={post_col: "__post_at_time"})
    post_by_shifted = post_by_time.rename(columns={time_col: "__time_shifted", "__post_at_time": post_col})
    out = out.drop(columns=[post_col]).merge(post_by_shifted, on="__time_shifted", how="left")
    out = out.drop(columns=["__time_shifted"])
    return out


# Convenience: a default null generator selection for DID-like data.

def default_null_generator(df: pd.DataFrame) -> Callable[[pd.DataFrame, np.random.Generator], pd.DataFrame]:
    """Choose a reasonable default null generator given columns."""
    if {"unit", "treat"}.issubset(df.columns):
        return lambda d, r: permute_unit_treat(d, r)
    if {"unit", "y"}.issubset(df.columns):
        return lambda d, r: shuffle_y_within_unit(d, r)
    return lambda d, r: d.copy()
