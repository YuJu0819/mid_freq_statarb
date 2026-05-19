"""
Pre-launch and post-death row masking for the long-format factor panel,
factored out of src/scripts/build_factor_panel.py during the phase-6
refactor.

Both functions are pure — they take a long-format `(ts, symbol, factor...)`
panel and NaN-mask the rows where a symbol was either pre-launch (data
forward-filled before the contract was actually trading) or post-death
(delisted / rebranded, with the feed emitting the last close indefinitely).
Without these masks the rolling-feature pipeline downstream produces
all-zero values that survive `dropna(subset=["y"])` and corrupt EBM
training.

Lifted verbatim from build_factor_panel — no behaviour change.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def mask_pre_launch_rows(
    panel: pd.DataFrame,
    min_active_days: int = 5,
    return_col: str = "ret_1d",
) -> pd.DataFrame:
    """
    NaN-mask pre-tradeable rows for each symbol.

    A symbol's price series is "pre-launch" when the underlying data is
    forward-filled (or zero-filled) before the symbol was actually trading
    on the venue. In that period `ret_1d == 0` exactly for many consecutive
    days, which downstream produces all-zero rolling features (volatility_30,
    price_roc, mom_final_score, ...). Those synthetic-zero rows survive
    `dropna(subset=["y"])` because y == 0 (not NaN), and end up corrupting
    EBM training as if they were real observations.

    Detection: for each symbol, find the FIRST date `d*` such that the symbol
    has accumulated at least `min_active_days` non-zero, non-NaN returns
    within the panel up to and including `d*`. All rows BEFORE `d*` get
    every numeric column set to NaN. From `d*` onwards the panel is
    untouched.

    Parameters
    ----------
    panel           : long-format panel with [ts, symbol, return_col, ...]
    min_active_days : minimum number of non-zero return days before a symbol
                      is considered tradeable. 5 is enough to distinguish
                      forward-fill from a low-volume day.
    return_col      : column used to detect activity (default "ret_1d")

    Returns
    -------
    panel : same shape, with pre-launch rows NaN-masked.
    """
    if return_col not in panel.columns:
        print(f"  [pre_launch_mask] '{return_col}' missing — skipping.")
        return panel

    panel = panel.sort_values(["symbol", "ts"]).reset_index(drop=True)
    feature_cols = [c for c in panel.columns
                    if c not in ("ts", "symbol")
                    and pd.api.types.is_numeric_dtype(panel[c])]

    # For each symbol, find first date where cumulative count of non-zero
    # returns reaches min_active_days.
    is_active = (panel[return_col].fillna(0) != 0).astype(int)
    cum_active = is_active.groupby(panel["symbol"]).cumsum()
    # Mask rows where cumulative count is still below threshold
    pre_launch_mask = cum_active < min_active_days

    n_masked = int(pre_launch_mask.sum())
    if n_masked == 0:
        print(f"  [pre_launch_mask] no pre-launch rows detected.")
        return panel

    # Per-symbol first-active date and rows-masked count for reporting.
    # Distinguish symbols that traded from panel start (lose only ~min_active_days
    # rows) from genuinely-late-launching symbols.
    first_active = (panel.loc[~pre_launch_mask]
                    .groupby("symbol")["ts"].min())
    rows_masked_per_sym = pre_launch_mask.groupby(panel["symbol"]).sum()
    panel_start = panel["ts"].min()
    # "Late-launching" = first active date is more than 30 days after panel start
    late_syms = first_active[
        first_active > panel_start + pd.Timedelta(days=30)]

    panel.loc[pre_launch_mask, feature_cols] = np.nan
    print(f"  [pre_launch_mask] masked {n_masked:,} rows "
          f"({n_masked/len(panel):.1%} of panel)  "
          f"threshold = {min_active_days} non-zero return days")
    print(f"  [pre_launch_mask] {len(late_syms)} symbols launched >30 days "
          f"after panel start (the contamination source)")
    if len(late_syms):
        latest = first_active.loc[late_syms.index].sort_values(
            ascending=False).head(5)
        print(f"  [pre_launch_mask] latest-launching: "
              + ", ".join(f"{s}@{d.strftime('%Y-%m-%d')}"
                          for s, d in latest.items()))
    return panel


def mask_post_death_rows(
    panel: pd.DataFrame,
    min_active_days: int = 5,
    return_col: str = "ret_1d",
) -> pd.DataFrame:
    """
    NaN-mask post-death (trailing forward-filled) rows for each symbol.

    Symmetric counterpart to mask_pre_launch_rows. After Binance delists or
    rebrands a symbol (e.g. MATIC→POL, RNDR→RENDER, AGIX→FET, FTM→S), the
    historical data feed keeps emitting the last close price, producing an
    indefinite tail of `ret_1d == 0`. Those synthetic-zero rows survive
    `dropna(subset=["y"])` and pollute training the same way pre-launch
    rows do.

    Detection: for each symbol, find the LAST date `d*` such that the symbol
    still has at least `min_active_days` non-zero, non-NaN returns from
    `d*` onward (i.e., looking forward to panel end). All rows AFTER `d*`
    get every numeric column set to NaN. Implemented as a reverse cumsum,
    fully symmetric to the pre-launch helper.

    Parameters
    ----------
    panel           : long-format panel with [ts, symbol, return_col, ...]
    min_active_days : same threshold as pre-launch (default 5)
    return_col      : column used to detect activity (default "ret_1d")

    Returns
    -------
    panel : same shape, with post-death rows NaN-masked.
    """
    if return_col not in panel.columns:
        print(f"  [post_death_mask] '{return_col}' missing — skipping.")
        return panel

    panel = panel.sort_values(["symbol", "ts"]).reset_index(drop=True)
    feature_cols = [c for c in panel.columns
                    if c not in ("ts", "symbol")
                    and pd.api.types.is_numeric_dtype(panel[c])]

    # For each symbol, count remaining non-zero returns FROM each row to
    # the symbol's last row (reverse cumsum on the forward-time series).
    is_active = (panel[return_col].fillna(0) != 0).astype(int)
    rev_cum = (
        is_active[::-1]
        .groupby(panel["symbol"][::-1], sort=False)
        .cumsum()[::-1]
    )
    post_death_mask = rev_cum < min_active_days

    n_masked = int(post_death_mask.sum())
    if n_masked == 0:
        print(f"  [post_death_mask] no post-death rows detected.")
        return panel

    last_active = (panel.loc[~post_death_mask]
                   .groupby("symbol")["ts"].max())
    panel_end = panel["ts"].max()
    # "Dead-tail" = last active date is more than 30 days before panel end
    dead_syms = last_active[
        last_active < panel_end - pd.Timedelta(days=30)]

    panel.loc[post_death_mask, feature_cols] = np.nan
    print(f"  [post_death_mask] masked {n_masked:,} rows "
          f"({n_masked/len(panel):.1%} of panel)  "
          f"threshold = {min_active_days} non-zero return days")
    print(f"  [post_death_mask] {len(dead_syms)} symbols dead >30 days "
          f"before panel end (the contamination source)")
    if len(dead_syms):
        earliest_dead = last_active.loc[dead_syms.index].sort_values().head(5)
        print(f"  [post_death_mask] earliest deaths: "
              + ", ".join(f"{s}@{d.strftime('%Y-%m-%d')}"
                          for s, d in earliest_dead.items()))
    return panel
