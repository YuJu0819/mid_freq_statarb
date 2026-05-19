"""
Factor panel I/O helpers, factored out of src/scripts/build_factor_panel.py
during the phase-6 refactor.

Contents
--------
  load_metrics_store  read open_interest / ls_ratio per symbol from the
                      historical metrics archive (CSV) + the recent
                      ls_ratio parquet accumulation store
  attach_signals      left-join strategy weight parquets onto the panel by
                      (ts, symbol) so the EBM can train on (or alongside)
                      the strategy outputs

Both are pure data-loaders with file I/O; no closure dependencies. Lifted
verbatim from build_factor_panel.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .data.loader import _load_local_metrics


def load_metrics_store(
    symbols: "list[str]",
    start_date: str,
    end_date: str,
    metrics_dir: str = "./data/metrics",
    ls_parquet_dir: str = "./data/ls_ratio",
) -> "tuple[dict[str, pd.Series], dict[str, pd.Series]]":
    """
    Load open_interest and ls_ratio for each symbol, merging two sources:

      Primary  : data/metrics/{symbol}_metrics_*.csv
                 Full historical archive (downloaded by download_metrics.py).
                 Contains both open_interest and ls_ratio.

      Fallback : data/ls_ratio/{symbol}_ls_ratio.parquet
                 Recent accumulation store (download_ls_ratio.py).
                 Covers only the last N days but has finer granularity.

    Returns
    -------
    oi_store  : {symbol: pd.Series(ts → open_interest)}
    ls_store  : {symbol: pd.Series(ts → ls_ratio)}

    Why this replaces the old load_ls_store + _load_local_oi pair:
      - _load_local_oi() used the wrong filename pattern
        ({symbol}-metrics-{date}.csv with hyphens) while the actual files
        use underscores ({symbol}_metrics_{date}.csv).
      - load_ls_store() read only the recent parquet accumulation store,
        missing all historical dates in the backtest window.
    """
    t0 = pd.to_datetime(start_date)
    t1 = pd.to_datetime(end_date)

    oi_store: "dict[str, pd.Series]" = {}
    ls_store: "dict[str, pd.Series]" = {}

    for sym in symbols:
        frames = []

        # ── Primary: historical metrics CSVs ─────────────────────────────────
        df_hist = _load_local_metrics(sym, metrics_dir)
        if not df_hist.empty:
            df_hist = df_hist[(df_hist["ts"] >= t0) & (df_hist["ts"] <= t1)]
            frames.append(df_hist)

        # ── Fallback: recent parquet accumulation store ───────────────────────
        parquet_path = os.path.join(ls_parquet_dir, f"{sym}_ls_ratio.parquet")
        if os.path.exists(parquet_path):
            try:
                df_rec = pd.read_parquet(parquet_path)
                df_rec["ts"] = pd.to_datetime(df_rec["ts"])
                df_rec = df_rec[(df_rec["ts"] >= t0) & (df_rec["ts"] <= t1)]
                frames.append(df_rec)
            except Exception as e:
                print(f"  [metrics_store] {sym} parquet: {e}")

        if not frames:
            continue

        merged = (
            pd.concat(frames)
            .sort_values("ts")
            .drop_duplicates("ts", keep="last")
            .reset_index(drop=True)
        )

        if "open_interest" in merged.columns:
            s = merged.set_index("ts")["open_interest"].dropna()
            if not s.empty:
                oi_store[sym] = s

        if "ls_ratio" in merged.columns:
            s = merged.set_index("ts")["ls_ratio"].dropna()
            if not s.empty:
                ls_store[sym] = s

    return oi_store, ls_store


def attach_signals(
    panel: pd.DataFrame,
    run_id: str,
    strategy_names: "list[str]" = ("momentum", "reversal"),
    signal_cols: "list[str]" = ("mom_signal", "rev_signal"),
) -> pd.DataFrame:
    """
    Loads weight parquets and left-joins them onto the panel by (ts, symbol).
    Missing entries become NaN (symbol not in that strategy's universe on that day).
    """
    base = f"./reports/strategies/{run_id}"
    for name, col in zip(strategy_names, signal_cols):
        path = os.path.join(base, f"{name}.parquet")
        if not os.path.exists(path):
            print(
                f"  [signals] {path} not found — column '{col}' will be NaN.")
            panel[col] = np.nan
            continue
        w = pd.read_parquet(path)
        if not pd.api.types.is_datetime64_any_dtype(w.index):
            w.index = pd.to_datetime(w.index)
        # Stack to long
        stacked = w.stack().reset_index()
        stacked.columns = ["ts", "symbol", col]
        panel = panel.merge(stacked, on=["ts", "symbol"], how="left")
        print(f"  [signals] Attached '{col}' from {path}")
    return panel
