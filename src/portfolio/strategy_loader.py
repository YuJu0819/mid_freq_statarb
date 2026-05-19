"""
Strategy-parquet discovery and alignment, factored out of backtest_combo
during the phase-3 refactor.

The combiner expects each individual strategy to ship a `(ts × symbol)`
weight matrix as `<name>.parquet` under the run directory. This module
discovers those parquets (with the same exclusion rules the combo script
applied inline), reads them, and aligns them onto a master timeline and
master column union so downstream code can stack them.

Lifted verbatim from backtest_combo.load_and_align_strategies. No
behaviour change.
"""
from __future__ import annotations

import glob
import os

import pandas as pd


# Files saved by the backtest engine and train_ebm_signal that are NOT
# strategy weight matrices (exclude these from auto-discovery).
EXCLUDE_PREFIXES: tuple[str, ...] = ("optimized_weights_",)
EXCLUDE_EXACT: set[str] = {"ebm_predictions.parquet"}


def load_and_align_strategies(
    run_dir: str,
    strategies: "list[str] | None" = None,
):
    """
    Load strategy weight parquets from `run_dir` and align them to a
    master `(ts × symbol)` timeline.

    Parameters
    ----------
    run_dir    : directory containing *.parquet weight files
    strategies : explicit list of base names (e.g. ["momentum","reversal","ebm"]).
                 If None, all *.parquet files are auto-discovered (excluding
                 optimized_weights_* and ebm_predictions.parquet).

    Returns
    -------
    (aligned_strategies, master_ts, master_cols)
        aligned_strategies : {name: DataFrame on master timeline}
        master_ts          : union of every strategy's date index
        master_cols        : union of every strategy's column set
    """
    if strategies:
        files = []
        for name in strategies:
            p = os.path.join(run_dir, f"{name}.parquet")
            if not os.path.exists(p):
                print(f"  [warn] {p} not found — skipping '{name}'.")
            else:
                files.append(p)
    else:
        all_files = glob.glob(os.path.join(run_dir, "*.parquet"))
        files = [
            f for f in all_files
            if os.path.basename(f) not in EXCLUDE_EXACT
            and not any(os.path.basename(f).startswith(pfx)
                        for pfx in EXCLUDE_PREFIXES)
        ]

    if not files:
        raise FileNotFoundError(f"No weight files found in {run_dir}")

    print(
        f"Found {len(files)} strategies: {[os.path.basename(f) for f in files]}")

    raw_strategies = {}
    for f in files:
        name = os.path.basename(f).replace(".parquet", "")
        df = pd.read_parquet(f)
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            df.index = pd.to_datetime(df.index)
        raw_strategies[name] = df

    # Create Union Index/Columns
    master_ts = pd.Index([])
    master_cols = pd.Index([])
    for df in raw_strategies.values():
        master_ts = master_ts.union(df.index)
        master_cols = master_cols.union(df.columns)

    master_ts = master_ts.sort_values().unique()
    master_cols = master_cols.sort_values().unique()

    # Align
    aligned_strategies = {}
    for name, df in raw_strategies.items():
        aligned_df = df.reindex(
            index=master_ts, columns=master_cols).fillna(0.0)
        aligned_strategies[name] = aligned_df

    return aligned_strategies, master_ts, master_cols
