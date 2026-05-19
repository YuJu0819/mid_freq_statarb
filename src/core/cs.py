"""
Cross-sectional and time-series z-score helpers, factored out of
build_factor_panel during the phase-1 refactor.

These are the LOCAL panel-helper versions (matching the original signatures
and behaviour exactly). A more elaborate NaN-aware variant lives in
src/factors.py as `calc_cs_zscore`; that one is used by the strategy
modules. The two implementations are kept separate during the refactor so
no numerical behaviour changes — merging them is deferred to a later
phase under explicit equivalence checks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _cs_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """Per-date cross-sectional z-score across symbols (panel-helper form).

    Matches the original `_cs_zscore` defined in build_factor_panel.py.
    Replace-zero-std semantics preserved verbatim.
    """
    mu = df.mean(axis=1)
    sd = df.std(axis=1).replace(0, np.nan)
    return df.sub(mu, axis=0).div(sd, axis=0)


def _ts_zscore(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Per-symbol time-series z-score over a rolling window.

    Matches the original `_ts_zscore` defined in build_factor_panel.py.
    `min_periods = window // 2` preserved verbatim.
    """
    mu = df.rolling(window, min_periods=window // 2).mean()
    sd = df.rolling(window, min_periods=window // 2).std().replace(0, np.nan)
    return (df - mu) / sd
