"""
Cross-sectional beta neutralization helper, factored out of
build_factor_panel during the phase-1 refactor.

Same OLS-residualization logic the strategy modules use — for each date,
regress the signal against the beta proxy across symbols and replace the
signal with the residual. Lifted verbatim from build_factor_panel._neutralize
so numerical behaviour is unchanged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _neutralize(signal_df: pd.DataFrame, beta_df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional beta-neutralization via OLS residuals (matches strategy).

    Verbatim port of build_factor_panel._neutralize. Inputs:
      signal_df : (ts × symbol) raw signal to neutralize
      beta_df   : (ts × symbol) per-symbol beta exposure to residualise against

    Behaviour notes preserved as-is:
      - rows where fewer than 2 symbols have a finite signal/beta are skipped
      - rows where the beta variance across symbols is < 1e-8 are skipped
        (no informative cross-section)
      - exceptions inside np.polyfit fall through silently (legacy behaviour)
      - the returned frame is reindexed to signal_df.index and NaN-filled with 0.0
    """
    out = signal_df.copy()
    common_idx = signal_df.index.intersection(beta_df.index)
    for ts in common_idx:
        y = signal_df.loc[ts]
        x = beta_df.loc[ts]
        mask = (y != 0) & y.notna() & x.notna() & ~np.isinf(y) & ~np.isinf(x)
        if mask.sum() < 2 or np.var(x[mask].values) < 1e-8:
            continue
        try:
            slope, intercept = np.polyfit(x[mask].values, y[mask].values, 1)
            out.loc[ts, mask] = y[mask].values - \
                (slope * x[mask].values + intercept)
        except Exception:
            pass
    return out.reindex(signal_df.index).fillna(0.0)
