"""
Regime-gating helper for the EBM Mixture-of-Experts trainer, factored out
of src/scripts/train_ebm_signal.py during the phase-4a refactor.

`RegimeSelector` maps each prediction date to a discrete regime label,
applying the two correctness guards the MoE pipeline depends on:

  * Ex-ante lag — the regime at time t is derived from the value at t-1,
    eliminating any look-ahead at prediction time.
  * Hysteresis — the active label only switches after the new regime has
    been observed for `hysteresis` consecutive periods, reducing
    expert-switching turnover at threshold crossings.

The companion ResidualMoE ensemble class lives in
src/alpha/residual_moe.py (kept separate so loky workers can pickle
instances under a stable module path — see that file's docstring).
"""
from __future__ import annotations

import pandas as pd


class RegimeSelector:
    """
    Maps each prediction date to a discrete regime label using:
      1. Ex-ante lag  : regime at time t is derived from the value at t-1,
                        eliminating any look-ahead bias at prediction time.
      2. Hysteresis   : the active label only switches after the new regime
                        has been observed for `hysteresis` consecutive periods,
                        reducing expert-switching turnover.

    Parameters
    ----------
    panel       : full factor panel (must contain `ts` and `regime_col`).
    regime_col  : name of the numeric regime column (e.g. volatility_regime_enc).
    hysteresis  : minimum consecutive days in the new regime before switching.
    """

    def __init__(
        self,
        panel: pd.DataFrame,
        regime_col: str,
        hysteresis: int = 3,
    ):
        self.regime_col = regime_col
        self.hysteresis = hysteresis

        # Market-wide — same value for all symbols on a date, so .first() is fine.
        dates = sorted(panel["ts"].unique())
        raw_regime = (
            panel.groupby("ts")[regime_col]
            .first()
            .reindex(dates)
        )

        # Store as string keys to avoid float-comparison issues (0.0 vs 0).
        raw_str = raw_regime.apply(
            lambda v: str(int(v)) if pd.notna(v) else "nan"
        )

        # Build the raw lookup (no lag) for use during IS expert training.
        self._raw_map: dict = dict(zip(dates, raw_str.values))

        # Build lagged + hysteresis-smoothed map for OOS expert selection.
        lagged = raw_str.shift(1)  # NaN for the very first date

        active: "str | None" = None
        pending: "str | None" = None
        pending_count: int = 0
        smoothed: dict = {}

        for date in dates:
            raw_val = lagged.get(date)
            is_nan = (raw_val is None) or (
                raw_val == "nan") or pd.isna(raw_val)

            if is_nan:
                smoothed[date] = active  # None until data arrives
                continue

            if active is None:
                active = raw_val
                pending = None
                pending_count = 0
            elif raw_val == active:
                pending = None
                pending_count = 0
            elif raw_val == pending:
                pending_count += 1
                if pending_count >= hysteresis:
                    active = pending
                    pending = None
                    pending_count = 0
            else:
                pending = raw_val
                pending_count = 1

            smoothed[date] = active

        self._smoothed_map: dict = smoothed

    def get_regime(self, ts) -> "str | None":
        """
        Returns the active (lagged + hysteresis-smoothed) regime for
        OOS prediction at time `ts`.  None if not enough history yet.
        """
        return self._smoothed_map.get(ts)

    def get_raw_regime(self, ts) -> "str | None":
        """
        Returns the actual (non-lagged) regime at time `ts`.
        Use this for in-sample expert training only.
        """
        return self._raw_map.get(ts)
