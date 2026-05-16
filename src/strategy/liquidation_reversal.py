import pandas as pd
import numpy as np
from .. import factors


class LiquidationReversalStrategy:
    def __init__(self,
                 half_life_decay=5,
                 leverage_scale=1.0,
                 ts_lookback=30,
                 sentiment_ma_window=50,
                 oi_level_lookback=90,
                 beta_lookback=60,
                 regime_window=5,
                 regime_filter_threshold=0.6):
        self.half_life_decay = half_life_decay
        self.leverage_scale = leverage_scale
        self.ts_lookback = ts_lookback
        self.sentiment_ma_window = sentiment_ma_window
        self.oi_level_lookback = oi_level_lookback
        self.beta_lookback = beta_lookback
        self.regime_window = regime_window
        self.regime_filter_threshold = regime_filter_threshold

    def calculate_ts_zscore(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        rolling_mean = df.rolling(window=window, min_periods=window//2).mean()
        rolling_std = df.rolling(window=window, min_periods=window//2).std()
        return (df - rolling_mean) / rolling_std.replace(0, np.nan).clip(-2, 2)

    def calculate_cs_zscore(self, df_wide: pd.DataFrame) -> pd.DataFrame:
        means = df_wide.mean(axis=1)
        stds = df_wide.std(axis=1)
        return df_wide.sub(means, axis=0).div(stds.replace(0, np.nan), axis=0)

    def neutralize_signal(self, signal_df: pd.DataFrame, beta_df: pd.DataFrame) -> pd.DataFrame:
        """
        Removes Beta exposure from the signals using Cross-Sectional Regression.
        CRITICAL: Masks 0s and NaNs so we only neutralize the ACTIVE bets.
        """
        neutralized_data = []

        # Iterate over indices that exist in both DataFrames
        common_idx = signal_df.index.intersection(beta_df.index)

        for ts in common_idx:
            y = signal_df.loc[ts]
            x = beta_df.loc[ts]

            # --- MASKING LOGIC ---
            valid_mask = (y != 0) & (y.notna()) & (
                x.notna()) & (~np.isinf(y)) & (~np.isinf(x))

            # Need at least 2 points to fit a line
            if valid_mask.sum() < 2:
                neutralized_data.append(y)
                continue

            Y_vals = y[valid_mask].values
            X_vals = x[valid_mask].values

            # --- FIX 2: Check for singular matrix (Low Variance) ---
            if np.var(X_vals) < 1e-8:
                # Variance is too low to fit a line; skip regression.
                neutralized_data.append(y)
                continue

            try:
                # Linear Regression: Y = m*X + c
                slope, intercept = np.polyfit(X_vals, Y_vals, 1)

                # Residual = Y - (m*X + c)
                residuals = Y_vals - (slope * X_vals + intercept)

                # Update only the active assets in the row
                new_row = y.copy()
                new_row[valid_mask] = residuals
                neutralized_data.append(new_row)

            except Exception:
                # If SVD fails, keep original signals
                neutralized_data.append(y)

        # Reconstruct DataFrame and ensure alignment with original index
        return pd.DataFrame(neutralized_data, index=common_idx, columns=signal_df.columns).reindex(signal_df.index).fillna(0.0)

    def _compute_signals_for_symbols(
        self,
        all_data: dict,
        active_symbols: list,
    ):
        """
        Core signal computation using ONLY `active_symbols` for CS operations.

        `all_data` still contains the full history of all symbols so the TS
        rolling window can warm up correctly — pre-epoch price/OI history is
        used for lookback but the CS normalization (mean/std across axis=1) only
        sees `active_symbols`.  This keeps the cross-sectional distribution
        stable and prevents universe-size changes from contaminating z-scores.

        Returns (weights_wide, score_history_df) indexed over the full date
        range of the data.  The caller slices to the epoch's active dates.
        """
        oi_dict, ls_dict, close_dict = {}, {}, {}
        for sym in active_symbols:
            df = all_data.get(sym)
            if df is None or df.empty:
                continue
            if 'open_interest' not in df.columns or 'ls_ratio' not in df.columns:
                continue
            oi_dict[sym] = df.set_index('ts')['open_interest']
            ls_dict[sym] = df.set_index('ts')['ls_ratio']
            close_dict[sym] = df.set_index('ts')['futures_close']

        df_oi = pd.DataFrame(oi_dict).ffill()
        df_ls = pd.DataFrame(ls_dict).ffill()
        df_close = pd.DataFrame(close_dict).ffill()

        raw_oi_chg = df_oi.pct_change().replace(
            [np.inf, -np.inf], np.nan).rolling(window=3).mean()
        raw_ls_chg = df_ls.pct_change().replace([np.inf, -np.inf], np.nan).rolling(window=3).mean()  # noqa: F841

        # CS zscore uses only active_symbols → consistent distribution within epoch
        cs_z_oi_chg = self.calculate_ts_zscore(raw_oi_chg, self.ts_lookback)
        final_z_oi_chg = self.calculate_ts_zscore(
            cs_z_oi_chg, self.ts_lookback)
        cs_z_ls_chg = self.calculate_ts_zscore(raw_ls_chg, self.ts_lookback)
        liquidation_shock = (-cs_z_oi_chg).clip(
            lower=0)

        # ma_trend = df_close.rolling(window=self.sentiment_ma_window).mean()
        # std_trend = df_close.rolling(window=self.sentiment_ma_window).std()
        # regime_score = ((df_close.rolling(window=self.regime_window).mean() - ma_trend) /
        #                 std_trend).clip(-3, 3).fillna(0.0)
        regime_score = (-cs_z_ls_chg).clip(lower=0)
        # regime_filter = self.calculate_cs_zscore(regime_score)
        # regime_filter = -cs_z_ls_chg
        # regime_score = regime_score.mask(
        #     regime_filter.abs() < self.regime_filter_threshold, 0)

        interaction_alpha = liquidation_shock * regime_score

        final_signal = interaction_alpha

        beta_df = factors.calc_beta_df(df_close, self.beta_lookback)
        final_signal = self.neutralize_signal(final_signal, beta_df)
        final_signal = final_signal.fillna(0.0)

        # 1. Smooth on the full signal (no zero contamination)
        final_signal = final_signal.ewm(
            halflife=self.half_life_decay, min_periods=self.half_life_decay).mean()

        ########################## regime masking ############################
        # Compute mask from the SMOOTHED signal so ranking is consistent
        # with the actual values being kept/discarded.
        is_active = final_signal.abs() > 1e-8
        smoothed_ia = interaction_alpha.ewm(
            halflife=self.half_life_decay, min_periods=self.half_life_decay).mean()
        abs_ia_active = smoothed_ia.abs().where(is_active)
        ia_pct = abs_ia_active.rank(axis=1, pct=True)
        q1_mask = is_active & (ia_pct >= 0.5)

        # Preserve UNMASKED signal for EBM consumption. The mask is a
        # trading-side filter (reject weakest-conviction names). The EBM
        # should see the full signal distribution so it can learn which
        # conviction levels actually predict forward return.
        final_signal_unmasked = final_signal.copy()

        # 2. Apply mask (currently disabled; uncomment to enable q1 cull)
        # final_signal = final_signal.where(q1_mask, 0.0)

        # 3. Normalize AFTER masking so surviving weights get full leverage.
        # Compute BOTH normalizations independently — when the mask is
        # disabled they're identical; when enabled the unmasked version
        # preserves its own L1-normalization.
        total_signal_strength = final_signal.abs().sum(axis=1).replace(0, 1.0)
        final_weights = final_signal.div(
            total_signal_strength, axis=0) * self.leverage_scale

        total_unmasked = final_signal_unmasked.abs().sum(axis=1).replace(0, 1.0)
        final_weights_unmasked = final_signal_unmasked.div(
            total_unmasked, axis=0) * self.leverage_scale

        def _stack(wide: pd.DataFrame, name: str) -> pd.Series:
            s = wide.stack()
            s.index.names = ['ts', 'symbol']
            s.name = name
            return s

        score_history_df = pd.concat([
            _stack(final_z_oi_chg,    'oi_z_score'),
            _stack(liquidation_shock, 'liquidation_shock'),
            _stack(regime_score,      'regime_score'),
            _stack(interaction_alpha, 'interaction_alpha'),
            _stack(final_signal,      'final_signal'),
            _stack(final_weights,     'position_qty'),
        ], axis=1).reset_index()

        return final_weights, final_weights_unmasked, score_history_df

    def generate_all_signals(self, all_data: dict, epoch_mask_df=None):
        """
        When epoch_mask_df is None: single-pass computation over all symbols.

        When epoch_mask_df is provided: per-epoch computation.
            For each distinct epoch (detected from epoch_mask_df), signals are
            computed using only that epoch's active symbols so the CS zscore
            distribution stays consistent within each epoch.  Full price/OI
            history is used for TS rolling warmup.  Only the epoch's active
            date rows are kept, then results are concatenated and re-indexed to
            the full date range.

            This avoids the contamination that occurs when a universe expansion
            (e.g. 150 → 300 symbols) shifts the CS distribution mid-backtest,
            causing the TS rolling window to mix two incompatible distributions
            for ~ts_lookback bars around every epoch transition.
        """
        if epoch_mask_df is None or epoch_mask_df.empty:
            all_symbols = [
                sym for sym, df in all_data.items()
                if not df.empty
                and 'open_interest' in df.columns
                and 'ls_ratio' in df.columns
            ]
            return self._compute_signals_for_symbols(all_data, all_symbols)

        # --- Detect epoch boundaries from epoch_mask_df ---
        # An epoch boundary is any date where the set of active symbols changes.
        # fillna(False) first: epoch_mask_df may contain NaN for dates outside
        # the backtest range, which causes boolean indexing to crash.
        active_col_sets = epoch_mask_df.fillna(False).apply(
            lambda row: frozenset(row.index[row]), axis=1
        )
        # Build list of (epoch_start_ts, active_symbols_set)
        epochs: list = []
        prev_set = None
        for ts, sym_set in active_col_sets.items():
            if sym_set != prev_set:
                epochs.append((ts, sym_set))
                prev_set = sym_set

        # Determine epoch end dates (one day before the next epoch starts)
        epoch_date_ranges = []
        all_dates = epoch_mask_df.index
        for i, (ep_start, sym_set) in enumerate(epochs):
            ep_end = epochs[i + 1][0] - \
                pd.Timedelta(days=1) if i + 1 < len(epochs) else all_dates[-1]
            epoch_date_ranges.append((ep_start, ep_end, list(sym_set)))

        # --- Per-epoch signal computation ---
        # Both masked and unmasked weight slices are collected so the engine
        # can route them to PnL vs. EBM-panel parquet respectively.
        weight_slices = []
        weight_unmasked_slices = []
        score_slices = []

        for ep_start, ep_end, active_symbols in epoch_date_ranges:
            if not active_symbols:
                continue

            print(f"  [Reversal] Epoch {ep_start.date()} → {ep_end.date()} "
                  f"({len(active_symbols)} symbols)")

            weights_ep, weights_unmasked_ep, scores_ep = \
                self._compute_signals_for_symbols(all_data, active_symbols)

            date_mask = (
                (weights_ep.index >= ep_start) & (weights_ep.index <= ep_end))
            weights_slice = weights_ep.loc[date_mask]
            weights_unmasked_slice = weights_unmasked_ep.loc[date_mask]
            scores_slice = scores_ep[
                (scores_ep['ts'] >= ep_start) & (scores_ep['ts'] <= ep_end)
            ]

            weight_slices.append(weights_slice)
            weight_unmasked_slices.append(weights_unmasked_slice)
            score_slices.append(scores_slice)

        if not weight_slices:
            empty_w = pd.DataFrame(0.0, index=epoch_mask_df.index,
                                   columns=epoch_mask_df.columns)
            return empty_w, empty_w.copy(), pd.DataFrame()

        # Concatenate and reindex to full date range with 0 for any missing dates
        def _assemble(slices):
            out = (pd.concat(slices)
                   .reindex(all_dates)
                   .fillna(0.0))
            return out.reindex(columns=epoch_mask_df.columns, fill_value=0.0)

        final_weights = _assemble(weight_slices)
        final_weights_unmasked = _assemble(weight_unmasked_slices)
        final_scores = pd.concat(score_slices, ignore_index=True)

        return final_weights, final_weights_unmasked, final_scores
