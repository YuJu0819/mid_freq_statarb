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
        return (df - rolling_mean) / rolling_std.replace(0, np.nan)

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

    def generate_all_signals(self, all_data: dict):
        # 1. Prepare Wide DataFrames
        oi_dict = {}
        ls_dict = {}
        close_dict = {}

        for sym, df in all_data.items():
            if df.empty or 'open_interest' not in df.columns or 'ls_ratio' not in df.columns:
                continue
            oi_dict[sym] = df.set_index('ts')['open_interest']
            ls_dict[sym] = df.set_index('ts')['ls_ratio']
            close_dict[sym] = df.set_index('ts')['futures_close']

        # --- FIX 1: Use .ffill() instead of .fillna(method='ffill') ---
        df_oi = pd.DataFrame(oi_dict).ffill().fillna(0)
        df_ls = pd.DataFrame(ls_dict).ffill().fillna(1.0)
        df_close = pd.DataFrame(close_dict).ffill()

        # 2. Compute Changes
        raw_oi_chg = df_oi.pct_change()
        raw_ls_chg = df_ls.diff()

        # Clean infinite values
        raw_oi_chg = raw_oi_chg.replace([np.inf, -np.inf], np.nan)
        raw_ls_chg = raw_ls_chg.replace([np.inf, -np.inf], np.nan)

        # 3. Z-Score Normalization (CS -> TS)
        cs_z_oi_chg = self.calculate_cs_zscore(raw_oi_chg)
        final_z_oi_chg = self.calculate_ts_zscore(
            cs_z_oi_chg, self.ts_lookback)

        # 4. Continuous "Shock" Intensity
        liquidation_shock = (-final_z_oi_chg - 0.5).clip(lower=0.0)

        # 5. Regime Interaction Term
        ma_trend = df_close.rolling(window=self.sentiment_ma_window).mean()
        std_trend = df_close.rolling(window=self.sentiment_ma_window).std()

        # Cap outliers
        regime_score = ((df_close.rolling(window=self.regime_window).mean() - ma_trend) /
                        std_trend).clip(-3, 3).fillna(0.0)
        regime_filter = self.calculate_cs_zscore(regime_score)
        regime_score = regime_score.mask(
            regime_filter.abs() < self.regime_filter_threshold, 0)

        # Interaction
        interaction_alpha = liquidation_shock * regime_score

        # 6. Hawkes-like Decay (Memory)
        final_signal = interaction_alpha.ewm(
            halflife=self.half_life_decay, min_periods=0).mean()

        # --- STEP 7: BETA NEUTRALIZATION ---
        beta_df = factors.calc_beta_df(df_close, self.beta_lookback)
        final_signal = self.neutralize_signal(final_signal, beta_df)

        # 8. Execution Logic (Shift to avoid look-ahead)
        final_signal = final_signal.fillna(0.0)

        # 8b. Regime score conviction filter: keep only Q2 and Q3 of abs(regime_score)
        # among traded assets (non-zero final_signal) — mirrors analyze_factor_quantiles.
        # Ranks abs(regime_score) within the traded pool only; zeros out the bottom 1/3 (Q1).
        is_traded = final_signal != 0
        abs_regime_traded = regime_score.abs().where(is_traded)   # NaN for non-traded
        regime_pct = abs_regime_traded.rank(axis=1, pct=True)     # rank within traded only
        q1_mask = is_traded & (regime_pct <= (1 / 3))             # traded but in Q1
        final_signal = final_signal.where(~q1_mask, 0.0)

        # 9. Weight Normalization
        total_signal_strength = final_signal.abs().sum(axis=1).replace(0, 1.0)

        final_weights = final_signal.div(
            total_signal_strength, axis=0) * self.leverage_scale

        # 10. Build long-format score history for inspection
        def _stack(wide: pd.DataFrame, name: str) -> pd.Series:
            s = wide.stack()
            s.index.names = ['ts', 'symbol']
            s.name = name
            return s

        score_history_df = pd.concat([
            _stack(final_z_oi_chg,      'oi_z_score'),
            _stack(liquidation_shock,   'liquidation_shock'),
            _stack(regime_score,        'regime_score'),
            _stack(interaction_alpha,   'interaction_alpha'),
            _stack(final_signal,        'final_signal'),
            _stack(final_weights,       'position_qty'),
        ], axis=1).reset_index()

        return final_weights, score_history_df
