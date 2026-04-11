import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional
from ..core.event import SignalEvent
from .. import factors
from collections import defaultdict
MAX_WEIGHT_PER_ASSET = 0.1  # 10% Cap


class FinalStrategy:
    def __init__(self, lookback: int = 90, quantile: float = 0.2, min_volume_usd: float = 10_000_000,
                 funding_lookback: int = 14, funding_threshold: float = 0.002, funding_z_threshold=1,
                 trend_ma_length=30,
                 smooth_lookback: int = 10,
                 vol_lookback: int = 30,
                 vol_adj_factor: float = 0.5,
                 inverse_in_weak_regime: bool = True,
                 factor_mask_config: Optional[Dict] = None,
                 conviction_top_fraction: Optional[float] = None,
                 ):
        self.lookback = lookback
        self.quantile = quantile
        self.min_volume_usd = min_volume_usd
        self.funding_lookback = funding_lookback
        self.funding_threshold = funding_threshold
        self.funding_z_threshold = funding_z_threshold
        self.trend_ma_length = trend_ma_length
        self.smooth_lookback = smooth_lookback
        self.vol_lookback = vol_lookback
        self.vol_adj_factor = vol_adj_factor
        self.inverse_in_weak_regime = inverse_in_weak_regime
        self.factor_mask_config = factor_mask_config
        self.conviction_top_fraction = conviction_top_fraction

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

    def on_rebalance(self, data: Dict[str, pd.DataFrame]) -> Tuple[Dict, Dict]:
        # --- 0. Handle Empty Data ---
        if not data:
            return {}, {}

        # --- 2. Construct Aligned Wide DataFrames ---
        # We only need the recent history required for calculation
        required_history = max(
            self.lookback, self.vol_lookback,) + self.smooth_lookback + 10

        # Extract series for each component
        closes_dict = {}
        oi_dict = {}
        basis_dict = {}
        vol_ratio_dict = {}
        funding_dict = {}
        trend_regime_dict = {}
        skew_regime_dict = {}
        vol_regime_dict = {}

        for sym, df in data.items():
            if len(df) < required_history:
                continue
            # Take tail to speed up construction
            tail_df = df.iloc[-required_history * 2:].copy()
            tail_df.set_index('ts', inplace=True)
            closes_dict[sym] = tail_df['futures_close']
            oi_dict[sym] = tail_df['open_interest']
            basis_dict[sym] = tail_df['basis']
            vol_ratio_dict[sym] = tail_df['volume_ratio']
            funding_dict[sym] = tail_df['funding_rate']

            # Regimes (Optional / Reporting)
            if 'trend_regime' in tail_df.columns:
                trend_regime_dict[sym] = tail_df['trend_regime']
            if 'skew_regime' in tail_df.columns:
                skew_regime_dict[sym] = tail_df['skew_regime']
            if 'volatility_regime' in tail_df.columns:
                vol_regime_dict[sym] = tail_df['volatility_regime']

        if not closes_dict:
            return {}, {}

        # Convert to DataFrame (Index=Time, Columns=Symbols)
        closes_wide = pd.DataFrame(closes_dict).ffill()
        oi_wide = pd.DataFrame(oi_dict).ffill()
        basis_wide = pd.DataFrame(basis_dict).ffill()
        vol_ratio_wide = pd.DataFrame(vol_ratio_dict).fillna(0.0)
        funding_wide = pd.DataFrame(funding_dict).fillna(0.0)

        # --- 3. Vectorized Factor Calculations (using factors.py) ---

        # Momentum
        price_roc = factors.calc_price_mom(
            closes_wide, self.lookback, self.smooth_lookback)
        oi_roc = factors.calc_oi_mom(
            oi_wide, self.lookback, self.smooth_lookback)
        basis_mom = factors.calc_basis_mom(
            basis_wide, closes_wide, self.lookback, self.smooth_lookback)
        vol_ratio_sig = factors.calc_vol_ratio_signal(
            vol_ratio_wide, self.lookback, self.lookback)

        # Fill NaNs
        price_roc = price_roc.fillna(0.0)
        oi_roc = oi_roc.fillna(0.0)
        basis_mom = basis_mom.fillna(0.0)
        vol_ratio_sig = vol_ratio_sig.fillna(1.0)

        # Scores
        trend_score = price_roc * (1 + 2 * oi_roc)

        # Sentiment Score logic: if basis or vol_ratio is invalid, 0. Else product * 5
        valid_sentiment = ~(np.isinf(basis_mom) | np.isinf(vol_ratio_sig))
        sentiment_score = (basis_mom * vol_ratio_sig *
                           5).where(valid_sentiment, 0.0)
        sentiment_score = self.neutralize_signal(sentiment_score, trend_score)
        combined_score = trend_score + sentiment_score

        # combined_score = beta_df
        # breakpoint()
        # Volatility
        volatility = factors.calc_volatility(closes_wide, self.vol_lookback)
        volatility = volatility.fillna(0.0)
        # Funding Z-Score
        funding_z = factors.calc_funding_zscore(
            funding_wide, self.funding_lookback)
        funding_z = funding_z.fillna(0.0)

        sent_z = factors.calc_cs_zscore(sentiment_score)
        sent_z = sent_z.fillna(0)
        # --- 4. Vectorized Adjustments (Funding Penalty & Regimes) ---

        # Funding Penalty Logicç
        funding_penalty = pd.DataFrame(
            1.0, index=combined_score.index, columns=combined_score.columns)

        # Boost Condition (1.5): (Z > Th and Score > 0) OR (Z < -Th and Score < 0)
        boost_mask = (
            ((funding_z > self.funding_z_threshold) & (combined_score > 0)) |
            ((funding_z < -self.funding_z_threshold) & (combined_score < 0))
        )

        kill_mask = (
            ((funding_z < -self.funding_z_threshold * 2) & (combined_score > 0)) |
            ((funding_z > self.funding_z_threshold * 2) & (combined_score < 0)) |
            (funding_z.abs() < self.funding_z_threshold * 0.1)
        )

        funding_penalty[boost_mask] = 1.5
        funding_penalty[kill_mask] = 0.5

        sent_penalty = pd.DataFrame(
            1.0, index=combined_score.index, columns=combined_score.columns)

        boost_mask = (
            (abs(sent_z) < 0.25)
        )
        # sent_penalty[boost_mask] = 0.5

        active_volatility = volatility.replace(0.0, np.nan)

        vol_ranks = active_volatility.rank(axis=1, pct=True).fillna(0.5)

        # Now continue as normal...
        vol_adjustment = (1 - vol_ranks * self.vol_adj_factor).clip(lower=0.0)

        # Regime Inversion Logic
        # (Matches reference: defaults to 1.0, logic commented out)
        regime_multiplier = pd.DataFrame(
            1.0, index=trend_score.index, columns=trend_score.columns)

        # Final Score Calculation
        adjusted_score = combined_score * vol_adjustment
        # final_score = adjusted_score * funding_penalty * sent_penalty
        final_score = adjusted_score
        beta_df = factors.calc_beta_df(closes_wide, 60)
        # final_score = self.neutralize_signal(final_score, beta_df)
        # --- 5. Signal Generation (Current Timestamp) ---
        current_scores = final_score.iloc[-1].dropna()

        if current_scores.empty:
            return {}, {}

        # --- 6. Construct Score Components (For Reporting) ---
        idx = -1
        score_components = {}

        def get_val(df_wide, s):
            if s in df_wide.columns:
                return df_wide[s].iloc[idx]
            return 0.0

        for sym in current_scores.index:
            score_components[sym] = {
                'price_roc': get_val(price_roc, sym),
                'oi_roc': get_val(oi_roc, sym),
                'trend_score': get_val(trend_score, sym),
                'basis_momentum': get_val(basis_mom, sym),
                'avg_volume_ratio': get_val(vol_ratio_sig, sym),
                'sentiment_score': get_val(sentiment_score, sym),
                'funding_penalty': get_val(funding_penalty, sym),
                'funding_z_score': get_val(funding_z, sym),
                'volatility': get_val(volatility, sym),
                'vol_adj_factor': get_val(vol_adjustment, sym),
                'regime_multiplier': get_val(regime_multiplier, sym),
                'current_regime': trend_regime_dict.get(sym, pd.Series(['Unknown'])).iloc[-1] if sym in trend_regime_dict else 'Unknown',
                'final_score_unadjusted': (get_val(combined_score, sym) * get_val(funding_penalty, sym)),
                'final_score': current_scores[sym]
            }
            # --- 7. Ranking and Filtering ---
        ranked_scores = current_scores.sort_values(ascending=False)
        symbols = ranked_scores.index.tolist()

        long_cutoff = int(len(symbols) * self.quantile)
        short_cutoff = int(len(symbols) * (1 - self.quantile))

        raw_longs = symbols[:long_cutoff]
        raw_shorts = symbols[short_cutoff:]

        # --- V-- FIXED EXPERIMENTAL FILTERING --V ---

        # 1. Initialize Score Maps (Default to 0)
        # These track how many filters each asset passed.
        long_counts = defaultdict(int)
        short_counts = defaultdict(int)

        has_mask_config = bool(self.factor_mask_config)

        if has_mask_config:
            # Normalize to list
            configs = self.factor_mask_config if isinstance(
                self.factor_mask_config, list) else [self.factor_mask_config]

            for conf in configs:
                mask_factor = conf.get('factor')
                keep_quantiles = conf.get('quantiles', [])
                n_bins = conf.get('n_bins', 5)

                # Extract factor values
                factor_vals = pd.Series(
                    {s: score_components[s].get(mask_factor, 0.0) for s in symbols})

                if not factor_vals.empty:
                    # Rank
                    ranks = factor_vals.rank(method='first', pct=True)
                    q_labels = np.ceil(ranks * n_bins).astype(int)

                    # Count Matches (Boost Score)
                    for sym in raw_longs:
                        if sym in q_labels and q_labels[sym] in keep_quantiles:
                            long_counts[sym] += 1

                    for sym in raw_shorts:
                        if sym in q_labels and q_labels[sym] in keep_quantiles:
                            short_counts[sym] += 1
                else:
                    # If a factor is missing, we just skip it (don't increment, but DON'T reset)
                    pass

        # --- 8. Assign Weights ---
        signals = {}
        long_weight_scale = 0.5
        short_weight_scale = 0.5

        def assign_normalized_weights(assets, target_total_exposure, counts_map, is_long=True):
            if not assets:
                return
            raw_scores = {}
            for i, sym in enumerate(assets):
                rank_component = (len(assets) - i) if is_long else (i + 1)
                mask_multiplier = 1  # Penalize (Default)
                if has_mask_config:
                    if counts_map[sym] > 0:

                        mask_multiplier = counts_map[sym] - 0.5
                raw_scores[sym] = rank_component * mask_multiplier

            total_raw_score = sum(raw_scores.values())

            if total_raw_score == 0:
                return

            for sym, score in raw_scores.items():
                # Normalize
                weight = (score / total_raw_score) * target_total_exposure

                final_weight = min(weight, MAX_WEIGHT_PER_ASSET)

                # Set Signal
                signals[sym] = SignalEvent(
                    symbol=sym, weight=final_weight if is_long else -final_weight)

        # Apply to Longs
        assign_normalized_weights(
            raw_longs, long_weight_scale, long_counts, is_long=True)

        # Apply to Shorts
        assign_normalized_weights(
            raw_shorts, short_weight_scale, short_counts, is_long=False)

        # --- 9. Fill Zeros for Others ---
        active_syms = set(signals.keys())
        for sym in data.keys():
            if sym not in active_syms:
                signals[sym] = SignalEvent(symbol=sym, weight=0.0)

        return signals, score_components

    def generate_all_signals(self, data: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        FULLY VECTORIZED Signal Generation.
        Runs the exact logic of 'on_rebalance' but on the entire history at once.
        Returns:
            weights_df: DataFrame (Index=Time, Columns=Assets)
            score_history_df: DataFrame (Long format for reporting)
        """
        # --- 1. Align Data to Wide DataFrames ---
        # Get the union of all timestamps
        all_ts = pd.Index([])
        for df in data.values():
            if not df.empty:
                all_ts = all_ts.union(df['ts'])
        all_ts = all_ts.sort_values().unique()

        # Helper to build wide frames
        def make_wide(col, fill_val=np.nan):
            d = {sym: df.set_index('ts')[col]
                 for sym, df in data.items() if col in df.columns}
            return pd.DataFrame(d).reindex(all_ts).ffill().fillna(fill_val)

        closes_wide = make_wide('futures_close')
        oi_wide = make_wide('open_interest')
        basis_wide = make_wide('basis')
        vol_ratio_wide = make_wide('volume_ratio', 0.0)
        funding_wide = make_wide('funding_rate', 0.0)

        # --- 2. Vectorized Factor Calculations ---
        # Exact same calls as on_rebalance
        price_roc = factors.calc_price_mom(
            closes_wide, self.lookback, self.smooth_lookback).fillna(0.0)
        oi_roc = factors.calc_oi_mom(
            oi_wide, self.lookback, self.smooth_lookback).fillna(0.0)
        basis_mom = factors.calc_basis_mom(
            basis_wide, closes_wide, self.lookback, self.smooth_lookback).fillna(0.0)
        vol_ratio_sig = factors.calc_vol_ratio_signal(
            vol_ratio_wide, self.lookback, self.lookback).fillna(1.0)

        # Trend Score
        trend_score = price_roc * (1 + 2 * oi_roc)

        # Sentiment Score
        valid_sentiment = ~(np.isinf(basis_mom) | np.isinf(vol_ratio_sig))
        sentiment_score = (basis_mom * vol_ratio_sig *
                           5).where(valid_sentiment, 0.0)

        # Neutralization
        # sentiment_score = self.neutralize_signal(sentiment_score, trend_score)

        combined_score = trend_score + sentiment_score

        # Volatility & Funding
        volatility = factors.calc_volatility(
            closes_wide, self.vol_lookback).fillna(0.0)
        funding_z = factors.calc_funding_zscore(
            funding_wide, self.funding_lookback).fillna(0.0)

        # FIX: Capture the return value
        sent_z = factors.calc_cs_zscore(sentiment_score).fillna(0.0)
        trend_z = factors.calc_cs_zscore(trend_score).fillna(0.0)
        # --- 3. Vectorized Adjustments ---

        # Volatility Adjustment
        active_volatility = volatility.replace(0.0, np.nan)
        vol_ranks = active_volatility.rank(axis=1, pct=True).fillna(0.5)
        vol_adjustment = (1 - vol_ranks * self.vol_adj_factor).clip(lower=0.0)

        # Final Score
        final_score = combined_score * vol_adjustment

        # Beta-neutralize the SCORE before ranking.
        beta_df = factors.calc_beta_df(closes_wide, 60)
        final_score = self.neutralize_signal(final_score, beta_df)

        # --- 4. Vectorized Selection & Weighting ---
        # Rank: 1 = lowest score, n_assets = highest score (integer ranks)
        n_assets = final_score.shape[1]
        int_ranks = final_score.rank(axis=1, method='first')
        pct_ranks = int_ranks / n_assets  # percentile equivalent for mask thresholds

        # trend_z percentile rank (cross-sectional, same timestamp axis)
        trend_z_pct = trend_z.rank(axis=1, pct=True)

        # Masks for Top/Bottom Quantiles — intersection with trend_z eligibility.
        # Longs: top final_score quantile AND top trend_z quantile (momentum confirms).
        # Shorts: bottom final_score quantile AND bottom trend_z quantile.
        long_mask = (pct_ranks > (1 - self.quantile))
        short_mask = (pct_ranks <= self.quantile)

        # Conviction filter: mirrors the Q3 bucket in analyze_factor_quantiles.
        # Ranks abs(trend_score) across ALL selected assets (longs + shorts jointly),
        # then keeps only the top conviction_top_fraction — identical to how the
        # reporting function groups traded assets by |trend_score| before bucketing.
        if self.conviction_top_fraction is not None:
            selected_mask = long_mask | short_mask
            abs_trend_selected = trend_score.abs().where(selected_mask)  # NaN for non-selected
            # rank() naturally excludes NaNs, so only selected assets are ranked
            abs_trend_pct = abs_trend_selected.rank(axis=1, pct=True)
            conviction_mask = abs_trend_pct > (1.0 - self.conviction_top_fraction)
            long_mask  = long_mask  & conviction_mask
            short_mask = short_mask & conviction_mask

        # Initialize Weights Matrix
        weights = pd.DataFrame(0.0, index=final_score.index,
                               columns=final_score.columns)

        # Rank-proportional weights — matches on_rebalance assign_normalized_weights.
        # Every selected asset gets a non-zero weight regardless of score magnitude.

        # Longs: higher int_rank (better score) → higher weight
        long_rank_scores = int_ranks.where(long_mask, 0.0)
        long_rank_sum = long_rank_scores.sum(axis=1).replace(0, 1.0)
        weights_long = long_rank_scores.div(long_rank_sum, axis=0) * 0.5

        # Shorts: lower int_rank (worse score) → higher absolute short weight
        short_rank_scores = (n_assets + 1 - int_ranks).where(short_mask, 0.0)
        short_rank_sum = short_rank_scores.sum(axis=1).replace(0, 1.0)
        weights_short = -(short_rank_scores.div(short_rank_sum, axis=0)) * 0.5

        weights = weights_long.fillna(0.0) + weights_short.fillna(0.0)

        # Cap Weights
        weights = weights.clip(-MAX_WEIGHT_PER_ASSET, MAX_WEIGHT_PER_ASSET)

        # --- 5. Prepare Reporting Data (Long Format) ---
        # Used for analyze_factor_quantiles

        components = {
            'final_score': final_score,
            'trend_score': trend_score,
            'sentiment_score': sentiment_score,
            'funding_z_score': funding_z,
            'volatility': volatility,
            'price_roc': price_roc
        }

        # Stack DataFrames for report
        base_df = final_score.stack().reset_index()
        base_df.columns = ['ts', 'symbol', 'final_score']

        for name, df in components.items():
            if name == 'final_score':
                continue
            stacked = df.stack().reset_index()
            stacked.columns = ['ts', 'symbol', name]
            base_df = base_df.set_index(['ts', 'symbol'])
            stacked = stacked.set_index(['ts', 'symbol'])
            base_df[name] = stacked[name]
            base_df = base_df.reset_index()

        # Add positions and close prices
        stacked_w = weights.stack().reset_index()
        stacked_w.columns = ['ts', 'symbol', 'position_qty']  # Proxy

        stacked_p = closes_wide.stack().reset_index()
        stacked_p.columns = ['ts', 'symbol', 'close_price']

        base_df = base_df.set_index(['ts', 'symbol'])
        base_df['position_qty'] = stacked_w.set_index(['ts', 'symbol'])[
            'position_qty']
        base_df['close_price'] = stacked_p.set_index(['ts', 'symbol'])[
            'close_price']

        score_history_df = base_df.reset_index()

        return weights, score_history_df
