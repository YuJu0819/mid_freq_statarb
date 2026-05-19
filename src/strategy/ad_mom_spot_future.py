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

    def neutralize_signal(
        self, signal_df: pd.DataFrame, beta_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Removes Beta exposure from the signals using Cross-Sectional Regression.
        CRITICAL: Masks 0s and NaNs so we only neutralize the ACTIVE bets.

        Phase-7 refactor: this body was a byte-identical copy of
        LiquidationReversalStrategy.neutralize_signal — both now delegate
        to src/alpha/neutralize.py. The method is preserved with the same
        signature so every existing self.neutralize_signal(...) call site
        keeps working.
        """
        from ..alpha.neutralize import _neutralize
        return _neutralize(signal_df, beta_df)

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

    def _compute_signals_for_symbols(
        self,
        all_data: Dict[str, pd.DataFrame],
        active_symbols: List[str],
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Core signal computation using ONLY `active_symbols` for CS operations.

        `all_data` still contains the full history of all symbols so the TS
        rolling window can warm up correctly — pre-epoch price history is
        used for lookback but CS normalization (vol rank, z-score, final rank)
        only sees `active_symbols`.  This keeps the cross-sectional distribution
        stable and prevents universe-size changes from contaminating scores.

        Returns (weights_wide, score_history_df) indexed over the full date
        range of the data.  The caller slices to the epoch's active dates.
        """
        # Build wide frames from active_symbols only
        all_ts_set: set = set()
        for sym in active_symbols:
            df = all_data.get(sym)
            if df is not None and not df.empty:
                all_ts_set.update(df['ts'].tolist())
        all_ts = pd.DatetimeIndex(sorted(all_ts_set))

        def make_wide(col, fill_val=np.nan):
            d = {sym: all_data[sym].set_index('ts')[col]
                 for sym in active_symbols
                 if sym in all_data and col in all_data[sym].columns}
            return pd.DataFrame(d).reindex(all_ts).ffill().fillna(fill_val)

        closes_wide = make_wide('futures_close')
        oi_wide = make_wide('open_interest')
        basis_wide = make_wide('basis')
        vol_ratio_wide = make_wide('volume_ratio', 0.0)
        funding_wide = make_wide('funding_rate', 0.0)

        # Factor calculations
        price_roc = factors.calc_price_mom(
            closes_wide, self.lookback, self.smooth_lookback).fillna(0.0)
        oi_roc = factors.calc_oi_mom(
            oi_wide, self.lookback, self.smooth_lookback).fillna(0.0)
        basis_mom = factors.calc_basis_mom(
            basis_wide, closes_wide, self.lookback, self.smooth_lookback).fillna(0.0)
        vol_ratio_sig = factors.calc_vol_ratio_signal(
            vol_ratio_wide, self.lookback, self.lookback).fillna(1.0)

        trend_score = price_roc * (1 + 2 * oi_roc)
        valid_sentiment = ~(np.isinf(basis_mom) | np.isinf(vol_ratio_sig))
        sentiment_score = (basis_mom * vol_ratio_sig *
                           5).where(valid_sentiment, 0.0)
        combined_score = trend_score + sentiment_score

        volatility = factors.calc_volatility(
            closes_wide, self.vol_lookback).fillna(0.0)
        funding_z = factors.calc_funding_zscore(
            funding_wide, self.funding_lookback).fillna(0.0)

        # CS operations — only active_symbols are present, so distribution is clean
        active_volatility = volatility.replace(0.0, np.nan)
        vol_ranks = active_volatility.rank(axis=1, pct=True).fillna(0.5)
        vol_adjustment = (1 - vol_ranks * self.vol_adj_factor).clip(lower=0.0)

        trend_z = factors.calc_cs_zscore(trend_score).fillna(0.0)

        final_score = combined_score * vol_adjustment

        # Beta-neutralize
        beta_df = factors.calc_beta_df(closes_wide, 60)
        final_score = self.neutralize_signal(final_score, beta_df)

        # Rank-based selection & weighting
        int_ranks = final_score.rank(axis=1, method='first')
        n_active = final_score.notna().sum(axis=1).replace(0, 1)
        pct_ranks = int_ranks.div(n_active, axis=0)

        long_mask = pct_ranks > (1 - self.quantile)
        short_mask = pct_ranks <= self.quantile

        if self.conviction_top_fraction is not None:
            selected_mask = long_mask | short_mask
            abs_trend_selected = trend_score.abs().where(selected_mask)
            abs_trend_pct = abs_trend_selected.rank(axis=1, pct=True)
            conviction_mask = abs_trend_pct > (
                1.0 - self.conviction_top_fraction)
            long_mask = long_mask & conviction_mask
            short_mask = short_mask & conviction_mask

        long_rank_scores = int_ranks.where(long_mask, 0.0)
        long_rank_sum = long_rank_scores.sum(axis=1).replace(0, 1.0)
        weights_long = long_rank_scores.div(long_rank_sum, axis=0) * 0.5

        short_rank_scores = int_ranks.rsub(
            n_active + 1, axis=0).where(short_mask, 0.0)
        short_rank_sum = short_rank_scores.sum(axis=1).replace(0, 1.0)
        weights_short = -(short_rank_scores.div(short_rank_sum, axis=0)) * 0.5

        weights = (weights_long.fillna(0.0) + weights_short.fillna(0.0)).clip(
            -MAX_WEIGHT_PER_ASSET, MAX_WEIGHT_PER_ASSET)

        # Preserve the pre-mask weights for downstream EBM consumption.
        # Two parallel weight matrices from this point forward:
        #   - `weights`          → post quality-mask (Q1 zeroed) → traded / PnL
        #   - `weights_unmasked` → full signal preserved          → EBM panel
        # The quality mask reflects a TRADING decision (skip weakest-signal
        # bucket to reduce noise / cost). For ML, we want the EBM to see the
        # full signal distribution so it can learn the value of all buckets.
        weights_unmasked = weights.copy()

        # Mask Q1 of trend score among traded assets — keep only Q2 and Q3.
        # Uses abs(trend_score) ranked within the traded set, which exactly
        # matches the reporting's analyze_factor_quantiles bucketing logic
        # (reporting does: factor_wide.abs() → rank within is_traded).
        # Q1 = weakest signal magnitude → zero out; Q2+Q3 = kept.
        is_traded = weights != 0
        abs_trend_traded = volatility.abs().where(is_traded)   # NaN for non-traded
        abs_trend_pct = abs_trend_traded.rank(axis=1, pct=True)
        q1_mask = is_traded & (abs_trend_pct >= (1.0 / 3))
        weights = weights.where(~q1_mask, 0.0)

        # Build long-format score history (uses TRADED weights so factor
        # quantile analysis reflects what the strategy actually held).
        def _stack(wide: pd.DataFrame, name: str) -> pd.Series:
            s = wide.stack()
            s.index.names = ['ts', 'symbol']
            s.name = name
            return s

        score_history_df = pd.concat([
            _stack(final_score,     'final_score'),
            _stack(trend_score,     'trend_score'),
            _stack(sentiment_score, 'sentiment_score'),
            _stack(funding_z,       'funding_z_score'),
            _stack(volatility,      'volatility'),
            _stack(price_roc,       'price_roc'),
            _stack(weights,         'position_qty'),
            _stack(closes_wide,     'close_price'),
        ], axis=1).reset_index()

        return weights, weights_unmasked, score_history_df

    def generate_all_signals(self, data: Dict[str, pd.DataFrame], epoch_mask_df=None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        When epoch_mask_df is None: single-pass computation over all symbols.

        When epoch_mask_df is provided: per-epoch computation (same pattern as
        LiquidationReversalStrategy).  For each distinct epoch, signals are
        computed using only that epoch's active symbols so the CS distribution
        stays consistent within each epoch.  Full price history is used for TS
        rolling warmup.  Only the epoch's active date rows are kept, then
        results are concatenated and re-indexed to the full date range.

        Returns:
            weights_df: DataFrame (Index=Time, Columns=Assets)
            score_history_df: DataFrame (Long format for reporting)
        """
        if epoch_mask_df is None or epoch_mask_df.empty:
            active_symbols = [sym for sym, df in data.items() if not df.empty]
            return self._compute_signals_for_symbols(data, active_symbols)

        # --- Detect epoch boundaries ---
        active_col_sets = epoch_mask_df.fillna(False).apply(
            lambda row: frozenset(row.index[row]), axis=1
        )
        epochs: list = []
        prev_set = None
        for ts, sym_set in active_col_sets.items():
            if sym_set != prev_set:
                epochs.append((ts, sym_set))
                prev_set = sym_set

        all_dates = epoch_mask_df.index
        epoch_date_ranges = []
        for i, (ep_start, sym_set) in enumerate(epochs):
            ep_end = epochs[i + 1][0] - \
                pd.Timedelta(days=1) if i + 1 < len(epochs) else all_dates[-1]
            epoch_date_ranges.append((ep_start, ep_end, list(sym_set)))

        # --- Per-epoch signal computation ---
        # Collect both masked (for PnL) and unmasked (for EBM) weight slices
        # per epoch, then concatenate independently.
        weight_slices = []
        weight_unmasked_slices = []
        score_slices = []

        for ep_start, ep_end, active_symbols in epoch_date_ranges:
            if not active_symbols:
                continue

            print(f"  [Momentum] Epoch {ep_start.date()} → {ep_end.date()} "
                  f"({len(active_symbols)} symbols)")

            weights_ep, weights_unmasked_ep, scores_ep = \
                self._compute_signals_for_symbols(data, active_symbols)

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

        # Concatenate and reindex to full date range — both versions
        def _assemble(slices):
            out = (pd.concat(slices)
                   .reindex(all_dates)
                   .fillna(0.0))
            return out.reindex(columns=epoch_mask_df.columns, fill_value=0.0)

        final_weights = _assemble(weight_slices)
        final_weights_unmasked = _assemble(weight_unmasked_slices)
        final_scores = pd.concat(score_slices, ignore_index=True)

        return final_weights, final_weights_unmasked, final_scores
