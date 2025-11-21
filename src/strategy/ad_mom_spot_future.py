import pandas as pd
import numpy as np
from typing import Dict, Tuple
from ..core.event import SignalEvent
from .. import factors


class FinalStrategy:
    def __init__(self, lookback: int = 90, quantile: float = 0.2, min_volume_usd: float = 10_000_000,
                 funding_lookback: int = 14, funding_threshold: float = 0.002, funding_z_threshold=1,
                 trend_ma_length=30,
                 smooth_lookback: int = 10,
                 vol_lookback: int = 30,
                 vol_adj_factor: float = 0.5,
                 inverse_in_weak_regime: bool = True
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

    def on_rebalance(self, data: Dict[str, pd.DataFrame]) -> Tuple[Dict, Dict]:
        # --- 0. Handle Empty Data ---
        if not data:
            return {}, {}

        # V-- NEW: SUNDAY CHECK TO AVOID MONDAY LOSS --V
        # # Check the last timestamp of the first available asset
        # sample_df = next(iter(data.values()))
        # if not sample_df.empty:
        #     current_ts_ms = sample_df['ts'].iloc[-1]
        #     current_date = pd.to_datetime(current_ts_ms, unit='ms')
        #     # Monday=0, ..., Saturday=5, Sunday=6.
        #     if current_date.weekday() in [4, 2, 6]:
        #         signals = {sym: SignalEvent(symbol=sym, weight=0.0) for sym in data.keys()}
        #         return signals, {}
        # ^-- END SUNDAY CHECK --^

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
            tail_df = df.iloc[-required_history:].copy()
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

        combined_score = trend_score + sentiment_score

        # Volatility
        volatility = factors.calc_volatility(closes_wide, self.vol_lookback)
        volatility = volatility.fillna(0.0)

        # Funding Z-Score
        funding_z = factors.calc_funding_zscore(
            funding_wide, self.funding_lookback)
        funding_z = funding_z.fillna(0.0)

        # --- 4. Vectorized Adjustments (Funding Penalty & Regimes) ---

        # Funding Penalty Logic
        funding_penalty = pd.DataFrame(
            1.0, index=combined_score.index, columns=combined_score.columns)

        # Boost Condition (1.5): (Z > Th and Score > 0) OR (Z < -Th and Score < 0)
        boost_mask = (
            ((funding_z > self.funding_z_threshold) & (combined_score > 0)) |
            ((funding_z < -self.funding_z_threshold) & (combined_score < 0))
        )

        # Kill Condition (0.0):
        # Reference: (funding_z < -self.funding_z_threshold * 1.3 and combined_score > 0) or
        #            (funding_z > self.funding_z_threshold * 1.3 and combined_score > 0)
        kill_mask = (
            ((funding_z < -self.funding_z_threshold * 1.3) & (combined_score > 0)) |
            ((funding_z > self.funding_z_threshold * 1.3) & (combined_score > 0))
        )

        funding_penalty[boost_mask] = 1.5
        funding_penalty[kill_mask] = 0.0

        active_volatility = volatility.replace(0.0, np.nan)

        vol_ranks = active_volatility.rank(axis=1, pct=True).fillna(0.5)

        # Now continue as normal...
        vol_adjustment = (1 - vol_ranks * self.vol_adj_factor).clip(lower=0.0)

        # Regime Inversion Logic
        # (Matches reference: defaults to 1.0, logic commented out)
        regime_multiplier = pd.DataFrame(
            1.0, index=trend_score.index, columns=trend_score.columns)
        # if self.inverse_in_weak_regime:
        #     # Example vectorized logic if you enable it later:
        #     # vol_regime_wide = pd.DataFrame(vol_regime_dict).ffill()
        #     # weak_mask = vol_regime_wide.isin(["Low Volatility"])
        #     # regime_multiplier[weak_mask] = 0.0
        #     pass

        # Final Score Calculation
        adjusted_score = combined_score * vol_adjustment * regime_multiplier
        final_score = adjusted_score * funding_penalty

        # --- 5. Signal Generation (Current Timestamp) ---
        current_scores = final_score.iloc[-1].dropna()

        if current_scores.empty:
            return {}, {}

        # Ranking
        ranked_scores = current_scores.sort_values(ascending=False)
        symbols = ranked_scores.index.tolist()
        n_assets = len(symbols)

        long_cutoff = int(n_assets * self.quantile)
        short_cutoff = int(n_assets * (1 - self.quantile))

        longs = symbols[:long_cutoff]
        shorts = symbols[short_cutoff:]

        signals = {}
        long_weight_scale = 0.5
        short_weight_scale = 0.5

        # Long Weights
        if longs:
            long_denom = sum(len(longs) - i for i in range(len(longs))) or 1
            for i, sym in enumerate(longs):
                w = long_weight_scale * (len(longs) - i) / long_denom
                signals[sym] = SignalEvent(symbol=sym, weight=w)

        # Short Weights
        if shorts:
            short_denom = sum(i + 1 for i in range(len(shorts))) or 1
            for i, sym in enumerate(shorts):
                w = -short_weight_scale * (i + 1) / short_denom
                signals[sym] = SignalEvent(symbol=sym, weight=w)

        # Zero others
        active_syms = set(longs + shorts)
        for sym in data.keys():
            if sym not in active_syms:
                signals[sym] = SignalEvent(symbol=sym, weight=0.0)

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

        return signals, score_components
