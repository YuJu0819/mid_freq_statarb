import pandas as pd
import numpy as np
from typing import Dict, Tuple
from ..core.event import SignalEvent


class FinalStrategy:
    def __init__(self, lookback: int = 90, quantile: float = 0.2, min_volume_usd: float = 10_000_000,
                 funding_lookback: int = 14, funding_threshold: float = 0.002, funding_z_threshold=1,
                 trend_ma_length=30,
                 smooth_lookback: int = 10,  # New: Lookback for smoothing momentum factors
                 vol_lookback: int = 30,  # New: Lookback for volatility calculation
                 # New: How much volatility adjusts the score (0 to 1)
                 vol_adj_factor: float = 0.5
                 ):
        self.lookback = lookback
        self.quantile = quantile
        self.min_volume_usd = min_volume_usd
        self.funding_lookback = funding_lookback
        # Keep original threshold if needed elsewhere
        self.funding_threshold = funding_threshold
        self.funding_z_threshold = funding_z_threshold
        self.trend_ma_length = trend_ma_length
        self.smooth_lookback = smooth_lookback  # Store smooth lookback
        self.vol_lookback = vol_lookback       # Store volatility lookback
        self.vol_adj_factor = vol_adj_factor   # Store volatility adjustment factor

    def on_rebalance(self, data: Dict[str, pd.DataFrame]) -> Tuple[Dict, Dict]:
        final_scores = {}
        score_components = {}
        intermediate_calcs = {}  # Store intermediate calculations

        # --- Regime logic is commented out as per user's provided code ---
        current_quantile = self.quantile
        long_weight_scale = 0.5  # Default exposure
        short_weight_scale = 0.5  # Default exposure

        # --- Calculations for each asset ---
        for symbol, df in data.items():
            required_len = max(self.lookback,
                               self.vol_lookback, self.smooth_lookback) + 5
            if len(df) < required_len:
                continue

            # --- Momentum Calculations (Raw) ---
            price_roc_raw = df["futures_close"].pct_change(self.lookback)
            oi_roc_raw = df["open_interest"].pct_change(self.lookback)
            close_price = df['futures_close'].iloc[-1]
            if close_price == 0 or np.isnan(close_price):
                close_price = 1e-12
            basis_raw = df["basis"] / close_price
            basis_momentum_raw = basis_raw.diff(self.lookback)
            avg_volume_ratio_raw = df["volume_ratio"].rolling(
                self.lookback).mean()

            # --- MODIFICATION 1: Smooth Momentum Factors ---
            price_roc = price_roc_raw.rolling(
                self.smooth_lookback).mean().iloc[-1]
            oi_roc = oi_roc_raw.rolling(self.smooth_lookback).mean().iloc[-1]
            if np.isnan(oi_roc):
                oi_roc = 0.0
            basis_momentum = basis_momentum_raw.rolling(
                self.smooth_lookback).mean().iloc[-1]
            # Use the latest rolling average value
            avg_volume_ratio = avg_volume_ratio_raw.iloc[-1]

            # --- Trend Score ---
            trend_score = price_roc * (1 + 2 * oi_roc)

            # --- Sentiment Score ---
            if np.isnan(basis_momentum) or np.isnan(avg_volume_ratio) or np.isinf(basis_momentum) or np.isinf(avg_volume_ratio):
                sentiment_score = 0.0
            else:
                sentiment_score = basis_momentum * avg_volume_ratio * 5

            combined_score = trend_score + sentiment_score

            # --- MODIFICATION 2: Calculate Volatility for Adjustment ---
            daily_returns = df["futures_close"].pct_change()
            volatility = daily_returns.rolling(
                self.vol_lookback).std().iloc[-1]

            # --- Funding Rate Z-Score Penalty ---
            funding_rates = df["funding_rate"].iloc[-self.funding_lookback:]
            rolling_mean_fr = funding_rates.mean()
            rolling_std_fr = funding_rates.std()
            current_funding_rate = df["funding_rate"].iloc[-1]
            funding_z_score = 0.0
            if rolling_std_fr is not None and rolling_std_fr != 0 and not np.isnan(rolling_std_fr):
                funding_z_score = (current_funding_rate -
                                   rolling_mean_fr) / rolling_std_fr

            funding_penalty = 1.0
            if not np.isnan(combined_score) and not np.isinf(combined_score):
                if (funding_z_score > self.funding_z_threshold and combined_score > 0):
                    funding_penalty = 1.5
                elif (funding_z_score < -self.funding_z_threshold and combined_score < 0):
                    funding_penalty = 1.5
            else:
                combined_score = 0.0

            intermediate_calcs[symbol] = {
                'combined_score': combined_score if not np.isnan(combined_score) else 0.0,
                'volatility': volatility if not np.isnan(volatility) else 0.0,
                'funding_penalty': funding_penalty,
                # Store original smoothed components
                'price_roc': price_roc, 'oi_roc': oi_roc, 'trend_score': trend_score,
                'basis_momentum': basis_momentum, 'avg_volume_ratio': avg_volume_ratio,
                'sentiment_score': sentiment_score, 'funding_z_score': funding_z_score
            }

        if not intermediate_calcs:
            return {}, {}

        # --- Cross-sectional Volatility Normalization ---
        volatilities = pd.Series(
            {sym: calc['volatility'] for sym, calc in intermediate_calcs.items()})
        vol_ranks = volatilities.rank(pct=True).fillna(0.5)

        # --- Calculate Final Adjusted Scores ---
        for symbol, calcs in intermediate_calcs.items():
            combined_score = calcs['combined_score']
            funding_penalty = calcs['funding_penalty']
            normalized_vol = vol_ranks.get(symbol, 0.5)

            # --- MODIFICATION 2 (cont.): Apply Volatility Adjustment ---
            vol_adjustment = max(0, (1 - normalized_vol * self.vol_adj_factor))
            adjusted_score = combined_score * vol_adjustment
            final_score = adjusted_score * funding_penalty

            final_scores[symbol] = final_score

            # Store final components
            score_components[symbol] = {
                'price_roc': calcs['price_roc'] if not np.isnan(calcs['price_roc']) else 0.0,
                'oi_roc': calcs['oi_roc'] if not np.isnan(calcs['oi_roc']) else 0.0,
                'trend_score': calcs['trend_score'] if not np.isnan(calcs['trend_score']) else 0.0,
                'basis_momentum': calcs['basis_momentum'] if not np.isnan(calcs['basis_momentum']) else 0.0,
                'avg_volume_ratio': calcs['avg_volume_ratio'] if not np.isnan(calcs['avg_volume_ratio']) else 0.0,
                'sentiment_score': calcs['sentiment_score'] if not np.isnan(calcs['sentiment_score']) else 0.0,
                'funding_penalty': funding_penalty,
                'funding_z_score': calcs['funding_z_score'] if not np.isnan(calcs['funding_z_score']) else 0.0,
                'volatility': calcs['volatility'] if not np.isnan(calcs['volatility']) else 0.0,
                'vol_adj_factor': vol_adjustment,
                'final_score_unadjusted': combined_score * funding_penalty if not np.isnan(combined_score) else 0.0,
                'final_score': final_score if not np.isnan(final_score) else 0.0
            }

        if not final_scores:
            return {}, {}

        # --- Ranking and Signal Generation ---
        ranked = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        signals = {}
        num_assets = len(ranked)

        long_cutoff_idx = int(num_assets * current_quantile)
        short_cutoff_idx = int(num_assets * (1 - current_quantile))

        longs = ranked[:long_cutoff_idx]
        shorts = ranked[short_cutoff_idx:]

        if not longs or not shorts:
            for symbol in data.keys():
                signals[symbol] = SignalEvent(symbol=symbol, weight=0.0)
            return signals, score_components

        # --- Weighting Logic ---
        long_scores = [len(longs) - i for i in range(len(longs))]
        total_long_score = sum(long_scores) if sum(long_scores) != 0 else 1
        for i, (symbol, score) in enumerate(longs):
            signals[symbol] = SignalEvent(
                symbol=symbol, weight=long_weight_scale * (long_scores[i] / total_long_score))

        short_scores = [i + 1 for i in range(len(shorts))]
        total_short_score = sum(short_scores) if sum(short_scores) != 0 else 1
        for i, (symbol, score) in enumerate(shorts):
            signals[symbol] = SignalEvent(
                symbol=symbol, weight=-short_weight_scale * (short_scores[i] / total_short_score))

        long_symbols = {s[0] for s in longs}
        short_symbols = {s[0] for s in shorts}
        for symbol in data.keys():
            if symbol not in long_symbols and symbol not in short_symbols:
                signals[symbol] = SignalEvent(symbol=symbol, weight=0.0)

        return signals, score_components
