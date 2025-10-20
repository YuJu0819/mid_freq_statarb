import pandas as pd
import numpy as np
from typing import Dict, Tuple
from ..core.event import SignalEvent


class FinalStrategy:
    def __init__(self, lookback: int = 90, quantile: float = 0.2, min_volume_usd: float = 10_000_000,
                 funding_lookback: int = 14, funding_threshold: float = 0.002, funding_z_threshold=1, trend_ma_length=30):
        self.lookback = lookback
        # Default quantile used in trending markets
        self.quantile = quantile
        self.min_volume_usd = min_volume_usd
        self.funding_lookback = funding_lookback
        self.funding_threshold = funding_threshold
        self.funding_z_threshold = funding_z_threshold
        self.trend_ma_length = trend_ma_length  # Store MA length

    def on_rebalance(self, data: Dict[str, pd.DataFrame]) -> Tuple[Dict, Dict]:
        final_scores = {}
        score_components = {}

        # --- Get Current Regime ---
        # proxy_symbol = next(iter(data))
        # if data[proxy_symbol].empty or len(data[proxy_symbol]) < self.trend_ma_length:
        #     # Not enough data for MA or regime calculation
        #     return {}, {}

        # latest_market_data = data[proxy_symbol].iloc[-1]
        # current_trend_regime = latest_market_data.get(
        #     'trend_regime', 'Unknown')
        # current_adx = latest_market_data.get(
        #     'adx', 0)  # Get ADX value if available

        # # Calculate Trend Direction using MA
        # btc_closes = data[proxy_symbol]['futures_close']
        # current_btc_price = btc_closes.iloc[-1]
        # btc_ma = btc_closes.rolling(
        #     window=self.trend_ma_length).mean().iloc[-1]
        # is_uptrend = current_btc_price > btc_ma

        # # --- Set Quantile AND Exposure Scale based on regime AND direction ---
        # # current_quantile = 0.1 if current_trend_regime == 'Ranging' else self.quantile
        # current_quantile = self.quantile
        # # Determine exposure scales
        # if current_trend_regime == 'Strong Trend':
        #     if is_uptrend:
        #         long_weight_scale = 0.6  # Tilt Long
        #         short_weight_scale = 0.4
        #         trend_direction = "UPTREND"
        #     else:
        #         long_weight_scale = 0.4  # Tilt Short
        #         short_weight_scale = 0.6
        #         trend_direction = "DOWNTREND"
        #     print(
        #         f"Strong Trend ({trend_direction}) on {latest_market_data['ts']}. Tilting exposure: {long_weight_scale*100:.0f}%L / {short_weight_scale*100:.0f}%S. Quantile: {current_quantile}")
        # else:  # Weak Trend or Ranging
        #     long_weight_scale = 0.5  # Neutral
        #     short_weight_scale = 0.5
        #     trend_direction = "NEUTRAL/WEAK"
        #     print(
        #         f"{current_trend_regime} ({trend_direction}) on {latest_market_data['ts']}. Neutral exposure: {long_weight_scale*100:.0f}%L / {short_weight_scale*100:.0f}%S. Quantile: {current_quantile}")
        # --- Original Calculations ---
        for symbol, df in data.items():
            if len(df) < self.lookback + 5:
                continue

            price_roc = df["futures_close"].pct_change(self.lookback).iloc[-1]
            oi_roc = df["open_interest"].pct_change(self.lookback).iloc[-1]
            if np.isnan(oi_roc):
                oi_roc = 0.0
            # Ensure denominator is not zero or NaN
            close_price = df['futures_close'].iloc[-1]
            if close_price == 0 or np.isnan(close_price):
                close_price = 1e-12  # Avoid division by zero
            basis_momentum = df["basis"].diff(
                self.lookback).iloc[-1] / close_price

            avg_volume_ratio = df["volume_ratio"].rolling(
                self.lookback).mean().iloc[-1]

            trend_score = price_roc * (1 + 2 * oi_roc)
            # Add checks for NaN/inf in sentiment score components
            if np.isnan(basis_momentum) or np.isnan(avg_volume_ratio) or np.isinf(basis_momentum) or np.isinf(avg_volume_ratio):
                sentiment_score = 0.0  # Default to zero if components are invalid
            else:
                sentiment_score = basis_momentum * avg_volume_ratio * 5

            combined_score = trend_score + sentiment_score

            # --- MODIFICATION: Use Funding Rate Z-Score ---
            funding_rates = df["funding_rate"].iloc[-self.funding_lookback:]
            rolling_mean_fr = funding_rates.mean()
            rolling_std_fr = funding_rates.std()
            current_funding_rate = df["funding_rate"].iloc[-1]

            funding_z_score = 0.0
            # Avoid division by zero if std dev is zero
            if rolling_std_fr is not None and rolling_std_fr != 0 and not np.isnan(rolling_std_fr):
                funding_z_score = (current_funding_rate -
                                   rolling_mean_fr) / rolling_std_fr

            funding_penalty = 1.0
            # Apply penalty based on Z-score threshold
            if not np.isnan(combined_score) and not np.isinf(combined_score):
                # Penalty if funding is extremely positive and score is positive
                if (funding_z_score > self.funding_z_threshold and combined_score > 0):
                    funding_penalty = 1.5  # Or some other penalty factor
                # Penalty if funding is extremely negative and score is negative
                elif (funding_z_score < -self.funding_z_threshold and combined_score < 0):
                    funding_penalty = 1.5  # Or some other penalty factor
            else:
                combined_score = 0.0

            final_score = combined_score * funding_penalty
            # Ensure final score is not NaN before storing
            if np.isnan(final_score):
                final_score = 0.0

            final_scores[symbol] = final_score

            # Store components, ensuring they are serializable (replace NaN with None or 0)
            score_components[symbol] = {
                'price_roc': price_roc if not np.isnan(price_roc) else 0.0,
                'oi_roc': oi_roc if not np.isnan(oi_roc) else 0.0,
                'trend_score': trend_score if not np.isnan(trend_score) else 0.0,
                'basis_momentum': basis_momentum if not np.isnan(basis_momentum) else 0.0,
                'avg_volume_ratio': avg_volume_ratio if not np.isnan(avg_volume_ratio) else 0.0,
                'sentiment_score': sentiment_score if not np.isnan(sentiment_score) else 0.0,
                'funding_penalty': funding_penalty,
                'funding_z_score': funding_z_score if not np.isnan(funding_z_score) else 0.0,
                'final_score': final_score
            }

        if not final_scores:
            return {}, {}

        ranked = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        signals = {}
        num_assets = len(ranked)

        # --- MODIFICATION: Use dynamic quantile ---
        long_cutoff_idx = int(num_assets * current_quantile)
        short_cutoff_idx = int(num_assets * (1 - current_quantile))

        longs = ranked[:long_cutoff_idx]
        shorts = ranked[short_cutoff_idx:]

        if not longs or not shorts:
            # If no longs or shorts selected (e.g., quantile too small), flatten all
            for symbol in data.keys():
                signals[symbol] = SignalEvent(symbol=symbol, weight=0.0)
            return signals, score_components

        # --- Original Weighting Logic ---
        long_scores = [len(longs) - i for i in range(len(longs))]
        total_long_score = sum(long_scores) if sum(
            long_scores) != 0 else 1  # Avoid division by zero
        for i, (symbol, score) in enumerate(longs):
            signals[symbol] = SignalEvent(
                symbol=symbol, weight=0.5 * (long_scores[i] / total_long_score))

        short_scores = [i + 1 for i in range(len(shorts))]
        total_short_score = sum(short_scores) if sum(
            short_scores) != 0 else 1  # Avoid division by zero
        for i, (symbol, score) in enumerate(shorts):
            signals[symbol] = SignalEvent(
                symbol=symbol, weight=-0.5 * (short_scores[i] / total_short_score))

        long_symbols = {s[0] for s in longs}
        short_symbols = {s[0] for s in shorts}
        for symbol in data.keys():
            if symbol not in long_symbols and symbol not in short_symbols:
                signals[symbol] = SignalEvent(symbol=symbol, weight=0.0)

        return signals, score_components
