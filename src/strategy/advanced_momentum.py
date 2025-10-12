import pandas as pd
import numpy as np
from typing import Dict, Tuple
from ..core.event import SignalEvent


class AdvancedMomentumStrategy:
    """
    A strategy that scores assets based on trend strength, confirmed by Open Interest,
    and penalizes scores when funding rates suggest a crowded trade.
    """

    def __init__(self, lookback: int = 90, quantile: float = 0.2, min_volume_usd: float = 10_000_000,
                 funding_lookback: int = 14, funding_threshold: float = 0.002):
        self.lookback = lookback
        self.quantile = quantile
        self.min_volume_usd = min_volume_usd
        self.funding_lookback = funding_lookback
        self.funding_threshold = funding_threshold

    def on_rebalance(self, data: Dict[str, pd.DataFrame]) -> Tuple[Dict, Dict]:
        final_scores = {}
        score_components = {}  # Dictionary to hold the component breakdown

        for symbol, df in data.items():
            if len(df) < self.lookback + 5:
                continue

            # --- 1. Volume Filter ---
            avg_volume_usd = (df["futures_volume"] *
                              df["futures_close"]).iloc[-30:].mean()
            if avg_volume_usd < self.min_volume_usd:
                continue

            # --- 2. Calculate Core Score Components ---
            price_roc = df["futures_close"].pct_change(self.lookback).iloc[-1]
            oi_roc = df["open_interest"].pct_change(self.lookback).iloc[-1]
            if np.isnan(oi_roc):
                oi_roc = 0.0

            trend_score = price_roc * (1 + oi_roc)

            # --- 3. Apply Funding Penalty ---
            avg_funding_rate = df["funding_rate"].iloc[-self.funding_lookback:].mean()
            funding_penalty = 1.0
            if (avg_funding_rate > self.funding_threshold and trend_score > 0) or \
               (avg_funding_rate < -self.funding_threshold and trend_score < 0):
                funding_penalty = 0.25

            final_score = trend_score * funding_penalty
            final_scores[symbol] = final_score

            # --- Store all components for inspection ---
            score_components[symbol] = {
                'price_roc': price_roc,
                'oi_roc': oi_roc,
                'trend_score': trend_score,
                'avg_funding_rate': avg_funding_rate,
                'funding_penalty': funding_penalty,
                'final_score': final_score
            }

        if not final_scores:
            return {}, {}

        # --- 4. Ranking and Weighting ---
        ranked = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        signals = {}
        num_assets = len(ranked)
        long_cutoff_idx = int(num_assets * self.quantile)
        short_cutoff_idx = int(num_assets * (1 - self.quantile))
        longs = ranked[:long_cutoff_idx]
        shorts = ranked[short_cutoff_idx:]

        if longs and shorts:
            long_scores = [len(longs) - i for i in range(len(longs))]
            total_long_score = sum(long_scores)
            for i, (symbol, score) in enumerate(longs):
                signals[symbol] = {'weight': (
                    long_scores[i] / total_long_score)}

            short_scores = [len(shorts) - i for i in range(len(shorts))]
            total_short_score = sum(short_scores)
            for i, (symbol, score) in enumerate(shorts):
                signals[symbol] = {'weight': -1 *
                                   (short_scores[i] / total_short_score)}

            long_symbols = {s[0] for s in longs}
            short_symbols = {s[0] for s in shorts}
            for symbol in data.keys():
                if symbol not in long_symbols and symbol not in short_symbols:
                    signals[symbol] = {'weight': 0.0}

        return signals, score_components
