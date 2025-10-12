import pandas as pd
import numpy as np
from typing import Dict, Tuple
from ..core.event import SignalEvent


class FinalStrategy:
    def __init__(self, lookback: int = 90, quantile: float = 0.2, min_volume_usd: float = 10_000_000,
                 funding_lookback: int = 14, funding_threshold: float = 0.002):
        self.lookback = lookback
        self.quantile = quantile
        self.min_volume_usd = min_volume_usd
        self.funding_lookback = funding_lookback
        self.funding_threshold = funding_threshold

    def on_rebalance(self, data: Dict[str, pd.DataFrame]) -> Tuple[Dict, Dict]:
        final_scores = {}
        score_components = {}

        for symbol, df in data.items():
            if len(df) < self.lookback + 5:
                continue

            # avg_volume_usd = (df["futures_volume"] *
            #                   df["futures_close"]).iloc[-30:].mean()
            # if avg_volume_usd < self.min_volume_usd:
            #     continue

            price_roc = df["futures_close"].pct_change(self.lookback).iloc[-1]
            oi_roc = df["open_interest"].pct_change(self.lookback).iloc[-1]
            if np.isnan(oi_roc):
                oi_roc = 0.0
            basis_momentum = df["basis"].diff(
                30).iloc[-1] / df['futures_close'].iloc[-1]
            avg_volume_ratio = df["volume_ratio"].rolling(30).mean().iloc[-1]

            trend_score = price_roc * (1 + oi_roc)
            sentiment_score = basis_momentum * avg_volume_ratio
            combined_score = trend_score + sentiment_score

            avg_funding_rate = df["funding_rate"].iloc[-self.funding_lookback:].mean()
            funding_penalty = 1.0
            if (avg_funding_rate > self.funding_threshold and combined_score > 0) or \
               (avg_funding_rate < -self.funding_threshold and combined_score < 0):
                funding_penalty = 0.25

            final_score = combined_score * funding_penalty
            final_scores[symbol] = final_score

            score_components[symbol] = {
                'price_roc': price_roc, 'oi_roc': oi_roc, 'trend_score': trend_score,
                'basis_momentum': basis_momentum, 'avg_volume_ratio': avg_volume_ratio,
                'sentiment_score': sentiment_score, 'funding_penalty': funding_penalty,
                'final_score': final_score
            }

        if not final_scores:
            return {}, {}

        ranked = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        signals = {}
        num_assets = len(ranked)
        long_cutoff_idx = int(num_assets * self.quantile)
        short_cutoff_idx = int(num_assets * (1 - self.quantile))
        longs = ranked[:long_cutoff_idx]
        shorts = ranked[short_cutoff_idx:]

        if not longs or not shorts:
            return {}, score_components

        long_scores = [len(longs) - i for i in range(len(longs))]
        total_long_score = sum(long_scores)
        for i, (symbol, score) in enumerate(longs):
            signals[symbol] = SignalEvent(
                symbol=symbol, weight=(long_scores[i] / total_long_score))

        short_scores = [len(shorts) - i for i in range(len(shorts))]
        total_short_score = sum(short_scores)
        for i, (symbol, score) in enumerate(shorts):
            signals[symbol] = SignalEvent(
                symbol=symbol, weight=-1 * (short_scores[i] / total_short_score))

        long_symbols = {s[0] for s in longs}
        short_symbols = {s[0] for s in shorts}
        for symbol in data.keys():
            if symbol not in long_symbols and symbol not in short_symbols:
                signals[symbol] = SignalEvent(symbol=symbol, weight=0.0)

        return signals, score_components
