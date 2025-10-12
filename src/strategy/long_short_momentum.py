import pandas as pd
from typing import Dict
from ..core.event import SignalEvent


class LongShortMomentumStrategy:
    """
    A dollar-neutral, cross-sectional momentum strategy.
    Goes long the top quantile and short the bottom quantile of assets based on momentum.
    """

    def __init__(self, lookback: int = 90, quantile: float = 0.2):
        self.lookback = lookback
        self.quantile = quantile
        self.min_data_points = lookback + 5

    def on_rebalance(self, data: Dict[str, pd.DataFrame]) -> Dict[str, SignalEvent]:
        momentum = {}
        for symbol, df in data.items():
            if len(df) < self.min_data_points:
                continue
            roc = (df["close"].pct_change(self.lookback).iloc[-1])
            momentum[symbol] = roc

        if not momentum:
            return {}

        # --- 1. Rank assets and determine quantiles ---
        ranked = sorted(momentum.items(), key=lambda x: x[1], reverse=True)
        num_assets = len(ranked)
        long_cutoff_idx = int(num_assets * self.quantile)
        short_cutoff_idx = int(num_assets * (1 - self.quantile))

        longs = ranked[:long_cutoff_idx]
        shorts = ranked[short_cutoff_idx:]

        if not longs or not shorts:
            # Not enough assets to form a long-short portfolio
            return {sym: SignalEvent(symbol=sym, weight=0.0) for sym in data.keys()}

        # V-- NEW RANK-BASED WEIGHTING LOGIC --V
        # --- 2. Calculate rank-based weights (dollar-neutral) ---
        signals = {}

        # --- Calculate for Longs ---
        # Assign scores based on rank (e.g., top asset gets N, second gets N-1, ...)
        long_scores = [len(longs) - i for i in range(len(longs))]
        total_long_score = sum(long_scores)

        for i, (symbol, score) in enumerate(longs):
            weight = long_scores[i] / total_long_score
            signals[symbol] = SignalEvent(symbol=symbol, weight=weight)

        # --- Calculate for Shorts ---
        # Same logic, but weights are negative
        short_scores = [len(shorts) - i for i in range(len(shorts))]
        total_short_score = sum(short_scores)

        for i, (symbol, score) in enumerate(shorts):
            weight = -1 * (short_scores[i] / total_short_score)
            signals[symbol] = SignalEvent(symbol=symbol, weight=weight)
        # ^-- END OF NEW LOGIC --^
        # --- 3. Signal to exit all other positions ---
        long_symbols = {s[0] for s in longs}
        short_symbols = {s[0] for s in shorts}
        for symbol in data.keys():
            if symbol not in long_symbols and symbol not in short_symbols:
                signals[symbol] = SignalEvent(symbol=symbol, weight=0.0)

        return signals
