from typing import Dict, Tuple
import pandas as pd
import numpy as np
from ..core.event import SignalEvent
from ..alpha.pipeline import AlphaPipeline
from ..portfolio.optimizer import PortfolioOptimizer


class DistributedStrategy:
    def __init__(self, lookback=90, correlation_lookback=30, use_optimization=False):
        # Engines
        self.pipeline = AlphaPipeline(lookback=lookback)
        self.optimizer = PortfolioOptimizer(
            max_leverage=1.0,
            max_position=0.10,  # Max 10% per coin
            lambda_risk=2.0     # Moderate risk aversion
        )
        self.correlation_lookback = correlation_lookback
        self.use_optimization = use_optimization

    def _estimate_covariance(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Calculates the Sample Covariance Matrix from historical returns.
        This tells the optimizer how assets move together.
        """
        # 1. Extract recent Close prices
        # We need a matrix of Time x Assets
        closes = {}
        required_len = self.correlation_lookback + 5

        for sym, df in data.items():
            if len(df) >= required_len:
                # Get recent history
                closes[sym] = df['futures_close'].iloc[-required_len:]

        if not closes:
            return pd.DataFrame()

        # 2. Calculate Returns
        price_df = pd.DataFrame(closes).ffill()
        returns_df = price_df.pct_change().dropna()

        if returns_df.empty:
            return pd.DataFrame()

        # 3. Calculate Covariance
        # This is an N x N matrix
        cov_matrix = returns_df.cov()

        # Simple Shrinkage (Optional but recommended for stability)
        # Pulls off-diagonal correlations slightly towards zero
        cov_matrix = cov_matrix * 0.9 + \
            np.eye(len(cov_matrix)) * \
            (cov_matrix.values.diagonal().mean() * 0.1)

        return cov_matrix

    def on_rebalance(self, data: Dict[str, pd.DataFrame]) -> Tuple[Dict, Dict]:
        # --- 1. Strict History Filtering (CRITICAL FIX) ---
        # Match FinalStrategy's required_history logic
        # lookback(90) + vol_lookback(30) + smooth(10) + buffer(10) ~ 140
        required_len = 90

        valid_data = {}
        for sym, df in data.items():
            if len(df) >= required_len:
                valid_data[sym] = df

        if not valid_data:
            return {}, {}

        # --- 2. Run Pipeline on VALID assets only ---
        alpha_df = self.pipeline.run(valid_data)
        if alpha_df.empty:
            return {}, {}

        # --- 3. Sort & Select (Split Bucket Logic) ---
        # Sort Descending (High Score = Best)
        current_scores = alpha_df['score'].sort_values(ascending=False)
        symbols = current_scores.index.tolist()

        # Exact quantile logic from FinalStrategy
        long_cutoff = int(len(symbols) * 0.4)
        short_cutoff = int(len(symbols) * (1 - 0.4))

        raw_longs = symbols[:long_cutoff]
        raw_shorts = symbols[short_cutoff:]

        # --- 4. Weight Calculation (Manual Ranking) ---
        signals = {}
        MAX_WEIGHT = 0.1  # 10% Cap

        # Helper to calculate rank weights exactly like FinalStrategy
        def apply_rank_weights(assets, is_long):
            if not assets:
                return
            n = len(assets)
            # Sum of 1..N
            total_rank_sum = n * (n + 1) / 2
            target_exposure = 0.5  # 50% per side

            for i, sym in enumerate(assets):
                # Longs: Top (i=0) gets rank N. Shorts: Top (i=0) gets rank 1
                rank = (n - i) if is_long else (i + 1)

                # Raw weight
                w = (rank / total_rank_sum) * target_exposure
                # Cap
                w = min(w, MAX_WEIGHT)

                signals[sym] = SignalEvent(
                    symbol=sym,
                    weight=w if is_long else -w
                )

        apply_rank_weights(raw_longs, is_long=True)
        apply_rank_weights(raw_shorts, is_long=False)

        # --- 5. Zero-Fill for Reporting ---
        # This ensures logs show 0.0 for assets we ignored
        active_syms = set(signals.keys())
        for sym in data.keys():
            if sym not in active_syms:
                signals[sym] = SignalEvent(symbol=sym, weight=0.0)

        debug_scores = {sym: {'final_score': val}
                        for sym, val in current_scores.items()}

        return signals, debug_scores
