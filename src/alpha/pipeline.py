import pandas as pd
import numpy as np
from .. import factors


class AlphaPipeline:
    def __init__(self, lookback=30, smooth=10, vol_lookback=30,
                 funding_lookback=180, funding_threshold=0.002, funding_z_threshold=1.5,
                 vol_adj_factor=0.5):
        self.lookback = lookback
        self.smooth = smooth
        self.vol_lookback = vol_lookback
        self.funding_lookback = funding_lookback
        self.funding_threshold = funding_threshold
        self.funding_z_threshold = funding_z_threshold
        self.vol_adj_factor = vol_adj_factor

    def run(self, data: dict, market_regime_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Returns a DataFrame of FINAL Alpha Scores (Trend + Sentiment - Penalties).
        Calculates on FULL history first to ensure Z-Scores are correct.
        """
        # 1. Prepare Wide Dataframes (Full History)
        closes = pd.DataFrame({s: df['futures_close']
                              for s, df in data.items()}).ffill()
        opens = pd.DataFrame({s: df['open_interest']
                             for s, df in data.items()}).ffill()
        basis = pd.DataFrame({s: df['basis']
                             for s, df in data.items()}).ffill()
        vol_ratio = pd.DataFrame({s: df['volume_ratio']
                                 for s, df in data.items()}).fillna(0.0)
        funding = pd.DataFrame({s: df['funding_rate']
                               for s, df in data.items()}).fillna(0.0)

        # 2. Vectorized Factor Calculations (Full History)
        # We do NOT use .iloc[-1] here. We keep the time index.
        price_roc = factors.calc_price_mom(
            closes, self.lookback, self.smooth).fillna(0.0)
        oi_roc = factors.calc_oi_mom(
            opens, self.lookback, self.smooth).fillna(0.0)
        basis_mom = factors.calc_basis_mom(
            basis, closes, self.lookback, self.smooth).fillna(0.0)
        vol_ratio_sig = factors.calc_vol_ratio_signal(
            vol_ratio, self.lookback, self.lookback).fillna(1.0)

        volatility = factors.calc_volatility(
            closes, self.vol_lookback).fillna(0.0)

        # Funding Z-Score (Rolling)
        funding_z = factors.calc_funding_zscore(
            funding, self.funding_lookback).fillna(0.0)

        # 3. Score Composition
        trend_score = price_roc * (1 + 2 * oi_roc)

        # Sentiment Score
        valid_sentiment = ~(np.isinf(basis_mom) | np.isinf(vol_ratio_sig))
        sentiment_score = (basis_mom * vol_ratio_sig *
                           5).where(valid_sentiment, 0.0)

        # --- V-- CRITICAL FIX: SENTIMENT Z-SCORE --V ---
        # Now we can calculate the Z-Score because we still have the full history.
        # We use the same 'calc_funding_zscore' logic (rolling mean/std) but applied to sentiment.
        # This tells us: "Is sentiment for this coin usually high or low?"
        sent_z = factors.calc_funding_zscore(
            sentiment_score, self.lookback).fillna(0.0)
        # -----------------------------------------------

        combined_score = trend_score + sentiment_score

        # 4. Adjustments (Vectorized over full history)

        # A. Funding Penalty
        funding_penalty = pd.DataFrame(
            1.0, index=combined_score.index, columns=combined_score.columns)

        boost_mask = (
            ((funding_z > self.funding_z_threshold) & (combined_score > 0)) |
            ((funding_z < -self.funding_z_threshold) & (combined_score < 0))
        )

        kill_mask = (
            ((funding_z < -self.funding_z_threshold * 2) & (combined_score > 0)) |
            ((funding_z > self.funding_z_threshold * 2) & (combined_score < 0)) |
            (abs(funding_z) < self.funding_z_threshold * 0.1)
        )

        funding_penalty[boost_mask] = 1.5
        funding_penalty[kill_mask] = 0.5

        # B. Sentiment Penalty (Using the correct Time-Series Z-Score)
        # "Boost if sentiment is statistically normal (not extreme)"
        sent_penalty = pd.DataFrame(
            1.0, index=combined_score.index, columns=combined_score.columns)
        sent_penalty[abs(sent_z) < 1] = 1.5

        # C. Volatility Adjustment
        active_vol = volatility.replace(0.0, np.nan)
        # Rank across columns (axis=1) for each timestamp
        vol_rank = active_vol.rank(axis=1, pct=True).fillna(0.5)
        vol_adj = (1 - vol_rank * self.vol_adj_factor).clip(lower=0.0)

        # 5. Final Score Calculation (Full History)
        final_scores_series = combined_score * vol_adj * funding_penalty * sent_penalty

        # --- 6. Final Slice (The "Live" Values) ---
        # NOW we slice to get the most recent values for the rebalance.

        # Extract the last row for everything
        current_scores = final_scores_series.iloc[-1]

        # Pack into result DataFrame
        # Note: We slice the components too so the Strategy gets the latest values for masking
        result_df = pd.DataFrame({
            'score': current_scores,
            'trend_score': trend_score.iloc[-1],
            'volatility': volatility.iloc[-1],
            'funding_z': funding_z.iloc[-1],
            'sentiment_score': sentiment_score.iloc[-1],
            'sent_z': sent_z.iloc[-1]  # Available for inspection/masking
        })

        return result_df
