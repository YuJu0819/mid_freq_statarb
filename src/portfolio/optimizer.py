import cvxpy as cp
import numpy as np
import pandas as pd


class PortfolioOptimizer:
    def __init__(self, max_leverage=1.0, max_position=0.10, dollar_neutral=True):
        self.max_leverage = max_leverage
        self.max_position = max_position
        self.dollar_neutral = dollar_neutral

    def optimize(self, alpha_scores: pd.Series, current_weights: pd.Series = None) -> pd.Series:
        """
        Solves for optimal weights given an Alpha Score vector.
        """
        # 1. Clean Data (Remove NaNs)
        alpha_scores = alpha_scores.dropna()
        assets = alpha_scores.index.tolist()
        n = len(assets)

        if n == 0:
            return pd.Series()

        # 2. Define Variables
        w = cp.Variable(n)
        alpha_vec = alpha_scores.values

        # 3. Define Objective
        # Maximize: w * alpha (Exposure to high scores)
        # Note: In a real risk model, you would subtract lambda * (w.T @ Cov @ w) here.
        objective = cp.Maximize(w @ alpha_vec)

        # 4. Define Constraints
        constraints = [
            cp.sum(cp.abs(w)) <= self.max_leverage,  # Gross Exposure <= 100%
            cp.abs(w) <= self.max_position           # No single bet > 10%
        ]

        if self.dollar_neutral:
            constraints.append(cp.sum(w) == 0)       # Longs = Shorts

        # 5. Solve
        try:
            prob = cp.Problem(objective, constraints)
            # ECOS is standard, usually pre-installed
            prob.solve(solver=cp.ECOS)

            if w.value is None:
                print("[Optimizer] Solution failed (None). Returning zeros.")
                return pd.Series(0, index=assets)

            # 6. Format Output
            # Tiny weights (< 0.01%) are noise; clean them to 0
            weights = pd.Series(w.value, index=assets)
            weights[weights.abs() < 0.0001] = 0.0

            return weights

        except Exception as e:
            print(f"[Optimizer] Solver Error: {e}")
            return pd.Series(0, index=assets)
