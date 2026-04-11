import cvxpy as cp
import pandas as pd
import numpy as np


class PortfolioOptimizer:
    def __init__(self, max_leverage=1.0, max_position=0.10, lambda_risk=0.1):
        """
        :param lambda_risk: Risk aversion parameter. 
                            Higher = Safer portfolio (weights shrink towards 0 or lower correlation).
                            Lower = Aggressive (chases Alpha regardless of volatility).
        """
        self.max_leverage = max_leverage
        self.max_position = max_position
        self.lambda_risk = lambda_risk

    def optimize_mean_variance(self, alpha_scores: pd.Series, cov_matrix: pd.DataFrame) -> pd.Series:
        """
        Solves for weights that maximize Alpha while minimizing Portfolio Variance.
        """
        # 1. Align Data
        # Ensure Alpha and Covariance have the exact same assets in the same order
        common_assets = alpha_scores.index.intersection(cov_matrix.index)
        if len(common_assets) == 0:
            return pd.Series()

        alpha_vec = alpha_scores.loc[common_assets].values
        Sigma = cov_matrix.loc[common_assets, common_assets].values
        n_assets = len(common_assets)

        # 2. Define Variables
        w = cp.Variable(n_assets)

        # 3. Define Terms
        # Return Term: w * Alpha
        ret = alpha_vec @ w

        # Risk Term: w.T * Cov * w (Quadratic Form)
        # We perform Cholesky decomposition protection or use psd_wrap if matrix is noisy
        risk = cp.quad_form(w, cp.psd_wrap(Sigma))

        # Objective: Maximize Risk-Adjusted Return
        objective = cp.Maximize(ret - self.lambda_risk * risk)

        # 4. Constraints
        constraints = [
            # Dollar Neutral (Long = Short)
            cp.sum(w) == 0,
            cp.sum(cp.abs(w)) <= self.max_leverage,  # Gross Exposure <= 1.0
            # cp.sum(cp.abs(w)) >= self.max_leverage * 0.8,
            cp.abs(w) <= self.max_position,         # Position Limit
        ]

        # 5. Solve
        try:
            # Let CVXPY choose the best installed solver (fixes ECOS error)
            prob = cp.Problem(objective, constraints)
            prob.solve()

            if w.value is None:
                print(f"[Optimizer] Failed. Status: {prob.status}")
                return pd.Series(0.0, index=common_assets)

            # 6. Cleanup Weights
            result = pd.Series(w.value, index=common_assets)

            # Zero out "dust" (tiny floating point noise)
            result[result.abs() < 0.0001] = 0.0

            return result

        except Exception as e:
            print(f"[Optimizer] Solver Crash: {e}")
            return pd.Series(0.0, index=common_assets)

    # ... inside PortfolioOptimizer class ...

    def optimize_linear(self, alpha_scores: pd.Series) -> pd.Series:
        """
        Heuristic Method: Assigns weights directly proportional to Alpha Score.
        Formula: weight = (score / sum_of_abs_scores) * max_leverage
        """
        # 1. Clean Data
        scores = alpha_scores.copy().dropna()
        assets = scores.index

        if scores.empty:
            return pd.Series()

        # 2. Enforce Dollar Neutrality (Optional but recommended)
        # This ensures Longs match Shorts even if your raw scores are skewed.
        # Logic: Shift scores so their mean is 0.
        scores = scores - scores.mean()

        # 3. Normalize to Target Leverage
        # "How much conviction is this asset relative to the whole pot?"
        total_abs_score = scores.abs().sum()

        if total_abs_score == 0:
            return pd.Series(0.0, index=assets)

        # Scale: A score of 2.0 gets 2x the weight of a score of 1.0
        raw_weights = (scores / total_abs_score) * self.max_leverage

        # 4. Cap at Max Position
        # Clip weights that exceed your 10% limit
        final_weights = raw_weights.clip(-self.max_position, self.max_position)

        # 5. Clean Dust
        final_weights[final_weights.abs() < 0.0001] = 0.0

        return final_weights
