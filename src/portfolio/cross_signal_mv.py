"""
Cross-signal mean-variance optimisation, factored out of
src/scripts/backtest_combo.py during the phase-5 refactor.

The combiner blends per-strategy weights through a daily quadratic
program that maximises `μᵀλ − ρ·λᵀΣλ` subject to `λ ≥ 0, Σλ = 1`. With
only a handful of strategies and a short rolling lookback the sample
covariance is noisy enough that the optimiser routinely flips between
corner solutions; this module also provides the Ledoit-Wolf / diagonal
shrinkage estimator used to stabilise Σ.

Contents
--------
  compute_strategy_returns  per-date per-strategy return series from
                            lagged weights × asset returns
  _stable_cov               covariance estimator with shrinkage targets
                            (sample / diagonal / ledoit_wolf)
  optimize_signal_weights   quadratic-program solver for λ

All three are byte-equivalent to the previously-inlined originals.
"""
from __future__ import annotations

import cvxpy as cp
import numpy as np
import pandas as pd


def compute_strategy_returns(
    aligned_strategies: dict,
    returns_df: pd.DataFrame,
    all_ts: pd.Index,
) -> pd.DataFrame:
    """
    Computes the daily portfolio return series for each strategy using
    lagged weights — return on day t = weights posted at end-of-day t-1
    dotted with asset returns on day t.

    Returns a DataFrame of shape (n_dates, n_strategies).
    """
    common_cols = returns_df.columns
    strat_rets = {}
    for name, w_df in aligned_strategies.items():
        w_aligned = w_df.reindex(index=all_ts, columns=common_cols).fillna(0.0)
        lagged_w = w_aligned.shift(1).fillna(0.0)
        r_aligned = returns_df.reindex(index=all_ts, columns=common_cols).fillna(0.0)
        strat_rets[name] = (lagged_w * r_aligned).sum(axis=1)
    return pd.DataFrame(strat_rets, index=all_ts)


def _stable_cov(
    rets: pd.DataFrame,
    method: str = "ledoit_wolf",
    shrinkage: "float | None" = None,
) -> np.ndarray:
    """
    Covariance estimator with shrinkage toward a structured target.

    With only 3 strategies and a ~30-day lookback the sample covariance
    has ≈ 33 degrees of freedom for 6 off-diagonal entries — very noisy.
    Shrinkage toward `var_mean · I` (Ledoit-Wolf target) or toward the
    diagonal stabilises Σ and prevents the corner-solution flips we saw
    on the unsmoothed λ.

    method
        "sample"      — no shrinkage (legacy behaviour)
        "diagonal"    — zero off-diagonals (assumes strategies uncorrelated)
        "ledoit_wolf" — shrink toward var_mean · I with optimal intensity

    shrinkage : optional float ∈ [0, 1]
        If provided, overrides the data-driven intensity. Only meaningful
        for "ledoit_wolf" / "diagonal".
    """
    X = rets.values
    n_samples, n_assets = X.shape
    sample = np.cov(X, rowvar=False)
    if n_assets <= 1:
        return np.atleast_2d(sample)

    if method == "sample":
        return sample + np.eye(n_assets) * 1e-8

    if method == "diagonal":
        target = np.diag(np.diag(sample))
        alpha = 0.5 if shrinkage is None else float(shrinkage)
        return (1 - alpha) * sample + alpha * target + np.eye(n_assets) * 1e-8

    # ledoit_wolf
    if shrinkage is None:
        try:
            from sklearn.covariance import LedoitWolf
            lw = LedoitWolf().fit(X)
            return lw.covariance_ + np.eye(n_assets) * 1e-8
        except ImportError:
            # Manual fallback: scaled-identity shrinkage with a fixed alpha.
            shrinkage = 0.3
    # Manual LW target = var_mean · I, intensity = shrinkage
    var_mean = float(np.trace(sample) / n_assets)
    target = var_mean * np.eye(n_assets)
    alpha = float(shrinkage)
    return (1 - alpha) * sample + alpha * target + np.eye(n_assets) * 1e-8


def optimize_signal_weights(
    strat_ret_window: pd.DataFrame,
    lambda_risk: float,
    cov_method: str = "ledoit_wolf",
    cov_shrinkage: "float | None" = None,
) -> pd.Series:
    """
    Mean-variance optimization in strategy space.

    Maximises: μᵀλ − lambda_risk · λᵀΣλ
    Subject to: λ ≥ 0,  sum(λ) = 1

    μ = mean daily return per strategy over the lookback window.
    Σ = covariance of daily strategy returns (estimator controlled by
        cov_method — defaults to Ledoit-Wolf shrinkage for stability
        with the small strategy count).

    Falls back to equal weights on solver failure or insufficient data.
    """
    names = strat_ret_window.columns.tolist()
    n = len(names)

    if n == 1:
        return pd.Series(1.0, index=names)

    mu = strat_ret_window.mean().values
    Sigma = _stable_cov(strat_ret_window, method=cov_method,
                        shrinkage=cov_shrinkage)

    lam = cp.Variable(n)
    objective = cp.Maximize(
        mu @ lam - lambda_risk * cp.quad_form(lam, cp.psd_wrap(Sigma))
    )
    constraints = [lam >= 0, cp.sum(lam) == 1]

    try:
        cp.Problem(objective, constraints).solve()
        if lam.value is None:
            return pd.Series(1.0 / n, index=names)
        result = pd.Series(lam.value, index=names).clip(lower=0)
        total = result.sum()
        return result / total if total > 1e-8 else pd.Series(1.0 / n, index=names)
    except Exception:
        return pd.Series(1.0 / n, index=names)
