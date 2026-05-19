"""
Pure helper functions for the EBM walk-forward trainer, factored out of
src/scripts/train_ebm_signal.py during the phase-4a refactor.

Everything in this module is at module scope, has no closure over the
training script's state, and is byte-equivalent to the inlined originals.
The training script keeps backward-compatible re-export shims so any
external code that imported these symbols from
`src.scripts.train_ebm_signal` continues to work.

Contents
--------
  _fold_portfolio_perf   in-sample Sharpe / total return for one fold
  _embargo_gap            days to skip between train end and prediction date
  _block_bootstrap_counts block-bootstrap-with-replacement per-date counts
  _ensemble_importances   mean of term_importances across a bag list
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _fold_portfolio_perf(
    train_data: pd.DataFrame,
    y_pred: np.ndarray,
    quantile: float,
    beta_col: "str | None" = None,
) -> dict:
    """
    In-sample portfolio Sharpe and total return for one fold.
    Uses rank-proportional top/bottom-quantile weights on y_raw (plain return),
    matching the OOS weight construction logic.

    If beta_col is provided, predictions are cross-sectionally beta-neutralized
    per date (OLS residualization) before ranking — matching OOS neutralize_scores.
    """
    df = train_data.copy()
    df["_pred"] = y_pred

    daily_rets = {}
    for ts, grp in df.groupby("ts"):
        grp = grp.dropna(subset=["y_raw"])
        n_assets = len(grp)
        if n_assets < 4:
            continue

        # Beta-neutralize predictions to match OOS neutralize_scores
        if beta_col and beta_col in grp.columns:
            pred = grp["_pred"].copy()
            beta = grp[beta_col]
            valid = pred.notna() & beta.notna() & ~np.isinf(pred) & ~np.isinf(beta)
            if valid.sum() >= 3 and np.var(beta[valid].values) > 1e-8:
                slope, intercept = np.polyfit(
                    beta[valid].values, pred[valid].values, 1)
                pred[valid] = pred[valid] - (slope * beta[valid] + intercept)
            grp = grp.copy()
            grp["_pred"] = pred

        int_ranks = grp["_pred"].rank(method="first")   # 1 = lowest score
        long_m = int_ranks > (n_assets * (1 - quantile))
        short_m = int_ranks <= (n_assets * quantile)
        if long_m.sum() == 0 or short_m.sum() == 0:
            continue

        long_rank_scores = int_ranks[long_m]
        long_w = (long_rank_scores / long_rank_scores.sum()) * 0.5

        short_rank_scores = (n_assets + 1 - int_ranks[short_m])
        short_w = (short_rank_scores / short_rank_scores.sum()) * 0.5

        w = pd.Series(0.0, index=grp.index)
        w[long_m] = long_w.values
        w[short_m] = -short_w.values
        daily_rets[ts] = float((w * grp["y_raw"]).sum())

    if len(daily_rets) < 5:
        return {"sharpe": np.nan, "total_return": np.nan, "n_days": len(daily_rets),
                "daily_rets": daily_rets}

    rets = pd.Series(daily_rets)
    sharpe = float(rets.mean() / (rets.std() + 1e-12) * np.sqrt(252))
    total_ret = float((1 + rets).prod() - 1)
    return {"sharpe": sharpe, "total_return": total_ret, "n_days": len(rets),
            "daily_rets": daily_rets}


def _embargo_gap(n_train_dates: int, target_horizon: int, embargo_pct: float) -> int:
    """
    Periods to skip between train end and prediction date.
    At minimum target_horizon; embargo_pct adds a fractional buffer to guard
    against leakage from overlapping multi-day labels.
    """
    return max(target_horizon, int(n_train_dates * embargo_pct))


def _block_bootstrap_counts(
    n_dates: int, block_size: int, n_blocks: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Block bootstrap with REPLACEMENT — returns per-date counts.

    Samples `n_blocks` consecutive blocks of length `block_size` from
    [0, n_dates), with replacement. A given date can appear in multiple
    blocks; this function tallies how many blocks cover each date and
    returns the count vector (length n_dates, dtype int16).

    This is the canonical block bootstrap: duplicates ARE preserved, so
    each bag's effective sample size equals `n_dates` (with repeats),
    and the out-of-bag fraction follows the standard 1 − 1/e ≈ 37% law,
    giving real variance reduction on averaging. Returning unique indices
    (the previous behaviour) collapsed each bag back toward the full
    training set and killed the variance-reduction benefit.

    All sampling is strictly within [0, n_dates) — no peeking forward.
    """
    if block_size <= 0 or n_dates <= block_size:
        return np.ones(n_dates, dtype=np.int16)
    counts = np.zeros(n_dates, dtype=np.int16)
    max_start = n_dates - block_size
    starts = rng.integers(0, max_start + 1, size=n_blocks)
    for s in starts:
        end = min(s + block_size, n_dates)
        counts[s:end] += 1
    return counts


def _ensemble_importances(model_list: list) -> pd.Series:
    """Average term importances across a list of EBM models.

    Used to aggregate per-bag importances into a single per-fold series.
    Two functionally-identical copies were previously nested inside
    `walk_forward` and `walk_forward_ho_moe`; both now delegate here.
    """
    imp = [
        pd.Series(m.term_importances(), index=list(m.term_names_))
        for m in model_list
    ]
    return pd.concat(imp, axis=1).mean(axis=1)
