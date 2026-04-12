"""
EBM (Explainable Boosting Machine) signal generation.

Walk-forward: model retrained every `--retrain_freq` periods, predicts the
next period only, so there is zero look-ahead by construction.

Residual-Based Mixture of Experts (MoE) Mode (--use_moe)
---------------------------------------------------------
A two-stage ensemble that captures regime-specific alpha on top of a
global EBM baseline:
  1. Global Phase : A single EBM is trained on the full training window.
  2. Residual Calc: y_residual = y_true − y_global_pred (in-sample).
  3. Expert Phase : Per-regime "Expert" EBMs are trained on the residuals
                   so each expert learns the systematic error the global
                   model makes in that regime.
  Inference       : Score_total = Global_Score + (λ × Expert_Score)
  Regime Gating   : Expert selection uses a 1-period lagged regime label
                   plus optional hysteresis to avoid rapid expert switching.

Target options
--------------
  cs_rank   Cross-sectional percentile rank of forward return (default).
  raw       Raw forward return.

Feature normalization options
-----------------------------
  cs   cs-z-score  ts   ts-z-score  rank   rank  none   raw

Signal construction
-------------------
  After walk-forward prediction the raw scores are:
    1. Beta-neutralized via OLS residualization (--beta_neutral, default on).
    2. Converted to weights via quantile selection + chosen weight_mode.

General utilities live in:  src/alpha/ml_utils.py

Outputs (all in ./reports/strategies/{run_id}/)
------------------------------------------------
  ebm.parquet                weight matrix (ts × symbol) — pipeline-compatible
  ebm_predictions.parquet    raw OOS prediction scores (ts × symbol)
  ebm_feature_importance.csv per-fold feature importances (global model)
  ebm_expert_importance_regime_*.csv  per-fold expert importances (MoE mode)
  ebm_report.png             importance bar + OOS IC + cumulative PnL preview

Usage
-----
  python -m src.scripts.train_ebm_signal \\
      --run_id batch_v1 \\
      --panel_path ./data/ml/factor_panel_2024-01-01_2025-01-01.parquet \\
      --target_col ret_1d  --target_horizon 1 \\
      --train_window 252   --retrain_freq 21  --min_train_periods 90 \\
      --feature_norm cs    --target_type cs_rank \\
      --quantile 0.3       --max_weight 0.10    --weight_mode rank

  # MoE mode
  python -m src.scripts.train_ebm_signal \\
      --run_id moe_v1 \\
      --panel_path ./data/ml/factor_panel_2024-01-01_2025-01-01.parquet \\
      --use_moe --regime_col volatility_regime_enc \\
      --moe_boost_lambda 0.5 --expert_interactions 5 --moe_hysteresis 3
"""
from ..core.utils import ensure_dir
from ..alpha.ml_utils import (
    normalize_features,
    build_target,
    predictions_to_weights,
    neutralize_scores,
    compute_portfolio_performance,
    compute_ic,
)

import argparse
import os
import pickle
import warnings

from joblib import parallel_backend

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    from interpret.glassbox import ExplainableBoostingRegressor
except ImportError:
    raise ImportError(
        "The 'interpret' package is required.\n"
        "Install it with:  pip install interpret"
    )


# ---------------------------------------------------------------------------
# Default feature list
# ---------------------------------------------------------------------------

_SIGNAL_COLS = {"mom_signal", "rev_signal"}
# Meta / identifier columns + market-level (cross-sectionally constant) columns.
# The *_regime strings, adx, and the *_regime_enc numerics are broadcast from
# market-wide calcs in build_factor_panel.py, so every symbol shares the same
# value on a given date. As EBM main effects these add a per-date constant to
# all symbols' scores — cancelled out by long/short ranking — and they only
# corrupt FAST's pair search. Excluded from `--features all`; re-add explicitly
# via --features if you want to experiment with them as interaction partners.
_META_COLS = {
    "ts", "symbol",
    "volatility_regime", "trend_regime", "skew_regime",
    "adx", "market_adx",
    "volatility_regime_enc", "trend_regime_enc", "skew_regime_enc",
}
_TARGET_COLS = {"ret_1d", "ret_5d", "ret_20d"}

DEFAULT_FEATURES = [
    # price
    "ret_1d", "ret_5d", "ret_20d",
    "volatility_30", "vol_rank_cs", "beta_60", "skewness_90",
    # momentum
    "price_roc", "oi_roc", "basis_norm", "basis_mom",
    "vol_ratio_sig", "trend_score", "sentiment_score", "combined_score",
    "funding_z", "funding_penalty", "mom_final_score",
    # reversal
    "ls_ratio", "ls_chg_1d", "oi_pct_chg_1d",
    "cs_z_oi_chg", "ts_z_oi_chg", "liquidation_shock",
    "regime_score", "interaction_alpha", "reversal_hawkes", "rev_final_score",
    # market regime — numeric-encoded
    "market_adx",
    "volatility_regime_enc",
    "trend_regime_enc",
    "skew_regime_enc",
    # delta family (rolling-5 mean minus lag-10 of itself)
    # Only factors with lookback <= 30d; long-window factors (beta_60, skewness_90,
    # funding_z, liquidation_shock, regime_score, interaction_alpha) are excluded
    # because a 10-day delta on a 40-180d stat is near-constant and uninformative.
    "volatility_30_delta", "vol_rank_cs_delta",
    "price_roc_delta", "oi_roc_delta", "basis_norm_delta", "basis_mom_delta",
    "vol_ratio_sig_delta", "trend_score_delta", "sentiment_score_delta",
    "combined_score_delta", "mom_final_score_delta",
    "ls_ratio_delta", "rev_final_score_delta",
]
_FILTERED_COLS: set = set()   # populate to enable --features filtered


# ---------------------------------------------------------------------------
# EBM walk-forward training
# ---------------------------------------------------------------------------


def _fold_portfolio_perf(train_data: pd.DataFrame,
                         y_pred: np.ndarray,
                         quantile: float,
                         beta_col: str = None) -> dict:
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


def _block_bootstrap_indices(
    n: int, block_size: int, rng: np.random.Generator
) -> np.ndarray:
    """
    Block bootstrap: sample consecutive blocks of row indices with replacement.
    Preserves local temporal autocorrelation so each bag reflects realistic
    time-series behaviour. Returns sorted unique indices (size ≈ n).
    """
    if block_size <= 0 or n <= block_size:
        return np.arange(n)
    n_blocks = max(1, n // block_size)
    max_start = n - block_size
    starts = rng.integers(0, max_start + 1, size=n_blocks)
    idx = np.concatenate([np.arange(s, min(s + block_size, n))
                         for s in starts])
    return np.unique(idx)


# ---------------------------------------------------------------------------
# Regime-gating helpers (MoE)
# ---------------------------------------------------------------------------


class RegimeSelector:
    """
    Maps each prediction date to a discrete regime label using:
      1. Ex-ante lag  : regime at time t is derived from the value at t-1,
                        eliminating any look-ahead bias at prediction time.
      2. Hysteresis   : the active label only switches after the new regime
                        has been observed for `hysteresis` consecutive periods,
                        reducing expert-switching turnover.

    Parameters
    ----------
    panel       : full factor panel (must contain `ts` and `regime_col`).
    regime_col  : name of the numeric regime column (e.g. volatility_regime_enc).
    hysteresis  : minimum consecutive days in the new regime before switching.
    """

    def __init__(
        self,
        panel: pd.DataFrame,
        regime_col: str,
        hysteresis: int = 3,
    ):
        self.regime_col = regime_col
        self.hysteresis = hysteresis

        # Market-wide — same value for all symbols on a date, so .first() is fine.
        dates = sorted(panel["ts"].unique())
        raw_regime = (
            panel.groupby("ts")[regime_col]
            .first()
            .reindex(dates)
        )

        # Store as string keys to avoid float-comparison issues (0.0 vs 0).
        raw_str = raw_regime.apply(
            lambda v: str(int(v)) if pd.notna(v) else "nan"
        )

        # Build the raw lookup (no lag) for use during IS expert training.
        self._raw_map: dict = dict(zip(dates, raw_str.values))

        # Build lagged + hysteresis-smoothed map for OOS expert selection.
        lagged = raw_str.shift(1)  # NaN for the very first date

        active: str | None = None
        pending: str | None = None
        pending_count: int = 0
        smoothed: dict = {}

        for date in dates:
            raw_val = lagged.get(date)
            is_nan = (raw_val is None) or (
                raw_val == "nan") or pd.isna(raw_val)

            if is_nan:
                smoothed[date] = active  # None until data arrives
                continue

            if active is None:
                active = raw_val
                pending = None
                pending_count = 0
            elif raw_val == active:
                pending = None
                pending_count = 0
            elif raw_val == pending:
                pending_count += 1
                if pending_count >= hysteresis:
                    active = pending
                    pending = None
                    pending_count = 0
            else:
                pending = raw_val
                pending_count = 1

            smoothed[date] = active

        self._smoothed_map: dict = smoothed

    def get_regime(self, ts) -> str | None:
        """
        Returns the active (lagged + hysteresis-smoothed) regime for
        OOS prediction at time `ts`.  None if not enough history yet.
        """
        return self._smoothed_map.get(ts)

    def get_raw_regime(self, ts) -> str | None:
        """
        Returns the actual (non-lagged) regime at time `ts`.
        Use this for in-sample expert training only.
        """
        return self._raw_map.get(ts)


class ResidualMoE:
    """
    Residual-Based Mixture of Experts ensemble.

    Holds one global EBM ensemble (trained on the full window target) and a
    per-regime expert EBM ensemble (trained on global residuals).

    Prediction
    ----------
    Score_total = Global_Score + (λ × Expert_Score)

    When the active regime has no trained expert the expert contribution
    falls back to zero, so the total score equals the global score.
    """

    def __init__(
        self,
        global_models: list,
        # str(regime) -> list[ExplainableBoostingRegressor]
        expert_dict: dict,
    ):
        self.global_models = global_models
        self.expert_dict = expert_dict

    # ------------------------------------------------------------------
    def predict_global(self, X: np.ndarray) -> np.ndarray:
        return np.mean([m.predict(X) for m in self.global_models], axis=0)

    def predict_expert(self, regime: str | None, X: np.ndarray) -> np.ndarray:
        if regime is None or regime not in self.expert_dict:
            return np.zeros(len(X))
        experts = self.expert_dict[regime]
        if not experts:
            return np.zeros(len(X))
        return np.mean([m.predict(X) for m in experts], axis=0)

    def predict_total(
        self,
        X: np.ndarray,
        regime: str | None,
        moe_boost_lambda: float,
    ) -> np.ndarray:
        g = self.predict_global(X)
        e = self.predict_expert(regime, X)
        return g + moe_boost_lambda * e

    # ------------------------------------------------------------------
    def global_importances(self) -> pd.Series:
        imp = [
            pd.Series(m.term_importances(), index=list(m.term_names_))
            for m in self.global_models
        ]
        return pd.concat(imp, axis=1).mean(axis=1)

    def expert_importances(self, regime: str) -> pd.Series | None:
        if regime not in self.expert_dict or not self.expert_dict[regime]:
            return None
        imp = [
            pd.Series(m.term_importances(), index=list(m.term_names_))
            for m in self.expert_dict[regime]
        ]
        return pd.concat(imp, axis=1).mean(axis=1)


# ---------------------------------------------------------------------------
# Walk-forward training
# ---------------------------------------------------------------------------


def walk_forward(
    panel: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    target_horizon: int,
    target_type: str,
    train_window: int,        # 0 = expanding
    retrain_freq: int,
    min_train_periods: int,
    quantile: float,
    ebm_kwargs: dict,
    save_models: bool,
    model_dir: str,
    beta_neutral: bool = False,
    beta_col: str = "beta_60",
    use_block_bagging: bool = False,
    n_outer_bags: int = 8,
    block_size: int = 21,
    embargo_pct: float = 0.01,
    # ── MoE ────────────────────────────────────────────────────────────
    use_moe: bool = False,
    regime_col: str = "volatility_regime_enc",
    moe_boost_lambda: float = 0.5,
    moe_hysteresis: int = 3,
    expert_ebm_kwargs: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, dict | None]:
    """
    Walk-forward EBM training.

    Non-MoE mode
    ------------
    When use_block_bagging=True, replaces EBM's internal outer_bags with a
    manual block-bootstrap ensemble.

    MoE mode (use_moe=True)
    -----------------------
    Each fold trains two stages:
      1. Global EBM  — full training window, same features/target as before.
      2. Expert EBMs — one per regime present in the training window,
                       each trained on the global model's in-sample residuals.
    OOS prediction at time t uses the regime label from t-1, smoothed by
    hysteresis, to select the active expert.

    Returns
    -------
    predictions_wide     : pd.DataFrame (ts × symbol) — raw OOS prediction scores
    importance_df        : pd.DataFrame — per-fold global model feature importances
    fold_stats_df        : pd.DataFrame — per-fold IS metrics
    is_rets              : pd.Series    — IS daily returns
    expert_importance_dfs: dict | None  — {regime_str: pd.DataFrame of fold importances}
                           Only populated when use_moe=True.
    """
    dates = sorted(panel["ts"].unique())
    n_dates = len(dates)
    label_cutoff_idx = n_dates - target_horizon

    all_preds: dict = {}
    all_importances: list = []
    all_fold_stats: list = []
    all_expert_importances: dict = {}  # regime_str -> list[pd.Series]
    is_daily_rets: dict = {}

    retrain_set = {
        i for i in range(min_train_periods, label_cutoff_idx)
        if (i - min_train_periods) % retrain_freq == 0
    }

    print(f"  Dates total: {n_dates}  |  "
          f"Prediction dates: {label_cutoff_idx - min_train_periods}  |  "
          f"Retraining {len(retrain_set)} times (every {retrain_freq} periods)")
    if use_block_bagging:
        print(f"  Block bagging ON: {n_outer_bags} bags  "
              f"block_size={block_size}  embargo_pct={embargo_pct}")

    # ── MoE initialisation ────────────────────────────────────────────────
    moe_active = False  # set True once regime_col is confirmed present
    regime_selector: RegimeSelector | None = None
    if use_moe:
        if regime_col not in panel.columns:
            print(f"  [warn] regime_col='{regime_col}' not found in panel; "
                  f"MoE disabled.")
        else:
            moe_active = True
            regime_selector = RegimeSelector(
                panel, regime_col, hysteresis=moe_hysteresis)
            eff_lambda = moe_boost_lambda
            eff_expert_kwargs = expert_ebm_kwargs or ebm_kwargs
            print(f"  MoE ON: regime_col={regime_col}  "
                  f"λ={eff_lambda}  hysteresis={moe_hysteresis}")

    # sentinel: list (standard) or ResidualMoE (moe); None = not yet trained
    models: list | ResidualMoE | None = None
    _first_train_logged = False
    rng = np.random.default_rng(ebm_kwargs.get("random_state", 42))

    def _fill_features(df: pd.DataFrame) -> pd.DataFrame:
        """Fill NaN features with 0 — neutral value for CS z-scored features."""
        df = df.copy()
        for col in feature_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0.0)
        return df

    def _predict_ensemble(X: np.ndarray) -> np.ndarray:
        """Average predictions across all models in the ensemble."""
        return np.mean([m.predict(X) for m in models], axis=0)

    def _train_ebm_ensemble(X: np.ndarray, y: np.ndarray, kwargs: dict) -> list:
        """Train a single or block-bagged EBM ensemble. Returns list of models."""
        if use_block_bagging:
            kwargs_bag = {**kwargs, "outer_bags": 1}
            bags = []
            for _ in range(n_outer_bags):
                idx = _block_bootstrap_indices(len(X), block_size, rng)
                m = ExplainableBoostingRegressor(**kwargs_bag)
                with parallel_backend("sequential"):
                    m.fit(X[idx], y[idx])
                bags.append(m)
            return bags
        else:
            m = ExplainableBoostingRegressor(**kwargs)
            with parallel_backend("sequential"):
                m.fit(X, y)
            return [m]

    def _ensemble_importances(model_list: list) -> pd.Series:
        """Average term importances across a list of EBM models."""
        imp = [
            pd.Series(m.term_importances(), index=list(m.term_names_))
            for m in model_list
        ]
        return pd.concat(imp, axis=1).mean(axis=1)

    for pred_idx in range(min_train_periods, label_cutoff_idx):
        pred_date = dates[pred_idx]

        # ── Retrain if needed ────────────────────────────────────────────────
        if pred_idx in retrain_set:
            gap = _embargo_gap(
                min(pred_idx, train_window if train_window > 0 else pred_idx),
                target_horizon, embargo_pct,
            )
            train_end_idx = max(0, pred_idx - gap)
            train_start_idx = max(
                0, train_end_idx - train_window) if train_window > 0 else 0

            train_dates = dates[train_start_idx:train_end_idx]
            train_data = panel[panel["ts"].isin(train_dates)].copy()
            train_data = train_data.dropna(subset=["y"])
            train_data = _fill_features(train_data)

            if len(train_data) < 50:
                if not _first_train_logged:
                    print(f"  [warn] pred_idx={pred_idx}: only {len(train_data)} "
                          f"valid training rows (need 50). Waiting for more data...")
                continue

            X_train = train_data[feature_cols].values
            y_train = train_data["y"].values

            if not _first_train_logged:
                print(f"  [info] First model fit at pred_idx={pred_idx} "
                      f"using {len(train_data)} rows ({len(train_dates)} dates × "
                      f"{len(train_data) // max(len(train_dates), 1)} avg symbols)")
                _first_train_logged = True

            # ── Two-stage MoE training ────────────────────────────────────────
            if moe_active:
                # Stage 1: Global EBM on full training window
                global_models = _train_ebm_ensemble(
                    X_train, y_train, ebm_kwargs)
                y_global_pred = np.mean(
                    [m.predict(X_train) for m in global_models], axis=0)

                # Stage 2: Expert EBMs per regime on global residuals
                y_residuals = y_train - y_global_pred

                # Use actual (non-lagged) regime for IS training
                train_regime_raw = train_data[regime_col].apply(
                    lambda v: str(int(v)) if pd.notna(v) else "nan"
                ).values

                expert_models_dict: dict = {}
                for regime_str in np.unique(train_regime_raw):
                    if regime_str == "nan":
                        continue
                    mask = train_regime_raw == regime_str
                    if mask.sum() < 30:
                        continue
                    X_reg = X_train[mask]
                    y_reg = y_residuals[mask]

                    # Use block bagging for experts only when subset is large enough
                    if use_block_bagging and len(X_reg) > block_size:
                        expert_models = _train_ebm_ensemble(
                            X_reg, y_reg, eff_expert_kwargs)
                    else:
                        em = ExplainableBoostingRegressor(**eff_expert_kwargs)
                        with parallel_backend("sequential"):
                            em.fit(X_reg, y_reg)
                        expert_models = [em]

                    expert_models_dict[regime_str] = expert_models
                    # Collect expert importances for this fold
                    imp_e = _ensemble_importances(expert_models)
                    imp_e.name = str(pred_date.date())
                    all_expert_importances.setdefault(
                        regime_str, []).append(imp_e)

                moe_ens = ResidualMoE(global_models, expert_models_dict)
                models = moe_ens  # sentinel: not None

                # IS total score (using actual regime — no lag needed for IS)
                y_global_is = moe_ens.predict_global(X_train)
                y_expert_is = np.zeros(len(X_train))
                for regime_str, exp_list in expert_models_dict.items():
                    mask = train_regime_raw == regime_str
                    if mask.sum() > 0:
                        y_expert_is[mask] = np.mean(
                            [m.predict(X_train[mask]) for m in exp_list], axis=0)
                y_pred_is = y_global_is + eff_lambda * y_expert_is

                # Global importances for this fold
                imp_g = _ensemble_importances(global_models)
                imp_g.name = str(pred_date.date())
                all_importances.append(imp_g)

                if save_models:
                    fold_path = os.path.join(
                        model_dir,
                        f"ebm_moe_{pred_date.strftime('%Y%m%d')}.pkl")
                    with open(fold_path, "wb") as f:
                        pickle.dump(moe_ens, f)

            # ── Standard single-stage EBM training ───────────────────────────
            else:
                models = _train_ebm_ensemble(X_train, y_train, ebm_kwargs)
                y_pred_is = _predict_ensemble(X_train)

                imp_avg = _ensemble_importances(models)
                imp_avg.name = str(pred_date.date())
                all_importances.append(imp_avg)

                if save_models:
                    fold_path = os.path.join(
                        model_dir,
                        f"ebm_model_{pred_date.strftime('%Y%m%d')}.pkl")
                    with open(fold_path, "wb") as f:
                        pickle.dump(models, f)

            # ── In-sample metrics (shared for both modes) ─────────────────────
            is_ic, _ = stats.spearmanr(y_pred_is, y_train)
            ss_res = np.sum((y_train - y_pred_is) ** 2)
            ss_tot = np.sum((y_train - np.mean(y_train)) ** 2)
            is_r2 = 1.0 - ss_res / (ss_tot + 1e-12)
            is_perf = _fold_portfolio_perf(
                train_data, y_pred_is, quantile,
                beta_col=beta_col if beta_neutral else None)
            is_daily_rets.update(is_perf.get("daily_rets", {}))
            all_fold_stats.append({
                "fold_date":        pred_date,
                "n_rows":           len(train_data),
                "n_dates":          len(train_dates),
                "is_ic":            float(is_ic),
                "is_r2":            float(is_r2),
                "is_sharpe":        is_perf["sharpe"],
                "is_total_return":  is_perf["total_return"],
            })

        # ── Predict ──────────────────────────────────────────────────────────
        if models is None:
            continue

        pred_data = panel[panel["ts"] == pred_date].copy()
        pred_data = _fill_features(pred_data)
        if pred_data.empty:
            continue

        X_pred = pred_data[feature_cols].values

        if moe_active and isinstance(models, ResidualMoE):
            # Use lagged + hysteresis regime to avoid look-ahead
            active_regime = regime_selector.get_regime(pred_date)
            scores = models.predict_total(X_pred, active_regime, eff_lambda)
        else:
            scores = _predict_ensemble(X_pred)

        all_preds[pred_date] = pd.Series(
            scores, index=pred_data["symbol"].values)

    if not all_preds:
        raise RuntimeError(
            "No predictions generated. Increase the date range or reduce min_train_periods.")

    predictions_wide = pd.DataFrame(all_preds).T
    predictions_wide.index.name = "ts"
    predictions_wide = predictions_wide.sort_index()

    importance_df = pd.DataFrame(
        all_importances) if all_importances else pd.DataFrame()
    fold_stats_df = pd.DataFrame(all_fold_stats)
    if not fold_stats_df.empty:
        fold_stats_df = fold_stats_df.set_index("fold_date")

    is_rets = pd.Series(is_daily_rets, name="is_ret").sort_index()

    # Convert per-regime importance lists to DataFrames
    expert_importance_dfs: dict | None = None
    if all_expert_importances:
        expert_importance_dfs = {
            r: pd.DataFrame(imp_list)
            for r, imp_list in all_expert_importances.items()
        }

    return predictions_wide, importance_df, fold_stats_df, is_rets, expert_importance_dfs


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def plot_report(
    importance_df: pd.DataFrame,
    ic_series: pd.Series,
    weights: pd.DataFrame,
    feature_cols: list[str],
    fold_stats_df: pd.DataFrame,
    oos_perf: dict,
    out_path: str,
    expert_importance_dfs: dict | None = None,
):
    """
    Generate the EBM signal report.

    When `expert_importance_dfs` is provided (MoE mode), an additional row is
    appended showing the expert feature importances for the most-represented
    regime, allowing direct Global vs Expert comparison.
    """
    has_expert = bool(expert_importance_dfs)
    n_rows = 4 if has_expert else 3
    fig_height = 18 if has_expert else 14

    fig = plt.figure(figsize=(18, fig_height))
    gs = gridspec.GridSpec(n_rows, 2, figure=fig, hspace=0.45, wspace=0.35)

    # 1. Global feature importance (full width)
    ax1 = fig.add_subplot(gs[0, :])
    title_prefix = "Global EBM" if has_expert else "EBM"
    if not importance_df.empty:
        mean_imp = importance_df.mean().sort_values(ascending=True)
        bar_colors = ["#F44336" if v <
                      0 else "#2196F3" for v in mean_imp.values]
        ax1.barh(mean_imp.index, mean_imp.values, color=bar_colors, alpha=0.85)
        ax1.axvline(0, color="black", lw=0.5)
        ax1.set_xlabel("Mean Importance (across folds)")
        ax1.set_title(
            f"{title_prefix} Feature Importances (avg across walk-forward folds)")
    else:
        ax1.text(0.5, 0.5, "No importance data", ha="center", va="center")

    # 2. OOS IC time series
    ax2 = fig.add_subplot(gs[1, 0])
    if not ic_series.empty:
        ic_roll = ic_series.rolling(21).mean()
        ax2.bar(ic_series.index, ic_series.values,
                color="#BBDEFB", alpha=0.6, label="Daily OOS IC")
        ax2.plot(ic_roll.index, ic_roll.values, color="#1565C0",
                 lw=1.5, label="21d rolling OOS IC")
        ax2.axhline(0, color="black", lw=0.5, linestyle="--")
        mean_ic = ic_series.mean()
        ir = mean_ic / (ic_series.std() + 1e-12)
        ax2.set_title(f"OOS IC  |  Mean={mean_ic:.3f}  IR={ir:.2f}")
        ax2.set_ylabel("Spearman IC")
        ax2.legend(fontsize=8)
        fig.autofmt_xdate()
    else:
        ax2.text(0.5, 0.5, "No OOS IC data", ha="center", va="center")

    # 3. IS IC per fold
    ax3 = fig.add_subplot(gs[1, 1])
    if not fold_stats_df.empty and "is_ic" in fold_stats_df.columns:
        is_ic_vals = fold_stats_df["is_ic"]
        bar_c = ["#4CAF50" if v > 0 else "#F44336" for v in is_ic_vals.values]
        ax3.bar(range(len(is_ic_vals)),
                is_ic_vals.values, color=bar_c, alpha=0.8)
        ax3.axhline(0, color="black", lw=0.5, linestyle="--")
        ax3.set_xlabel("Fold index")
        ax3.set_ylabel("Spearman IC (in-sample)")
        ax3.set_title(
            f"In-Sample IC per Fold  |  Mean={is_ic_vals.mean():.3f}")
        ax3.set_xticks(range(len(is_ic_vals)))
        ax3.set_xticklabels(
            [str(d.date()) for d in is_ic_vals.index],
            rotation=45, ha="right", fontsize=7)
    else:
        ax3.text(0.5, 0.5, "No fold stats", ha="center", va="center")

    # 4. OOS Cumulative PnL (full width, no IS overlay)
    ax4 = fig.add_subplot(gs[2, :])
    port_rets = oos_perf.get("port_rets", pd.Series(dtype=float))
    if not port_rets.empty:
        cumulative = (1 + port_rets).cumprod()
        ax4.plot(cumulative.index, cumulative.values,
                 color="#FF5722", lw=1.5, label="OOS PnL")
        ax4.fill_between(cumulative.index, 1.0, cumulative.values,
                         where=cumulative.values >= 1.0, alpha=0.12, color="#4CAF50")
        ax4.fill_between(cumulative.index, 1.0, cumulative.values,
                         where=cumulative.values < 1.0,  alpha=0.12, color="#F44336")
        ax4.axhline(1.0, color="black", lw=0.6, linestyle="--")
        ax4.set_ylabel("Cumulative Return")
        ax4.set_title("OOS Cumulative PnL", fontsize=9, pad=4)

        oos_sr = oos_perf.get("sharpe", np.nan)
        oos_ret = oos_perf.get("total_return", np.nan)
        oos_wr = oos_perf.get("win_rate", np.nan)
        stats_lines = [
            f"Sharpe  : {oos_sr:+.2f}",
            f"Return  : {oos_ret*100:+.1f}%",
            f"WinRate : {oos_wr*100:.0f}%",
        ]
        ax4.text(0.02, 0.97, "\n".join(stats_lines),
                 transform=ax4.transAxes, fontsize=7.5, family="monospace",
                 verticalalignment="top",
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                           alpha=0.85, edgecolor="#CCCCCC"))
        ax4.tick_params(axis="x", labelrotation=30, labelsize=7)
        ax4.xaxis.set_major_locator(plt.MaxNLocator(6))
    else:
        ax4.text(0.5, 0.5, "No OOS returns data", ha="center", va="center")

    # 5. Expert importances — most-represented regime (MoE only)
    if has_expert:
        ax6 = fig.add_subplot(gs[3, :])
        best_regime = max(
            expert_importance_dfs,
            key=lambda r: len(expert_importance_dfs[r])
        )
        exp_df = expert_importance_dfs[best_regime]
        if not exp_df.empty:
            mean_exp = exp_df.mean().sort_values(ascending=True)
            bar_colors_e = ["#F44336" if v < 0 else "#FF9800"
                            for v in mean_exp.values]
            ax6.barh(mean_exp.index, mean_exp.values,
                     color=bar_colors_e, alpha=0.85)
            ax6.axvline(0, color="black", lw=0.5)
            ax6.set_xlabel("Mean Importance (across folds)")
            n_folds_e = len(exp_df)
            ax6.set_title(
                f"Expert EBM Feature Importances — Regime {best_regime} "
                f"({n_folds_e} folds)  [orange = positive, red = negative]")
        else:
            ax6.text(0.5, 0.5, f"No expert importance data for regime {best_regime}",
                     ha="center", va="center")

    title = "EBM Signal Report (MoE)" if has_expert else "EBM Signal Report"
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved report → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Train an EBM signal via walk-forward and save weights.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Data ----------------------------------------------------------------
    ap.add_argument("--panel_path",  required=True,
                    help="Path to factor_panel_*.parquet from build_factor_panel.py")
    ap.add_argument("--run_id",      required=True,
                    help="Run ID — weights saved to ./reports/strategies/{run_id}/ebm.parquet")
    ap.add_argument("--features",    nargs="*", default=None,
                    help="Feature columns. Pass 'all' for every numeric column, "
                         "or omit for DEFAULT_FEATURES.")
    ap.add_argument("--include_signals", action="store_true",
                    help="Include mom_signal / rev_signal as features (meta-learning)")
    ap.add_argument("--exclude_features", nargs="*", default=[],
                    help="Feature columns to exclude (applied after --features)")

    # --- Target --------------------------------------------------------------
    ap.add_argument("--target_col",     default="ret_1d",
                    choices=["ret_1d", "ret_5d", "ret_20d"])
    ap.add_argument("--target_horizon", type=int, default=1,
                    help="Forward-shift horizon matching target_col (days)")
    ap.add_argument("--target_type",    default="raw", choices=["cs_rank", "raw"],
                    help="cs_rank: cross-sectional rank (recommended); raw: plain return")
    ap.add_argument("--target_beta_neutral", action="store_true", default=False,
                    help="Residualize training target against beta so the model learns "
                         "pure alpha. y_raw (used for PnL) is unaffected.")

    # --- Feature normalization -----------------------------------------------
    ap.add_argument("--feature_norm",  default="cs",
                    choices=["cs", "ts", "rank", "none"])
    ap.add_argument("--ts_z_window",   type=int, default=62,
                    help="Rolling window for ts normalization")

    # --- Walk-forward --------------------------------------------------------
    ap.add_argument("--train_window",      type=int, default=126,
                    help="Training window in periods (0 = expanding from start)")
    ap.add_argument("--retrain_freq",      type=int, default=21)
    ap.add_argument("--min_train_periods", type=int, default=90)

    # --- Signal construction -------------------------------------------------
    ap.add_argument("--quantile",    type=float, default=0.3,
                    help="Top/bottom quantile for long/short selection")
    ap.add_argument("--max_weight",  type=float, default=0.10,
                    help="Max absolute weight per asset")
    ap.add_argument("--weight_mode", default="zscore",
                    choices=["rank", "zscore", "raw", "equal"],
                    help="Magnitude method within selected basket. "
                         "rank=rank-proportional  zscore=z-score within basket  "
                         "raw=score-proportional  equal=equal weight")

    # --- Beta neutralization -------------------------------------------------
    ap.add_argument("--beta_neutral", action="store_true", default=False,
                    help="Beta-neutralize EBM scores before ranking")
    ap.add_argument("--no_beta_neutral", dest="beta_neutral",
                    action="store_false")
    ap.add_argument("--beta_col",     default="beta_60")

    # --- EBM hyperparameters -------------------------------------------------
    ap.add_argument("--max_rounds",           type=int,   default=200)
    ap.add_argument("--max_bins",             type=int,   default=256)
    ap.add_argument("--max_interaction_bins", type=int,   default=32)
    ap.add_argument("--interactions",         type=int,   default=15,
                    help="Number of pairwise interaction terms for Global EBM (0 = none)")
    ap.add_argument("--learning_rate",        type=float, default=0.01)
    ap.add_argument("--inner_bags",           type=int,   default=5,
                    help="Inner bags per boosting round (0 = EBM default, no bagging)")
    ap.add_argument("--outer_bags",           type=int,   default=1,
                    help="Outer bags for ensemble averaging (EBM default=8)")
    ap.add_argument("--min_samples_leaf",     type=int,   default=30)
    ap.add_argument("--random_state",         type=int,   default=42)

    # --- Block bagging -------------------------------------------------------
    ap.add_argument("--use_block_bagging", action="store_true",
                    help="Replace EBM outer_bags with manual block-bootstrap ensemble")
    ap.add_argument("--n_outer_bags",  type=int,   default=8,
                    help="Number of block-bootstrap bags (only when --use_block_bagging)")
    ap.add_argument("--block_size",    type=int,   default=21,
                    help="Block length in rows for block bootstrap (≈1 trading month)")
    ap.add_argument("--embargo_pct",   type=float, default=0.01,
                    help="Fraction of train window to embargo beyond target_horizon")

    # --- Mixture of Experts --------------------------------------------------
    ap.add_argument("--use_moe", action="store_true",
                    help="Enable Residual-Based Mixture of Experts (MoE) ensemble")
    ap.add_argument("--regime_col", default="trend_regime_enc",
                    help="Numeric column used for regime gating (must be integer-like). "
                         "Common choices: volatility_regime_enc, trend_regime_enc, "
                         "skew_regime_enc.")
    ap.add_argument("--moe_boost_lambda", type=float, default=0.5,
                    help="Expert contribution weight: Score = Global + λ × Expert")
    ap.add_argument("--expert_interactions", type=int, default=10,
                    help="Pairwise interaction terms for Expert EBMs (independent of "
                         "--interactions which applies to the Global EBM)")
    ap.add_argument("--expert_learning_rate", type=float, default=None,
                    help="Learning rate for Expert EBMs. Defaults to --learning_rate.")
    ap.add_argument("--moe_hysteresis", type=int, default=3,
                    help="Consecutive days the new regime must persist before switching "
                         "the active expert (reduces expert turnover).")

    # --- Output --------------------------------------------------------------
    ap.add_argument("--save_models", action="store_true",
                    help="Pickle each fold's EBM model (can be large)")

    args = ap.parse_args()

    # ── Paths ─────────────────────────────────────────────────────────────────
    out_dir = ensure_dir(f"./reports/strategies/{args.run_id}")
    model_dir = ensure_dir(os.path.join(
        out_dir, "ebm_models")) if args.save_models else None

    # ── Load panel ────────────────────────────────────────────────────────────
    print(f"Loading panel: {args.panel_path}")
    panel = pd.read_parquet(args.panel_path)
    panel["ts"] = pd.to_datetime(panel["ts"])
    panel = panel.sort_values(["ts", "symbol"]).reset_index(drop=True)
    print(f"  {len(panel):,} rows  |  {panel['symbol'].nunique()} symbols  |  "
          f"{panel['ts'].nunique()} dates")

    # ── Resolve feature columns ───────────────────────────────────────────────
    if args.features is not None and args.features not in (["all"], ["filtered"]):
        feature_cols = args.features
    elif args.features == ["all"]:
        numeric = panel.select_dtypes(include=[np.number]).columns.tolist()
        exclude = _META_COLS | _SIGNAL_COLS | {"y"}
        feature_cols = [c for c in numeric if c not in exclude]
    else:
        feature_cols = [c for c in DEFAULT_FEATURES if c in panel.columns]

    if args.include_signals:
        for sig in ["mom_signal", "rev_signal"]:
            if sig in panel.columns and sig not in feature_cols:
                feature_cols.append(sig)

    if args.exclude_features:
        feature_cols = [
            c for c in feature_cols if c not in args.exclude_features]

    feature_cols = [c for c in feature_cols if c in panel.columns]
    print(f"\nFeatures ({len(feature_cols)}): {feature_cols}")

    # ── Build target  (MUST precede normalize_features) ──────────────────────
    print(f"Building target: {args.target_type}({args.target_col}, "
          f"horizon={args.target_horizon})"
          + (" [beta-neutral]" if args.target_beta_neutral else ""))
    panel = build_target(
        panel, args.target_col, args.target_horizon, args.target_type,
        beta_neutral=args.target_beta_neutral, beta_col=args.beta_col,
    )

    # ── Normalize features ────────────────────────────────────────────────────
    print(f"\nNormalizing features (mode={args.feature_norm})...")
    panel = normalize_features(
        panel, feature_cols, args.feature_norm, args.ts_z_window)

    # ── Build EBM hyperparameter dicts ────────────────────────────────────────
    ebm_kwargs = dict(
        max_rounds=args.max_rounds,
        max_bins=args.max_bins,
        max_interaction_bins=args.max_interaction_bins,
        interactions=args.interactions,
        learning_rate=args.learning_rate,
        inner_bags=args.inner_bags,
        outer_bags=args.outer_bags,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.random_state,
        feature_names=feature_cols,
    )

    # Expert EBMs inherit global kwargs but override interactions and optionally lr.
    expert_ebm_kwargs: dict | None = None
    if args.use_moe:
        expert_ebm_kwargs = {**ebm_kwargs,
                             "interactions": args.expert_interactions}
        if args.expert_learning_rate is not None:
            expert_ebm_kwargs["learning_rate"] = args.expert_learning_rate

    print(f"\nStarting walk-forward EBM training...")
    print(f"  train_window={args.train_window}  retrain_freq={args.retrain_freq}  "
          f"min_train_periods={args.min_train_periods}")
    if args.use_moe:
        print(f"  MoE: regime_col={args.regime_col}  "
              f"λ={args.moe_boost_lambda}  "
              f"expert_interactions={args.expert_interactions}  "
              f"hysteresis={args.moe_hysteresis}")

    (predictions_wide, importance_df,
     fold_stats_df, is_rets,
     expert_importance_dfs) = walk_forward(
        panel=panel,
        feature_cols=feature_cols,
        target_col=args.target_col,
        target_horizon=args.target_horizon,
        target_type=args.target_type,
        train_window=args.train_window,
        retrain_freq=args.retrain_freq,
        min_train_periods=args.min_train_periods,
        beta_neutral=args.beta_neutral,
        beta_col=args.beta_col,
        quantile=args.quantile,
        ebm_kwargs=ebm_kwargs,
        save_models=args.save_models,
        model_dir=model_dir,
        use_block_bagging=args.use_block_bagging,
        n_outer_bags=args.n_outer_bags,
        block_size=args.block_size,
        embargo_pct=args.embargo_pct,
        use_moe=args.use_moe,
        regime_col=args.regime_col,
        moe_boost_lambda=args.moe_boost_lambda,
        moe_hysteresis=args.moe_hysteresis,
        expert_ebm_kwargs=expert_ebm_kwargs,
    )
    print(f"  Prediction matrix: {predictions_wide.shape}")

    # ── Beta neutralization of scores (before ranking) ───────────────────────
    if args.beta_neutral:
        print(f"\nBeta-neutralizing scores (col={args.beta_col})...")
        predictions_wide = neutralize_scores(
            predictions_wide, panel, beta_col=args.beta_col)
        print("  Done.")

    # ── Predictions → weights ────────────────────────────────────────────────
    print(f"\nConverting predictions to weights (mode={args.weight_mode}, "
          f"Q={args.quantile}, maxW={args.max_weight})...")
    weights = predictions_to_weights(
        predictions_wide,
        quantile=args.quantile,
        max_weight=args.max_weight,
        weight_mode=args.weight_mode,
    )

    # ── OOS metrics ──────────────────────────────────────────────────────────
    print("Computing OOS metrics...")
    ic_series = compute_ic(predictions_wide, panel,
                           args.target_col, args.target_horizon)
    oos_perf = compute_portfolio_performance(weights, panel)
    mean_ic = ic_series.mean()
    ic_std = ic_series.std()
    ir = mean_ic / (ic_std + 1e-12)
    ic_pos = (ic_series > 0).mean()
    print(f"  Mean IC = {mean_ic:.4f}  |  IC Std = {ic_std:.4f}  |  "
          f"IR = {ir:.2f}  |  IC > 0: {ic_pos:.1%}")

    # ── Save outputs ──────────────────────────────────────────────────────────
    weights.to_parquet(os.path.join(out_dir, "ebm.parquet"))
    print(f"\nWeights saved → {out_dir}/ebm.parquet")

    predictions_wide.to_parquet(os.path.join(
        out_dir, "ebm_predictions.parquet"))
    print(f"Predictions saved → {out_dir}/ebm_predictions.parquet")

    if not importance_df.empty:
        imp_path = os.path.join(out_dir, "ebm_feature_importance.csv")
        importance_df.to_csv(imp_path)
        print(f"Feature importances saved → {imp_path}")
        mean_imp = importance_df.mean().sort_values(ascending=False)
        n_folds = len(importance_df)
        present = importance_df.notna().sum()
        n_show = min(12, len(mean_imp))
        sep = "─" * 58
        print(f"\n  {sep}")
        print(f"  {'Feature':<30s} {'Importance':>10s}  {'Folds':>8s}")
        print(f"  {sep}")
        for feat, val in mean_imp.head(n_show).items():
            k = int(present.get(feat, 0))
            print(f"    {feat:<30s} {val:>10.4f}  {k:>4d}/{n_folds:<3d}")
        print(f"  {sep}")
        for feat, val in mean_imp.tail(n_show).iloc[::-1].items():
            k = int(present.get(feat, 0))
            print(f"    {feat:<30s} {val:>10.4f}  {k:>4d}/{n_folds:<3d}")
        print(f"  {sep}")
        is_interaction = mean_imp.index.to_series().str.contains(" & ")
        zero_int = mean_imp[is_interaction & (mean_imp.abs() < 1e-4)]
        if len(zero_int):
            print(f"  {len(zero_int)} interaction term(s) averaged < 1e-4 "
                  f"(FAST-proposed but boosting-inert). "
                  f"Consider lowering --interactions.")

    # Save expert importances (MoE mode)
    if expert_importance_dfs:
        for regime_str, exp_df in expert_importance_dfs.items():
            exp_imp_path = os.path.join(
                out_dir, f"ebm_expert_importance_regime_{regime_str}.csv")
            exp_df.to_csv(exp_imp_path)
            print(
                f"Expert importance (regime={regime_str}) saved → {exp_imp_path}")

    if not fold_stats_df.empty:
        fold_path = os.path.join(out_dir, "ebm_fold_stats.csv")
        fold_stats_df.to_csv(fold_path)
        print(f"Fold stats saved → {fold_path}")

        mean_is_ic = fold_stats_df["is_ic"].mean()
        mean_is_r2 = fold_stats_df["is_r2"].mean()
        mean_is_sr = fold_stats_df["is_sharpe"].mean()
        mean_is_ret = fold_stats_df["is_total_return"].mean()
        oos_sr = oos_perf.get("sharpe",       np.nan)
        oos_ret = oos_perf.get("total_return",  np.nan)
        oos_wr = oos_perf.get("win_rate",      np.nan)
        overfit_ic = mean_is_ic / \
            (mean_ic + 1e-12) if mean_ic != 0 else float("nan")
        overfit_sr = (mean_is_sr / (oos_sr + 1e-12)
                      if not np.isnan(oos_sr) and oos_sr != 0 else float("nan"))

        sep = "─" * 50
        print(f"\n{sep}")
        print(f"  {'Metric':<25s}  {'In-Sample':>10s}  {'OOS':>10s}")
        print(sep)
        print(f"  {'IC (Spearman)':<25s}  {mean_is_ic:>10.4f}  {mean_ic:>10.4f}")
        print(f"  {'R²':<25s}  {mean_is_r2:>10.4f}  {'—':>10s}")
        print(f"  {'Sharpe (ann.)':<25s}  {mean_is_sr:>10.2f}  {oos_sr:>10.2f}")
        print(
            f"  {'Total Return':<25s}  {mean_is_ret*100:>9.1f}%  {oos_ret*100:>9.1f}%")
        print(f"  {'Win Rate':<25s}  {'—':>10s}  {oos_wr*100:>9.0f}%")
        print(sep)
        print(
            f"  IS/OOS IC ratio : {overfit_ic:.2f}  |  Sharpe ratio : {overfit_sr:.2f}")
        overfit_label = ("severe overfit" if overfit_ic > 5 else
                         "moderate overfit" if overfit_ic > 2 else "acceptable")
        print(f"  Overfit signal  : {overfit_label}")
        print(sep)

    port_rets = oos_perf.get("port_rets", pd.Series(dtype=float))
    if not port_rets.empty:
        port_rets.to_csv(os.path.join(out_dir, "ebm_oos_returns.csv"),
                         header=["port_ret"])

    plot_report(importance_df, ic_series, weights, feature_cols,
                fold_stats_df, oos_perf,
                out_path=os.path.join(out_dir, "ebm_report.png"),
                expert_importance_dfs=expert_importance_dfs)

    # ── Config summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  EBM Signal Summary")
    print("=" * 55)
    print(f"  Panel          : {args.panel_path}")
    print(f"  Features       : {len(feature_cols)}")
    print(f"  Target         : {args.target_type}({args.target_col}, h={args.target_horizon})"
          + (" beta-neutral" if args.target_beta_neutral else ""))
    print(f"  Feature norm   : {args.feature_norm}")
    print(f"  Train window   : {args.train_window} "
          f"({'expanding' if args.train_window == 0 else 'rolling'})")
    print(f"  Retrain freq   : {args.retrain_freq}")
    print(
        f"  Weight mode    : {args.weight_mode}  Q={args.quantile}  maxW={args.max_weight}")
    print(
        f"  EBM max_rounds : {args.max_rounds}  interactions={args.interactions}")
    if args.use_moe:
        print(f"  MoE mode       : ON  regime={args.regime_col}  "
              f"λ={args.moe_boost_lambda}  "
              f"expert_interactions={args.expert_interactions}  "
              f"hysteresis={args.moe_hysteresis}")
        if expert_importance_dfs:
            regimes_trained = sorted(expert_importance_dfs.keys())
            print(f"  Expert regimes : {regimes_trained}")
    print(f"  OOS IC         : {mean_ic:.4f}  IR={ir:.2f}  IC>0={ic_pos:.1%}")
    print(f"  OOS Sharpe     : {oos_perf.get('sharpe', float('nan')):.2f}")
    print(
        f"  OOS Total Ret  : {oos_perf.get('total_return', float('nan'))*100:.1f}%")
    if not fold_stats_df.empty:
        print(f"  IS  IC  (avg)  : {fold_stats_df['is_ic'].mean():.4f}")
        print(f"  IS  R²  (avg)  : {fold_stats_df['is_r2'].mean():.4f}")
        print(f"  IS  Sharpe     : {fold_stats_df['is_sharpe'].mean():.2f}")
        print(
            f"  IS  Total Ret  : {fold_stats_df['is_total_return'].mean()*100:.1f}%")
    print(f"  Output         : {out_dir}")
    print("=" * 55)


if __name__ == "__main__":
    main()
