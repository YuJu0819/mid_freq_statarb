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
from ...core.utils import ensure_dir
from ...data.rolling_universe import (
    RollingUniverse, build_symbol_active_mask, resolve_epochs,
)
from ...alpha.ml_utils import (
    normalize_features,
    neutralize_features_on_adx,
    build_target,
    predictions_to_weights,
    neutralize_scores,
    compute_portfolio_performance,
    compute_ic,
)
from ...alpha.ho_moe import (
    MACRO_CANDIDATES,
    compute_macro_candidates,
    discover_regime_separator_cmi,
    fit_macro_regime_bin_edges,
    apply_macro_regime_bin_edges,
)

import argparse
import os
import pickle
import warnings

from joblib import Parallel, delayed

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
    "volatility_30", "vol_rank_cs", "beta_60", "skewness_45",
    # liquidity (NEW)
    "liquidity_log", "liquidity_rank_cs",
    # momentum
    "price_roc", "oi_roc",
    "basis_norm", "basis_norm_smooth3",  # raw + 3d-smoothed companion
    "basis_mom",
    "vol_ratio_sig", "trend_score", "sentiment_score", "combined_score",
    "funding_z", "funding_penalty", "mom_final_score",
    # reversal
    "ls_ratio",
    "ls_chg_1d", "ls_chg_smooth3",          # raw + 3d-smoothed companion
    "oi_pct_chg_1d", "oi_pct_chg_smooth3",  # raw + 3d-smoothed companion
    "cs_z_oi_chg", "ts_z_oi_chg", "liquidation_shock",
    "regime_score", "interaction_alpha", "reversal_hawkes", "rev_final_score",
    # market regime — numeric-encoded
    "market_adx",
    "volatility_regime_enc",
    "trend_regime_enc",
    "skew_regime_enc",
    # delta family (rolling-5 mean minus lag-10 of itself)
    # Only factors with lookback <= 30d; long-window factors (beta_60, skewness_45,
    # funding_z, liquidation_shock, regime_score, interaction_alpha) are excluded
    # because a 10-day delta on a 45-180d stat is near-constant and uninformative.
    "volatility_30_delta", "vol_rank_cs_delta",
    "price_roc_delta", "oi_roc_delta", "basis_norm_delta", "basis_mom_delta",
    "vol_ratio_sig_delta", "trend_score_delta", "sentiment_score_delta",
    "combined_score_delta", "mom_final_score_delta",
    "ls_ratio_delta", "rev_final_score_delta",
    "liquidity_rank_cs_delta",
]
_FILTERED_COLS = {
    'ls_chg_smooth3', 'oi_pct_chg_smooth3', 'basis_norm_smooth3', 'market_volatility', 'cs_z_oi_chg'
}

# ---------------------------------------------------------------------------
# EBM walk-forward training
# ---------------------------------------------------------------------------


# Phase-4a refactor: pure helpers lifted to src/alpha/ebm_utils.py.
# Re-exports keep every existing call site (including external imports of
# `src.scripts.train_ebm_signal._fold_portfolio_perf` etc.) working.
from ...alpha.ebm_utils import (  # noqa: E402,F401
    _fold_portfolio_perf,
    _embargo_gap,
    _block_bootstrap_counts,
    _ensemble_importances,
)


# ---------------------------------------------------------------------------
# Regime-gating helpers (MoE)
# ---------------------------------------------------------------------------


# Phase-4a refactor: RegimeSelector lifted to src/alpha/moe.py.
from ...alpha.moe import RegimeSelector  # noqa: E402,F401


# ResidualMoE lives in src.alpha.residual_moe so loky workers can pickle
# instances of it. When the class was defined here, running this file via
# `python -m src.scripts.train_ebm_signal` registered it under __main__,
# and worker processes (which re-import the module under its real path)
# couldn't resolve __main__.ResidualMoE → PicklingError.
from ...alpha.residual_moe import ResidualMoE  # noqa: E402
# Phase-4b refactor: bag-matrix construction + ensemble training. The
# imports are aliased so the thin closure wrappers inside walk_forward
# (which keep the legacy `_make_temporal_bags` / `_train_ebm_ensemble`
# names) don't shadow the module-level versions.
from ...alpha.bagging import (  # noqa: E402
    make_temporal_bags as _bagging_make_temporal_bags,
    train_ebm_ensemble as _bagging_train_ebm_ensemble,
)


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
    bag_symbol_frac: float = 1.0,
    bag_sym_excluded_as_val: bool = False,
    # ── MoE ────────────────────────────────────────────────────────────
    use_moe: bool = False,
    regime_col: str = "volatility_regime_enc",
    moe_boost_lambda: float = 0.5,
    moe_hysteresis: int = 3,
    expert_ebm_kwargs: dict | None = None,
    regime_selector: "RegimeSelector | None" = None,
    expert_panel: "pd.DataFrame | None" = None,
    pred_start_date: "pd.Timestamp | None" = None,
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

    # Effective prediction-start index. If pred_start_date is provided,
    # predictions begin at the first panel date >= pred_start_date.
    # Pre-pred_start dates are still kept in the panel as warmup history
    # for training, EWM features, and rolling lookbacks.
    pred_start_idx = min_train_periods
    if pred_start_date is not None:
        ps = pd.Timestamp(pred_start_date)
        candidates = [i for i, d in enumerate(dates) if pd.Timestamp(d) >= ps]
        if candidates:
            pred_start_idx = max(min_train_periods, candidates[0])

    all_preds: dict = {}
    all_importances: list = []
    all_fold_stats: list = []
    all_expert_importances: dict = {}  # regime_str -> list[pd.Series]
    is_daily_rets: dict = {}

    # Retrain schedule anchored to pred_start_idx so the first OOS prediction
    # date triggers an immediate retrain.
    retrain_set = {
        i for i in range(pred_start_idx, label_cutoff_idx)
        if (i - pred_start_idx) % retrain_freq == 0
    }

    print(f"  Dates total: {n_dates}  |  "
          f"Prediction dates: {label_cutoff_idx - pred_start_idx}  |  "
          f"First pred = {dates[pred_start_idx] if pred_start_idx < n_dates else 'N/A'}  |  "
          f"Retraining {len(retrain_set)} times (every {retrain_freq} periods)")
    if use_block_bagging:
        print(f"  Block bagging ON: {n_outer_bags} bags  "
              f"block_size={block_size}  embargo_pct={embargo_pct}")
    print(f"  Temporal validation holdout: last max(target_horizon={target_horizon}, "
          f"block_size={block_size}) rows per bag (prevents early-stopping leakage)")

    # ── MoE initialisation ────────────────────────────────────────────────
    moe_active = False  # set True once regime_col is confirmed present
    # `regime_selector` may be injected from outside (built from the raw panel
    # before normalize_features, so it reads correct integer regime labels).
    # If not provided, fall back to building from the current (possibly
    # normalized) panel — legacy behaviour.
    _internal_regime_selector: RegimeSelector | None = regime_selector
    if use_moe:
        if regime_col not in panel.columns:
            print(f"  [warn] regime_col='{regime_col}' not found in panel; "
                  f"MoE disabled.")
        else:
            moe_active = True
            if _internal_regime_selector is None:
                _internal_regime_selector = RegimeSelector(
                    panel, regime_col, hysteresis=moe_hysteresis)
            regime_selector = _internal_regime_selector
            eff_lambda = moe_boost_lambda
            eff_expert_kwargs = expert_ebm_kwargs or ebm_kwargs
            n_expert_jobs = max(1, os.cpu_count() or 1)
            print(f"  MoE ON: regime_col={regime_col}  "
                  f"λ={eff_lambda}  hysteresis={moe_hysteresis}  "
                  f"expert_parallel_jobs={n_expert_jobs}")

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

    # Phase-4b refactor: both helpers lifted to src/alpha/bagging.py with
    # their closure variables (block_size, target_horizon, rng,
    # use_block_bagging, n_outer_bags, bag_symbol_frac,
    # bag_sym_excluded_as_val) promoted to explicit keyword-only params.
    # Thin local wrappers below preserve the original signatures so every
    # interior call site inside walk_forward stays unchanged.
    def _make_temporal_bags(
        n: int, n_bags: int, use_blocks: bool,
        date_arr: "np.ndarray | None" = None,
        symbol_arr: "np.ndarray | None" = None,
        symbol_frac: float = 1.0,
        sym_excluded_as_val: bool = False,
    ) -> np.ndarray:
        return _bagging_make_temporal_bags(
            n, n_bags, use_blocks,
            date_arr=date_arr, symbol_arr=symbol_arr,
            symbol_frac=symbol_frac,
            sym_excluded_as_val=sym_excluded_as_val,
            block_size=block_size,
            target_horizon=target_horizon,
            rng=rng,
        )

    def _train_ebm_ensemble(
        X: np.ndarray, y: np.ndarray, kwargs: dict,
        date_arr: "np.ndarray | None" = None,
        symbol_arr: "np.ndarray | None" = None,
        force_no_bagging: bool = False,
    ) -> list:
        return _bagging_train_ebm_ensemble(
            X, y, kwargs,
            date_arr=date_arr, symbol_arr=symbol_arr,
            force_no_bagging=force_no_bagging,
            use_block_bagging=use_block_bagging,
            n_outer_bags=n_outer_bags,
            block_size=block_size,
            bag_symbol_frac=bag_symbol_frac,
            bag_sym_excluded_as_val=bag_sym_excluded_as_val,
            target_horizon=target_horizon,
            rng=rng,
        )

    # Phase-4a refactor: _ensemble_importances is now provided by
    # src/alpha/ebm_utils.py (imported at module top). The previous nested
    # definition was byte-identical and has been removed.

    for pred_idx in range(pred_start_idx, label_cutoff_idx):
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

            # If expert_panel is provided (ADX-neutralized), extract aligned
            # expert features for the same (ts, symbol) rows as train_data.
            X_train_expert: np.ndarray | None = None
            if expert_panel is not None:
                ep_train = (
                    expert_panel[expert_panel["ts"].isin(train_dates)]
                    .copy()
                )
                ep_train = _fill_features(ep_train)
                # Align rows to match train_data order via ts+symbol merge
                ep_train = (
                    train_data[["ts", "symbol"]]
                    .reset_index(drop=True)
                    .merge(ep_train, on=["ts", "symbol"], how="left")
                )
                X_train_expert = ep_train[feature_cols].fillna(0.0).values

            if not _first_train_logged:
                print(f"  [info] First model fit at pred_idx={pred_idx} "
                      f"using {len(train_data)} rows ({len(train_dates)} dates × "
                      f"{len(train_data) // max(len(train_dates), 1)} avg symbols)"
                      + (" [global on ADX-neutral, experts on raw features]"
                         if X_train_expert is not None else ""))
                _first_train_logged = True

            # ── Two-stage MoE training ────────────────────────────────────────
            if moe_active:
                # Stage 1: Global EBM on original features
                global_models = _train_ebm_ensemble(
                    X_train, y_train, ebm_kwargs,
                    date_arr=train_data["ts"].values,
                    symbol_arr=train_data["symbol"].values)
                y_global_pred = np.mean(
                    [m.predict(X_train) for m in global_models], axis=0)

                # Stage 2: Expert EBMs on residuals.
                # Use ADX-neutral features if expert_panel was provided,
                # otherwise fall back to the same X_train as global.
                y_residuals = y_train - y_global_pred
                X_for_experts = X_train_expert if X_train_expert is not None else X_train

                # Use actual (non-lagged) regime for IS training
                train_regime_raw = train_data[regime_col].apply(
                    lambda v: str(int(v)) if pd.notna(v) else "nan"
                ).values

                # Build list of regime training tasks (skip nan / too-small)
                regime_tasks = []
                for regime_str in np.unique(train_regime_raw):
                    if regime_str == "nan":
                        continue
                    mask = train_regime_raw == regime_str
                    if mask.sum() < 30:
                        breakpoint()
                        continue
                    regime_tasks.append((
                        regime_str, mask,
                        X_for_experts[mask],
                        y_residuals[mask],
                        train_data["ts"].values[mask],
                    ))

                def _fit_one_expert(X_reg, y_reg, date_arr_reg):
                    return _train_ebm_ensemble(
                        X_reg, y_reg, eff_expert_kwargs,
                        date_arr=date_arr_reg, force_no_bagging=True)

                # Train expert EBMs in parallel across regimes.
                # Use threading backend so we avoid nested loky workers when
                # this function itself is being run inside a loky worker
                # (per-epoch parallelism). Skip the Parallel call entirely if
                # there are no eligible regimes for this fold.
                if regime_tasks:
                    n_inner_jobs = max(
                        1, min(len(regime_tasks), n_expert_jobs))
                    expert_results = Parallel(
                        n_jobs=n_inner_jobs,
                        backend="threading",
                    )(
                        delayed(_fit_one_expert)(X_r, y_r, d_r)
                        for _, _, X_r, y_r, d_r in regime_tasks
                    )
                else:
                    expert_results = []

                expert_models_dict: dict = {}
                for (regime_str, _mask, *_), expert_models in zip(
                        regime_tasks, expert_results):
                    expert_models_dict[regime_str] = expert_models
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
                models = _train_ebm_ensemble(
                    X_train, y_train, ebm_kwargs,
                    date_arr=train_data["ts"].values,
                    symbol_arr=train_data["symbol"].values)
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
            # Expert inference uses ADX-neutral features if expert_panel provided
            X_pred_expert: np.ndarray | None = None
            if expert_panel is not None:
                ep_pred = expert_panel[expert_panel["ts"] == pred_date].copy()
                ep_pred = _fill_features(ep_pred)
                if not ep_pred.empty:
                    ep_pred = (
                        pred_data[["ts", "symbol"]]
                        .reset_index(drop=True)
                        .merge(ep_pred, on=["ts", "symbol"], how="left")
                    )
                    X_pred_expert = ep_pred[feature_cols].fillna(0.0).values
            scores = models.predict_total(
                X_pred, active_regime, eff_lambda, X_expert=X_pred_expert)
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
# HO-MoE walk-forward (CMI tournament + TS neutralization + market-wide MoE)
# ---------------------------------------------------------------------------


def walk_forward_ho_moe(
    panel: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    target_horizon: int,
    target_type: str,
    train_window: int,
    retrain_freq: int,
    min_train_periods: int,
    quantile: float,
    ebm_kwargs: dict,
    expert_ebm_kwargs: dict,
    save_models: bool,
    model_dir: str,
    beta_neutral: bool = False,
    beta_col: str = "beta_60",
    use_block_bagging: bool = False,
    n_outer_bags: int = 8,
    block_size: int = 21,
    embargo_pct: float = 0.01,
    # HO-MoE specifics
    moe_boost_lambda: float = 0.5,
    moe_hysteresis: int = 3,
    cmi_candidates: tuple = MACRO_CANDIDATES,
    cmi_ema_span: int = 3,
    cmi_q_target: int = 3,
    cmi_q_features: int = 5,
    cmi_q_candidates: int = 3,
    ts_neutral_window: int = 30,
    ts_neutralize: bool = True,
    n_regimes: int = 3,
    fix_separator: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, dict | None, pd.DataFrame, pd.DataFrame]:
    """
    Walk-forward EBM with the HO-MoE pipeline:

      * Pass 1 (every retrain) — CMI tournament across `cmi_candidates`
        on the trailing IS panel. EMA(span=cmi_ema_span) smooths per-candidate
        Average_CMI scores; the highest-scoring candidate wins and becomes
        the active Regime_Separator.

      * Pass 2 (every retrain) — when the winner changes, the panel is
        TS-neutralized against the new winner (per-symbol rolling OLS via
        `neutralize_features_on_adx`) and the macro regime label is re-
        derived from IS-fit time-series quantile edges.  A market-wide
        `RegimeSelector` (with hysteresis) gates expert selection at OOS.

      * Training — Global EBM on the TS-neutralized features predicts raw
        returns; per-regime Expert EBMs are trained on the raw normalized
        features against the global model's residuals.

    Returns the same five outputs as `walk_forward` plus a sixth — the
    per-fold CMI tournament log.
    """
    panel = panel.copy()
    missing = [c for c in cmi_candidates if c not in panel.columns]
    if missing:
        raise ValueError(
            f"walk_forward_ho_moe: macro candidate column(s) missing from "
            f"panel: {missing}. Run compute_macro_candidates() upstream.")

    dates = sorted(panel["ts"].unique())
    n_dates = len(dates)
    label_cutoff_idx = n_dates - target_horizon

    all_preds: dict = {}
    all_importances: list = []
    all_fold_stats: list = []
    all_expert_importances: dict = {}
    is_daily_rets: dict = {}
    cmi_log_rows: list = []
    regime_timeline_rows: list = []  # one row per OOS pred_date

    retrain_set = {
        i for i in range(min_train_periods, label_cutoff_idx)
        if (i - min_train_periods) % retrain_freq == 0
    }

    print(f"  Dates total: {n_dates}  |  "
          f"Prediction dates: {label_cutoff_idx - min_train_periods}  |  "
          f"Retraining {len(retrain_set)} times (every {retrain_freq} periods)")
    if fix_separator is not None:
        print(f"  HO-MoE: FIXED separator={fix_separator!r}  "
              f"(CMI tournament skipped)  "
              f"ts_neutralize={ts_neutralize}"
              + (f"  ts_neutral_window={ts_neutral_window}" if ts_neutralize else "")
              + f"  regimes={n_regimes}  λ={moe_boost_lambda}  "
              f"hysteresis={moe_hysteresis}")
    else:
        print(f"  HO-MoE: candidates={list(cmi_candidates)}  "
              f"ema_span={cmi_ema_span}  "
              f"ts_neutralize={ts_neutralize}"
              + (f"  ts_neutral_window={ts_neutral_window}" if ts_neutralize else "")
              + f"  regimes={n_regimes}  λ={moe_boost_lambda}  "
              f"hysteresis={moe_hysteresis}")

    # ── State ────────────────────────────────────────────────────────────
    ema_history: dict = {}
    current_winner: str | None = None
    panel_neutral: pd.DataFrame | None = None
    regime_selector: RegimeSelector | None = None
    models: ResidualMoE | None = None
    rng = np.random.default_rng(ebm_kwargs.get("random_state", 42))
    n_expert_jobs = max(1, os.cpu_count() or 1)
    LABEL_COL = "_ho_moe_regime_enc"

    def _fill_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in feature_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0.0)
        return df

    def _make_temporal_bags(
        n: int, n_bags: int, use_blocks: bool,
        date_arr: np.ndarray | None,
    ) -> np.ndarray:
        # Same shape as walk_forward's nested helper — date-aware split with
        # validation tail reserved for early stopping.
        if date_arr is None:
            raise ValueError("HO-MoE: date_arr required for temporal bags.")
        unique_dates = np.unique(date_arr)
        n_d = len(unique_dates)
        date_to_idx = {d: i for i, d in enumerate(unique_dates)}
        row_date_idx = np.array([date_to_idx[d] for d in date_arr])
        # val_dates = min(max(target_horizon, block_size), int(n_d * 0.1))
        val_dates = int(n_d * 0.2)
        train_date_cutoff = n_d - val_dates
        train_row_mask = row_date_idx < train_date_cutoff
        mat = np.zeros((n, n_bags), dtype=np.int8)
        mat[~train_row_mask, :] = -1
        if use_blocks:
            for b in range(n_bags):
                sel_date_idx = np.unique(np.concatenate([
                    np.arange(s, min(s + block_size, train_date_cutoff))
                    for s in rng.integers(
                        0, max(1, train_date_cutoff - block_size + 1),
                        size=max(1, train_date_cutoff // max(block_size, 1)))
                ]))
                sel_row_mask = np.isin(
                    row_date_idx, sel_date_idx) & train_row_mask
                mat[sel_row_mask, b] = 1
        else:
            mat[train_row_mask, :] = 1
        return mat

    def _train_ebm_ensemble(
        X: np.ndarray, y: np.ndarray, kwargs: dict,
        date_arr: np.ndarray,
        force_no_bagging: bool = False,
    ) -> list:
        n = len(X)
        n_bags = (n_outer_bags if (use_block_bagging and not force_no_bagging)
                  else kwargs.get("outer_bags", 8))
        use_blocks = use_block_bagging and not force_no_bagging
        bags_matrix = _make_temporal_bags(n, n_bags, use_blocks, date_arr)
        kw = {**kwargs, "outer_bags": n_bags}
        m = ExplainableBoostingRegressor(**kw)
        m.fit(X, y, bags=bags_matrix)
        return [m]

    # Phase-4a refactor: _ensemble_importances is now provided by
    # src/alpha/ebm_utils.py (imported at module top). The previous nested
    # definition was byte-equivalent (only the local-variable name `imps`
    # differed) and has been removed.

    _first_train_logged = False

    for pred_idx in range(min_train_periods, label_cutoff_idx):
        pred_date = dates[pred_idx]

        # ── Retrain ──────────────────────────────────────────────────────
        if pred_idx in retrain_set:
            gap = _embargo_gap(
                min(pred_idx, train_window if train_window > 0 else pred_idx),
                target_horizon, embargo_pct,
            )
            train_end_idx = max(0, pred_idx - gap)
            train_start_idx = (
                max(0, train_end_idx - train_window) if train_window > 0 else 0
            )
            train_dates = dates[train_start_idx:train_end_idx]
            train_data = panel[panel["ts"].isin(train_dates)].copy()
            train_data = train_data.dropna(subset=["y"])
            train_data = _fill_features(train_data)

            if len(train_data) < 50:
                if not _first_train_logged:
                    print(f"  [warn] pred_idx={pred_idx}: only {len(train_data)} "
                          f"valid training rows (need 50). Waiting for more data...")
                continue

            # ── Pass 1: separator selection ─────────────────────────────
            # When fix_separator is set the CMI tournament is skipped — every
            # fold uses the same column as Regime_Separator. Useful for A/B
            # testing individual macro candidates against the legacy static
            # volatility_regime_enc baseline.
            if fix_separator is not None:
                if fix_separator not in train_data.columns:
                    raise ValueError(
                        f"fix_separator='{fix_separator}' not in panel columns.")
                new_winner = fix_separator
                diag = pd.DataFrame({
                    "raw_cmi": {c: np.nan for c in cmi_candidates},
                    "ema_cmi": {c: np.nan for c in cmi_candidates},
                })
            else:
                new_winner, ema_history, diag = discover_regime_separator_cmi(
                    train_data, feature_cols, ema_history,
                    target_col="y",  # raw return — built via build_target upstream
                    candidates=cmi_candidates,
                    ema_span=cmi_ema_span,
                    q_target=cmi_q_target,
                    q_features=cmi_q_features,
                    q_candidates=cmi_q_candidates,
                )
            row = {"fold_date": pred_date, "winner": new_winner}
            for r in cmi_candidates:
                row[f"raw_cmi_{r}"] = float(diag["raw_cmi"].get(r, np.nan))
                row[f"ema_cmi_{r}"] = float(diag["ema_cmi"].get(r, np.nan))
            cmi_log_rows.append(row)

            # ── Pass 2: neutralize + relabel if winner flipped ─────────
            if new_winner != current_winner:
                if not _first_train_logged:
                    print(f"  [info] First fit at pred_idx={pred_idx} "
                          f"using {len(train_data)} rows ({len(train_dates)} dates)")
                print(f"    [HO-MoE] fold {pred_date.date()}  winner → "
                      f"{new_winner}  "
                      f"(raw_cmi={diag['raw_cmi'].to_dict()})")
                current_winner = new_winner

                # TS neutralization on the WHOLE panel against the winner —
                # the rolling-OLS implementation already respects min_periods
                # so early dates are gracefully NaN'd.
                #
                # When ts_neutralize is False (caller passed --no-adx_neutral)
                # the Global model trains on the same raw normalized features
                # as the experts — the global vs expert distinction then comes
                # purely from "market-wide alpha" vs "regime-conditional alpha"
                # rather than "regime-stripped" vs "raw" features.
                if ts_neutralize:
                    panel_neutral = neutralize_features_on_adx(
                        panel, feature_cols,
                        adx_col=current_winner,
                        window=ts_neutral_window,
                    )
                else:
                    panel_neutral = panel.copy()

                # IS-fit bin edges → apply to whole panel for label parity.
                edges = fit_macro_regime_bin_edges(
                    train_data, current_winner, n_quantiles=n_regimes
                )
                if edges is None:
                    panel[LABEL_COL] = float(n_regimes // 2)
                else:
                    labelled = apply_macro_regime_bin_edges(
                        panel, current_winner, edges,
                        label_col=LABEL_COL,
                    )
                    panel[LABEL_COL] = labelled[LABEL_COL].values
                panel_neutral[LABEL_COL] = panel[LABEL_COL].values

                regime_selector = RegimeSelector(
                    panel, LABEL_COL, hysteresis=moe_hysteresis,
                )

            # Pull post-neutralization training rows in the same row order
            # as `train_data` (rows already filtered to dropna(y) + filled).
            train_data_neut = (
                train_data[["ts", "symbol"]]
                .reset_index(drop=True)
                .merge(panel_neutral, on=["ts", "symbol"], how="left")
            )
            train_data_neut = _fill_features(train_data_neut)

            X_train_neut = train_data_neut[feature_cols].values
            X_train_raw = train_data[feature_cols].values
            y_train = train_data["y"].values

            # ── Stage 1: Global EBM on neutralized features ────────────
            global_models = _train_ebm_ensemble(
                X_train_neut, y_train, ebm_kwargs,
                date_arr=train_data["ts"].values,
            )
            y_global_is = np.mean(
                [m.predict(X_train_neut) for m in global_models], axis=0)
            y_residuals = y_train - y_global_is

            # ── Stage 2: Expert EBMs on raw features per regime ────────
            # `train_data` was sliced from `panel` BEFORE the winner-change
            # branch attached LABEL_COL, so the label is refreshed here via
            # a date→label map. Safe to run every fold — the lookup is a
            # cheap dict map and stays in sync with the latest separator.
            date_to_label = (
                panel.drop_duplicates("ts").set_index("ts")[LABEL_COL]
            )
            train_data = train_data.copy()
            train_data[LABEL_COL] = (
                train_data["ts"].map(date_to_label).astype("float32")
            )

            # Use the actual (non-lagged) market regime label for IS.
            train_regime_raw = train_data[LABEL_COL].apply(
                lambda v: str(int(v)) if pd.notna(v) else "nan"
            ).values

            regime_tasks = []
            for r_str in np.unique(train_regime_raw):
                if r_str == "nan":
                    continue
                mask = train_regime_raw == r_str
                if mask.sum() < 30:
                    continue
                regime_tasks.append((r_str, mask))

            def _fit_one_expert(mask):
                return _train_ebm_ensemble(
                    X_train_raw[mask], y_residuals[mask], expert_ebm_kwargs,
                    date_arr=train_data["ts"].values[mask],
                    force_no_bagging=True,
                )

            expert_results = Parallel(
                n_jobs=min(max(1, len(regime_tasks)), n_expert_jobs),
                backend="loky",
            )(
                delayed(_fit_one_expert)(mask)
                for _, mask in regime_tasks
            )

            expert_models_dict: dict = {}
            for (r_str, _mask), experts in zip(regime_tasks, expert_results):
                expert_models_dict[r_str] = experts
                imp_e = _ensemble_importances(experts)
                imp_e.name = str(pred_date.date())
                all_expert_importances.setdefault(r_str, []).append(imp_e)

            moe_ens = ResidualMoE(global_models, expert_models_dict)
            models = moe_ens

            # IS total predictions (uses actual non-lagged regime).
            y_expert_is = np.zeros(len(X_train_raw))
            for r_str, exps in expert_models_dict.items():
                mask = train_regime_raw == r_str
                if mask.sum() > 0:
                    y_expert_is[mask] = np.mean(
                        [m.predict(X_train_raw[mask]) for m in exps], axis=0)
            y_pred_is = y_global_is + moe_boost_lambda * y_expert_is

            imp_g = _ensemble_importances(global_models)
            imp_g.name = str(pred_date.date())
            all_importances.append(imp_g)

            if save_models:
                fold_path = os.path.join(
                    model_dir,
                    f"ebm_homoe_{pred_date.strftime('%Y%m%d')}.pkl")
                with open(fold_path, "wb") as f:
                    pickle.dump(moe_ens, f)

            # IS fold metrics (same shape as walk_forward).
            is_ic, _ = stats.spearmanr(y_pred_is, y_train)
            ss_res = np.sum((y_train - y_pred_is) ** 2)
            ss_tot = np.sum((y_train - np.mean(y_train)) ** 2)
            is_r2 = 1.0 - ss_res / (ss_tot + 1e-12)
            is_perf = _fold_portfolio_perf(
                train_data, y_pred_is, quantile,
                beta_col=beta_col if beta_neutral else None)
            is_daily_rets.update(is_perf.get("daily_rets", {}))
            all_fold_stats.append({
                "fold_date":       pred_date,
                "n_rows":          len(train_data),
                "n_dates":         len(train_dates),
                "is_ic":           float(is_ic),
                "is_r2":           float(is_r2),
                "is_sharpe":       is_perf["sharpe"],
                "is_total_return": is_perf["total_return"],
                "winner":          current_winner,
            })
            _first_train_logged = True

        # ── Predict ──────────────────────────────────────────────────────
        if models is None or panel_neutral is None:
            continue

        pred_neut = panel_neutral[panel_neutral["ts"] == pred_date].copy()
        pred_neut = _fill_features(pred_neut)
        if pred_neut.empty:
            continue
        pred_raw = panel[panel["ts"] == pred_date].copy()
        pred_raw = _fill_features(pred_raw)

        # Align raw features to neutralized row order (ts, symbol).
        aligned_raw = (
            pred_neut[["ts", "symbol"]]
            .reset_index(drop=True)
            .merge(pred_raw, on=["ts", "symbol"], how="left")
        )
        X_pred_neut = pred_neut[feature_cols].values
        X_pred_raw = aligned_raw[feature_cols].fillna(0.0).values

        active_regime = (
            regime_selector.get_regime(pred_date)
            if regime_selector else None
        )
        scores = models.predict_total(
            X_pred_neut, active_regime, moe_boost_lambda,
            X_expert=X_pred_raw,
        )
        all_preds[pred_date] = pd.Series(
            scores, index=pred_neut["symbol"].values)

        # Per-OOS-date timeline of (separator winner, lagged+hysteresis regime,
        # number of symbols scored, expert availability for the active regime).
        regime_timeline_rows.append({
            "ts": pred_date,
            "winner_separator": current_winner,
            "active_regime": active_regime,
            "n_symbols": int(len(pred_neut)),
            "expert_available": bool(
                active_regime is not None
                and active_regime in models.expert_dict
                and len(models.expert_dict[active_regime]) > 0
            ),
        })

    if not all_preds:
        raise RuntimeError(
            "walk_forward_ho_moe: no predictions generated. Increase the "
            "date range or reduce min_train_periods.")

    predictions_wide = pd.DataFrame(all_preds).T
    predictions_wide.index.name = "ts"
    predictions_wide = predictions_wide.sort_index()

    importance_df = (
        pd.DataFrame(all_importances) if all_importances else pd.DataFrame()
    )
    fold_stats_df = pd.DataFrame(all_fold_stats)
    if not fold_stats_df.empty:
        fold_stats_df = fold_stats_df.set_index("fold_date")
    is_rets = pd.Series(is_daily_rets, name="is_ret").sort_index()

    expert_importance_dfs: dict | None = None
    if all_expert_importances:
        expert_importance_dfs = {
            r: pd.DataFrame(imp_list)
            for r, imp_list in all_expert_importances.items()
        }

    cmi_log_df = pd.DataFrame(cmi_log_rows)
    if not cmi_log_df.empty:
        cmi_log_df = cmi_log_df.set_index("fold_date")

    regime_timeline_df = pd.DataFrame(regime_timeline_rows)
    if not regime_timeline_df.empty:
        regime_timeline_df = regime_timeline_df.set_index("ts").sort_index()

    return (predictions_wide, importance_df, fold_stats_df, is_rets,
            expert_importance_dfs, cmi_log_df, regime_timeline_df)


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
    n_rows = 5 if has_expert else 4
    fig_height = 22 if has_expert else 17

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

    # 5. Coverage — active symbols per day (long + short legs separately)
    ax5 = fig.add_subplot(gs[3, :])
    if weights is not None and not weights.empty:
        long_cnt = (weights > 0).sum(axis=1)
        short_cnt = (weights < 0).sum(axis=1)
        total_cnt = long_cnt + short_cnt
        ax5.plot(total_cnt.index, total_cnt.values,
                 color="#37474F", lw=1.2, label="Total active")
        ax5.plot(long_cnt.index, long_cnt.values,
                 color="#2E7D32", lw=1.0, alpha=0.85, label="Long leg")
        ax5.plot(short_cnt.index, short_cnt.values,
                 color="#C62828", lw=1.0, alpha=0.85, label="Short leg")
        ax5.fill_between(total_cnt.index, 0, total_cnt.values,
                         color="#37474F", alpha=0.07)
        ax5.set_ylabel("Active symbols")
        ax5.set_ylim(bottom=0)
        ax5.legend(fontsize=8, loc="lower right")
        cov_lines = [
            f"Total : mean={total_cnt.mean():.1f}  min={total_cnt.min()}  max={total_cnt.max()}",
            f"Long  : mean={long_cnt.mean():.1f}",
            f"Short : mean={short_cnt.mean():.1f}",
        ]
        ax5.set_title(
            "OOS Coverage  |  " + cov_lines[0],
            fontsize=9, pad=4)
        ax5.text(0.02, 0.97, "\n".join(cov_lines),
                 transform=ax5.transAxes, fontsize=7.5, family="monospace",
                 verticalalignment="top",
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                           alpha=0.85, edgecolor="#CCCCCC"))
        ax5.tick_params(axis="x", labelrotation=30, labelsize=7)
        ax5.xaxis.set_major_locator(plt.MaxNLocator(6))
    else:
        ax5.text(0.5, 0.5, "No coverage data", ha="center", va="center")

    # 6. Expert importances — most-represented regime (MoE only)
    if has_expert:
        ax6 = fig.add_subplot(gs[4, :])
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
# Per-epoch pipeline helper
# ---------------------------------------------------------------------------

def _run_epoch_pipeline(panel_epoch, feature_cols, args, ebm_kwargs,
                        expert_ebm_kwargs, model_dir):
    """
    Run the full EBM pipeline (build_target → normalize → walk_forward →
    neutralize → predictions_to_weights) on a single epoch's panel slice.

    `panel_epoch` must already be filtered to the epoch's active symbols but
    should contain FULL temporal history so that TS-based feature normalization
    and rolling lookback windows warm up correctly.

    Returns (weights, predictions_wide, importance_df, fold_stats_df,
             expert_importance_dfs, cmi_log_df). `cmi_log_df` is None unless
    --ho_moe is set, in which case it holds the per-fold tournament log.
    """
    ep = panel_epoch.copy()

    ep = build_target(
        ep, args.target_col, args.target_horizon, args.target_type,
        beta_neutral=args.target_beta_neutral, beta_col=args.beta_col,
    )

    # ── HO-MoE branch ─────────────────────────────────────────────────────
    if args.ho_moe:
        # Macro candidates are computed BEFORE normalize_features so they
        # reflect raw cross-sectional market state (CS-mean volatility, CS-
        # sum dollar volume, CS-std returns). normalize_features only acts
        # on feature_cols, leaving the macro columns intact for CMI scoring
        # and TS neutralization.
        ep = compute_macro_candidates(ep)
        ep = normalize_features(
            ep, feature_cols, args.feature_norm, args.ts_z_window)

        (predictions_wide, importance_df,
         fold_stats_df, _is_rets,
         expert_importance_dfs, cmi_log_df,
         regime_timeline_df) = walk_forward_ho_moe(
            panel=ep,
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
            expert_ebm_kwargs=expert_ebm_kwargs,
            save_models=args.save_models,
            model_dir=model_dir,
            use_block_bagging=args.use_block_bagging,
            n_outer_bags=args.n_outer_bags,
            block_size=args.block_size,
            embargo_pct=args.embargo_pct,
            moe_boost_lambda=args.moe_boost_lambda,
            moe_hysteresis=args.moe_hysteresis,
            cmi_candidates=MACRO_CANDIDATES,
            cmi_ema_span=args.cmi_ema_span,
            cmi_q_target=args.cmi_q_target,
            cmi_q_features=args.cmi_q_features,
            cmi_q_candidates=args.cmi_q_candidates,
            ts_neutral_window=args.ts_neutral_window,
            ts_neutralize=args.adx_neutral,
            fix_separator=args.ho_moe_fix_separator,
            n_regimes=args.n_regimes,
        )

        if args.beta_neutral:
            predictions_wide = neutralize_scores(
                predictions_wide, ep, beta_col=args.beta_col)

        weights = predictions_to_weights(
            predictions_wide,
            quantile=args.quantile,
            max_weight=args.max_weight,
            weight_mode=args.weight_mode,
        )
        return (weights, predictions_wide, importance_df, fold_stats_df,
                expert_importance_dfs, cmi_log_df, regime_timeline_df)

    # Build RegimeSelector from the raw (pre-normalization) panel so that
    # regime column values are still the original integer labels (0/1/2).
    # normalize_features turns them into rolling z-scores, after which
    # str(int(z_score)) produces wrong label strings for expert routing.
    raw_regime_selector = None
    if args.use_moe and args.regime_col in ep.columns:
        raw_regime_selector = RegimeSelector(
            ep, args.regime_col, hysteresis=args.moe_hysteresis)

    ep = normalize_features(
        ep, feature_cols, args.feature_norm, args.ts_z_window)

    ep_adx = None
    if args.adx_neutral:
        ep_adx = neutralize_features_on_adx(
            ep, feature_cols,
            adx_col=args.adx_col,
            window=args.adx_neutral_window,
        )

    # Global model uses ADX-neutralized features (if --adx_neutral);
    # experts use raw normalized features to capture regime-specific alpha.
    main_panel = ep_adx if ep_adx is not None else ep
    raw_panel_for_experts = ep if ep_adx is not None else None

    (predictions_wide, importance_df,
     fold_stats_df, _is_rets,
     expert_importance_dfs) = walk_forward(
        panel=main_panel,
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
        bag_symbol_frac=args.bag_symbol_frac,
        bag_sym_excluded_as_val=args.bag_sym_excluded_as_val,
        use_moe=args.use_moe,
        regime_col=args.regime_col,
        moe_boost_lambda=args.moe_boost_lambda,
        moe_hysteresis=args.moe_hysteresis,
        expert_ebm_kwargs=expert_ebm_kwargs,
        regime_selector=raw_regime_selector,
        expert_panel=raw_panel_for_experts,
        pred_start_date=getattr(args, "pred_start_date", None),
    )

    if args.beta_neutral:
        predictions_wide = neutralize_scores(
            predictions_wide, main_panel, beta_col=args.beta_col)

    weights = predictions_to_weights(
        predictions_wide,
        quantile=args.quantile,
        max_weight=args.max_weight,
        weight_mode=args.weight_mode,
    )

    return (weights, predictions_wide, importance_df, fold_stats_df,
            expert_importance_dfs, None, None)


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

    # --- ADX feature neutralization ------------------------------------------
    ap.add_argument("--adx_neutral", action="store_true", default=False,
                    help="Neutralize each feature against the macro series via "
                         "rolling OLS before training. Removes the systematic "
                         "component of each feature that co-varies with the "
                         "macro, leaving the idiosyncratic residual for the "
                         "model to learn. Legacy path: neutralizes against "
                         "--adx_col (default market_adx). HO-MoE path: "
                         "neutralizes the Global model's features against the "
                         "CMI-tournament winner (omit the flag to train the "
                         "Global model on raw features — the Global vs Expert "
                         "distinction then becomes market-wide vs regime-"
                         "conditional alpha rather than residualized vs raw).")
    ap.add_argument("--adx_col",    default="market_adx",
                    help="Column name of the market-wide ADX series in the panel "
                         "(default: market_adx)")
    ap.add_argument("--adx_neutral_window", type=int, default=120,
                    help="Rolling window (trading days) for the ADX-beta OLS "
                         "(default: 252)")

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
    ap.add_argument("--n_jobs",              type=int,   default=-2,
                    help="Number of parallel jobs for EBM outer-bag fitting. "
                         "-1 = all cores, -2 = all cores minus 1 (default), "
                         "1 = sequential (old behaviour)")
    ap.add_argument("--random_state",         type=int,   default=42)

    # --- Block bagging -------------------------------------------------------
    ap.add_argument("--use_block_bagging", action="store_true",
                    help="Replace EBM outer_bags with manual block-bootstrap ensemble")
    ap.add_argument("--n_outer_bags",  type=int,   default=8,
                    help="Number of block-bootstrap bags (only when --use_block_bagging)")
    ap.add_argument("--block_size",    type=int,   default=42,
                    help="Block length in rows for block bootstrap (≈1 trading month)")
    ap.add_argument("--embargo_pct",   type=float, default=0.01,
                    help="Fraction of train window to embargo beyond target_horizon")
    ap.add_argument("--bag_symbol_frac", type=float, default=1.0,
                    help="Per-bag symbol subsample fraction in (0, 1]. "
                         "1.0 = use all symbols (TIME-only diversity). "
                         "0.8 = each bag keeps a random 80%% of symbols, "
                         "decorrelating bags along the cross-sectional axis. "
                         "Sampling is per-bag, leak-free (does not touch "
                         "validation rows or any post-pred-date data).")
    ap.add_argument("--bag_sym_excluded_as_val", action="store_true",
                    help="When --bag_symbol_frac < 1.0, route the symbol-"
                         "excluded TRAINING rows into this bag's validation "
                         "set (interpret -1) instead of skipping them "
                         "(interpret 0). Effect: early-stopping per bag is "
                         "informed by cross-sectional generalization in "
                         "addition to the time-based holdout. Useful when "
                         "you want bags to stop at different boosting "
                         "rounds (more ensemble diversity) and you believe "
                         "the signal should generalize to unseen symbols "
                         "within the training window. CAUTION: with low "
                         "bag_symbol_frac (e.g. 0.5) the symbol-val portion "
                         "can dwarf the time-val portion and shift the "
                         "objective away from pure temporal generalization. "
                         "Default off — opt in to A/B against the legacy "
                         "behaviour.")

    # --- Mixture of Experts --------------------------------------------------
    ap.add_argument("--use_moe", action="store_true",
                    help="Enable Residual-Based Mixture of Experts (MoE) ensemble")
    ap.add_argument("--regime_col", default="volatility_regime_enc",
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

    # --- Hierarchical Orthogonalized MoE (HO-MoE) ----------------------------
    ap.add_argument("--ho_moe", action="store_true",
                    help="Enable HO-MoE: every retrain runs a CMI tournament "
                         "over (market_volatility, market_liquidity, "
                         "market_dispersion) on the trailing IS panel. The "
                         "EMA-smoothed winner becomes the active "
                         "Regime_Separator; features are TS-neutralized "
                         "against it for the Global model and a market-wide "
                         "MoE routes per-regime experts. Implies --use_moe "
                         "semantics and overrides --regime_col / --adx_neutral.")
    ap.add_argument("--cmi_ema_span", type=int, default=3,
                    help="EMA span (in folds) for the per-candidate CMI score "
                         "tournament smoothing.")
    ap.add_argument("--cmi_q_target",     type=int, default=3,
                    help="Quantile bins for the target Y in CMI binning.")
    ap.add_argument("--cmi_q_features",   type=int, default=3,
                    help="Quantile bins for each feature X_i in CMI binning.")
    ap.add_argument("--cmi_q_candidates", type=int, default=3,
                    help="Quantile bins for each macro candidate R in CMI binning.")
    ap.add_argument("--ts_neutral_window", type=int, default=30,
                    help="Rolling window (trading days) for the per-symbol "
                         "OLS used to TS-neutralize features against the "
                         "winning macro separator (Pass 2).")
    ap.add_argument("--n_regimes", type=int, default=3,
                    help="Number of regime buckets for HO-MoE expert routing "
                         "(time-series quantile of the winning macro series).")
    ap.add_argument("--ho_moe_fix_separator", default=None,
                    help="Skip the CMI tournament and use this column as the "
                         "Regime_Separator for every retrain. Typical values: "
                         "market_volatility, market_liquidity, market_dispersion. "
                         "Use this to A/B test individual macro candidates as "
                         "the regime axis against the legacy "
                         "volatility_regime_enc baseline. Other panel columns "
                         "(e.g. market_adx) are accepted but must exist after "
                         "compute_macro_candidates + normalize_features run.")

    # --- Output --------------------------------------------------------------
    ap.add_argument("--save_models", action="store_true",
                    help="Pickle each fold's EBM model (can be large)")
    ap.add_argument("--no_rolling_universe", action="store_true",
                    help="Skip rolling universe epoch mask on weights. "
                         "Use this to baseline-test EBM signal correctness.")
    ap.add_argument("--pred_start_date", default=None,
                    help="First OOS prediction date (YYYY-MM-DD). All panel "
                         "history before this date is kept and used for "
                         "training, EWM warmup, and rolling features, but "
                         "the walk-forward only emits predictions from this "
                         "date onwards. Use when the panel includes warmup "
                         "history (e.g. 2023) but tradeable performance "
                         "should begin at a specific calendar date "
                         "(e.g. 2024-01-01). The retrain schedule is "
                         "anchored to this date.")

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

    # ── Optional: anchor first prediction date ────────────────────────────────
    # Panel is NOT truncated. Pre-pred_start_date rows stay in the panel as
    # warmup history (training data, EWM state, rolling features). The
    # walk_forward simply skips emitting OOS predictions until pred_start_date,
    # and anchors its retrain schedule to that date.
    if args.pred_start_date:
        ps = pd.to_datetime(args.pred_start_date)
        warmup_dates = (panel["ts"] < ps).sum()
        print(f"  [pred_start_date={args.pred_start_date}] "
              f"first OOS prediction will be on or after {ps.date()}  |  "
              f"warmup rows kept for training: {warmup_dates:,}")

    # ── Resolve feature columns ───────────────────────────────────────────────
    if args.features is not None and args.features not in (["all"], ["filtered"]):
        feature_cols = args.features
    elif args.features == ["all"]:
        numeric = panel.select_dtypes(include=[np.number]).columns.tolist()
        exclude = _META_COLS | _SIGNAL_COLS | {"y"} | _FILTERED_COLS
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
        n_jobs=args.n_jobs,
        random_state=args.random_state,
        feature_names=feature_cols,
        # validation_size=0
        # smoothing_rounds=120
    )

    # Expert EBMs inherit global kwargs but override interactions and optionally lr.
    expert_ebm_kwargs: dict | None = None
    if args.use_moe or args.ho_moe:
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

    # ── Per-epoch training or single global pass ──────────────────────────────
    # With rolling universe: one EBM trained per epoch using only that epoch's
    # active symbols, so CS normalization (build_target, normalize_features) and
    # walk-forward ranking always operate over a consistent symbol set.
    # Full temporal history is kept per symbol for TS lookback warmup.
    # Without rolling universe (or --no_rolling_universe): single global pass.
    # Phase-2 refactor: use the shared preamble helper. Note this caller
    # derives the window from the panel itself rather than CLI args; same
    # semantics either way since resolve_epochs only consults start/end.
    ru_epochs = resolve_epochs(
        panel["ts"].min().strftime("%Y-%m-%d"),
        panel["ts"].max().strftime("%Y-%m-%d"),
        no_rolling_universe=args.no_rolling_universe) or []

    # HO-MoE only — populated by the per-fold CMI tournament and the per-
    # OOS-date regime gater. Default None so the CSV save blocks below work
    # in non-HO-MoE runs.
    cmi_log_df = None
    regime_timeline_df = None

    if ru_epochs:
        print(f"\n  Rolling universe: {len(ru_epochs)} epochs — "
              f"training a separate EBM per epoch.")

        def _run_one_epoch(ep_dict):
            """Train one epoch and return results with epoch metadata."""
            ep_start = pd.Timestamp(ep_dict["epoch_start"])
            ep_end = pd.Timestamp(ep_dict["epoch_end"])
            ep_syms = ep_dict["symbols"]

            panel_ep = panel[panel["symbol"].isin(ep_syms)].copy()
            if panel_ep.empty:
                return None

            (w_ep, pred_ep, imp_ep, fold_ep, exp_ep,
             cmi_ep, reg_ep) = _run_epoch_pipeline(
                panel_ep, feature_cols, args, ebm_kwargs,
                expert_ebm_kwargs, model_dir,
            )

            # Slice every per-epoch artifact to the epoch's active date
            # range. Without this, an epoch whose universe was only valid
            # in 2024 H1 would still contribute its 32 retrain rows (and
            # the corresponding global / expert importance rows) across
            # the whole panel calendar — polluting cross-epoch usage
            # analysis with experts that never actually traded in those
            # quarters.
            def _slice_dt(df, idx_is_str=False):
                if df is None or df.empty:
                    return df
                ts = pd.to_datetime(df.index) if idx_is_str else df.index
                return df.loc[(ts >= ep_start) & (ts <= ep_end)]

            w_ep = _slice_dt(w_ep)
            pred_ep = _slice_dt(pred_ep)
            reg_ep = _slice_dt(reg_ep)
            cmi_ep = _slice_dt(cmi_ep)
            # fold_stats_df / global importance frames use str(date) index;
            # convert before comparing.
            fold_ep = _slice_dt(fold_ep)
            imp_ep = _slice_dt(imp_ep, idx_is_str=True)
            # Importance frames keep all columns under .loc slicing — terms
            # that only appeared in dropped folds become all-NaN ghost
            # columns. Drop them so downstream aggregation reports honest
            # presence counts.
            if imp_ep is not None and not imp_ep.empty:
                imp_ep = imp_ep.dropna(axis=1, how="all")
            if exp_ep:
                exp_ep = {
                    r: _slice_dt(df, idx_is_str=True)
                    for r, df in exp_ep.items()
                }
                exp_ep = {
                    r: df.dropna(axis=1, how="all")
                    for r, df in exp_ep.items()
                    if df is not None and not df.empty
                }
                # Drop regimes that became empty after slicing — they had
                # no folds inside the epoch's active range.
                exp_ep = {r: df for r, df in exp_ep.items() if not df.empty}
            return w_ep, pred_ep, imp_ep, fold_ep, exp_ep, cmi_ep, reg_ep

        # Train all epochs. In HO-MoE the per-fold CMI EMA is stateful
        # within an epoch but each epoch's state is fresh — epochs can
        # still be parallelised. However the user specifically asked for
        # sequential epoch processing when --ho_moe to keep the pipeline
        # deterministic and trivially debuggable; we honour that.
        if args.ho_moe:
            n_epoch_jobs = 1
            print(f"  Sequential epoch training (HO-MoE) for "
                  f"{len(ru_epochs)} epochs")
        else:
            n_epoch_jobs = max(
                1, min(len(ru_epochs), max(1, os.cpu_count() or 1)))
            print(f"  Parallel epoch training: {n_epoch_jobs} jobs "
                  f"for {len(ru_epochs)} epochs")

        epoch_results = Parallel(n_jobs=n_epoch_jobs, backend="loky")(
            delayed(_run_one_epoch)(ep) for ep in ru_epochs
        ) if ru_epochs else []

        weight_slices, pred_slices = [], []
        imp_slices, fold_slices = [], []
        expert_imp_slices: dict = {}
        cmi_slices: list = []
        regime_slices: list = []

        for ep, result in zip(ru_epochs, epoch_results):
            if result is None:
                print(f"    [skip] Epoch {ep['epoch_start']} → "
                      f"{ep['epoch_end']}: no panel rows.")
                continue
            (w_ep, pred_ep, imp_ep, fold_ep, exp_ep,
             cmi_ep, reg_ep) = result
            print(f"    Epoch {ep['epoch_start']} → {ep['epoch_end']}: "
                  f"predictions {pred_ep.shape}")
            weight_slices.append(w_ep)
            pred_slices.append(pred_ep)
            if not imp_ep.empty:
                imp_slices.append(imp_ep)
            if not fold_ep.empty:
                fold_slices.append(fold_ep)
            for regime_str, df in (exp_ep or {}).items():
                expert_imp_slices.setdefault(regime_str, []).append(df)
            if cmi_ep is not None and not cmi_ep.empty:
                cmi_slices.append(cmi_ep)
            if reg_ep is not None and not reg_ep.empty:
                regime_slices.append(reg_ep)

        all_dates = pd.DatetimeIndex(sorted(panel["ts"].unique()))
        weights = (
            pd.concat(weight_slices).reindex(all_dates).fillna(0.0)
            if weight_slices else pd.DataFrame(0.0, index=all_dates, columns=[])
        )
        # predictions_wide: NaN on inactive dates so compute_ic's pred.dropna()
        # correctly skips them. Using fillna(0.0) would create all-zero rows that
        # look like constant predictions and cause ConstantInputWarning in spearmanr.
        predictions_wide = (
            pd.concat(pred_slices).reindex(all_dates)
            if pred_slices else pd.DataFrame(index=all_dates)
        )
        importance_df = pd.concat(imp_slices) if imp_slices else pd.DataFrame()
        fold_stats_df = pd.concat(
            fold_slices) if fold_slices else pd.DataFrame()
        expert_importance_dfs = {r: pd.concat(
            dfs) for r, dfs in expert_imp_slices.items()}
        cmi_log_df = (
            pd.concat(cmi_slices).sort_index() if cmi_slices else None
        )
        regime_timeline_df = (
            pd.concat(regime_slices).sort_index() if regime_slices else None
        )

        # OOS metrics need y_raw on the full panel. build_target adds it as a
        # plain forward return shift — no CS normalization, safe to run globally.
        panel = build_target(
            panel, args.target_col, args.target_horizon, args.target_type,
            beta_neutral=False, beta_col=args.beta_col,
        )

    else:
        # Single global pass (no rolling universe or --no_rolling_universe)
        print("\n  [FIXED UNIVERSE] Single global EBM pass.")
        panel = build_target(
            panel, args.target_col, args.target_horizon, args.target_type,
            beta_neutral=args.target_beta_neutral, beta_col=args.beta_col,
        )
        cmi_log_df = None

        if args.ho_moe:
            # HO-MoE branch — mirrors _run_epoch_pipeline so single-universe
            # backtests can use the same pipeline.
            panel = compute_macro_candidates(panel)
            panel = normalize_features(
                panel, feature_cols, args.feature_norm, args.ts_z_window)

            (predictions_wide, importance_df,
             fold_stats_df, _is_rets,
             expert_importance_dfs, cmi_log_df,
             regime_timeline_df) = walk_forward_ho_moe(
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
                expert_ebm_kwargs=expert_ebm_kwargs,
                save_models=args.save_models,
                model_dir=model_dir,
                use_block_bagging=args.use_block_bagging,
                n_outer_bags=args.n_outer_bags,
                block_size=args.block_size,
                embargo_pct=args.embargo_pct,
                moe_boost_lambda=args.moe_boost_lambda,
                moe_hysteresis=args.moe_hysteresis,
                cmi_candidates=MACRO_CANDIDATES,
                cmi_ema_span=args.cmi_ema_span,
                cmi_q_target=args.cmi_q_target,
                cmi_q_features=args.cmi_q_features,
                cmi_q_candidates=args.cmi_q_candidates,
                ts_neutral_window=args.ts_neutral_window,
                ts_neutralize=args.adx_neutral,
                fix_separator=args.ho_moe_fix_separator,
                n_regimes=args.n_regimes,
            )
            print(f"  Prediction matrix: {predictions_wide.shape}")

            if args.beta_neutral:
                print(f"\nBeta-neutralizing scores (col={args.beta_col})...")
                predictions_wide = neutralize_scores(
                    predictions_wide, panel, beta_col=args.beta_col)
                print("  Done.")

            weights = predictions_to_weights(
                predictions_wide,
                quantile=args.quantile,
                max_weight=args.max_weight,
                weight_mode=args.weight_mode,
            )
        else:
            # ── Legacy single-pass (static MoE / no MoE) ──────────────────
            # Build RegimeSelector from raw panel before normalization corrupts labels.
            raw_regime_selector = None
            if args.use_moe and args.regime_col in panel.columns:
                raw_regime_selector = RegimeSelector(
                    panel, args.regime_col, hysteresis=args.moe_hysteresis)

            panel = normalize_features(
                panel, feature_cols, args.feature_norm, args.ts_z_window)

            panel_adx = None
            if args.adx_neutral:
                panel_adx = neutralize_features_on_adx(
                    panel, feature_cols,
                    adx_col=args.adx_col,
                    window=args.adx_neutral_window,
                )

            # Global model uses ADX-neutralized features (if --adx_neutral);
            # experts use raw normalized features to capture regime-specific alpha.
            main_panel = panel_adx if panel_adx is not None else panel
            raw_panel_for_experts = panel if panel_adx is not None else None

            (predictions_wide, importance_df,
             fold_stats_df, _is_rets,
             expert_importance_dfs) = walk_forward(
                panel=main_panel,
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
                bag_symbol_frac=args.bag_symbol_frac,
                bag_sym_excluded_as_val=args.bag_sym_excluded_as_val,
                use_moe=args.use_moe,
                regime_col=args.regime_col,
                moe_boost_lambda=args.moe_boost_lambda,
                moe_hysteresis=args.moe_hysteresis,
                expert_ebm_kwargs=expert_ebm_kwargs,
                regime_selector=raw_regime_selector,
                expert_panel=raw_panel_for_experts,
                pred_start_date=getattr(args, "pred_start_date", None),
            )
            print(f"  Prediction matrix: {predictions_wide.shape}")

            if args.beta_neutral:
                print(f"\nBeta-neutralizing scores (col={args.beta_col})...")
                predictions_wide = neutralize_scores(
                    predictions_wide, main_panel, beta_col=args.beta_col)
                print("  Done.")

            weights = predictions_to_weights(
                predictions_wide,
                quantile=args.quantile,
                max_weight=args.max_weight,
                weight_mode=args.weight_mode,
            )

    # ── Trim OOS outputs to pred_start_date ──────────────────────────────────
    # All warmup-period rows in weights / predictions_wide are dropped here so
    # downstream metrics, parquet files, and plots reflect only the tradeable
    # OOS window. Pre-pred_start_date rows were kept upstream solely for
    # training warmup and are no longer needed.
    if args.pred_start_date:
        ps = pd.Timestamp(args.pred_start_date)
        before_w = len(weights)
        weights = weights.loc[weights.index >= ps]
        predictions_wide = predictions_wide.loc[predictions_wide.index >= ps]
        print(f"  Trimmed OOS outputs to ts >= {ps.date()}: "
              f"{before_w} → {len(weights)} dates")

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

    # HO-MoE CMI tournament log
    if cmi_log_df is not None and not cmi_log_df.empty:
        cmi_path = os.path.join(out_dir, "ebm_homoe_cmi_log.csv")
        cmi_log_df.to_csv(cmi_path)
        print(f"HO-MoE CMI log saved → {cmi_path}")
        winner_counts = cmi_log_df["winner"].value_counts()
        print(f"  Winner distribution across {len(cmi_log_df)} folds:")
        for w, c in winner_counts.items():
            print(f"    {w:<25s} {c:>4d}  ({c / len(cmi_log_df):.0%})")

    # HO-MoE per-OOS-date regime timeline (separator winner + active regime).
    if regime_timeline_df is not None and not regime_timeline_df.empty:
        reg_path = os.path.join(out_dir, "ebm_homoe_regime_timeline.csv")
        regime_timeline_df.to_csv(reg_path)
        print(f"HO-MoE regime timeline saved → {reg_path}")

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
