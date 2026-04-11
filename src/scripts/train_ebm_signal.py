"""
EBM (Explainable Boosting Machine) signal generation.

Walk-forward: model retrained every `--retrain_freq` periods, predicts the
next period only, so there is zero look-ahead by construction.

Target options
--------------
  cs_rank   Cross-sectional percentile rank of forward return (default).
            Removes market-wide beta; focuses the model on *which* symbol
            outperforms. Robust to return scale changes over time.
  raw       Raw forward return. Simpler, but scale can shift across regimes.

Feature normalization options
-----------------------------
  cs        Cross-sectional z-score per date (recommended for CS targets).
  ts        Time-series z-score over a rolling window.
  rank      Cross-sectional percentile rank per date.
  none      Raw factor values.

Signal construction
-------------------
  After walk-forward prediction the raw scores are:
    1. Beta-neutralized via OLS residualization (--beta_neutral, default on).
    2. Converted to weights via quantile selection + chosen weight_mode.

General utilities (normalize_features, build_target, predictions_to_weights,
neutralize_scores, compute_ic, compute_portfolio_performance) live in:
  src/alpha/ml_utils.py

Outputs (all in ./reports/strategies/{run_id}/)
------------------------------------------------
  ebm.parquet                weight matrix (ts × symbol) — pipeline-compatible
  ebm_predictions.parquet    raw OOS prediction scores (ts × symbol)
  ebm_feature_importance.csv per-fold feature importances
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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Walk-forward EBM training.

    When use_block_bagging=True, replaces EBM's internal outer_bags with a
    manual block-bootstrap ensemble: each bag is trained on a block-resampled
    subset of the training data, and predictions are averaged across bags.
    This avoids i.i.d. resampling that leaks temporal structure.

    Returns
    -------
    predictions_wide : pd.DataFrame (ts × symbol)  raw OOS prediction scores
    importance_df    : pd.DataFrame  per-fold feature importances
    fold_stats_df    : pd.DataFrame  per-fold IS metrics
    """
    dates = sorted(panel["ts"].unique())
    n_dates = len(dates)
    label_cutoff_idx = n_dates - target_horizon

    all_preds = {}
    all_importances = []
    all_fold_stats = []
    is_daily_rets = {}   # date → return; later folds overwrite earlier for overlapping dates

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

    # models is always a list; single-model mode wraps in [model]
    models: list | None = None
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

    for pred_idx in range(min_train_periods, label_cutoff_idx):
        pred_date = dates[pred_idx]

        # ── Retrain if needed ────────────────────────────────────────────────
        if pred_idx in retrain_set:
            # Embargo: push train_end back by an additional buffer beyond target_horizon
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

            if use_block_bagging:
                # Disable EBM's internal outer_bags (set to 1, minimum valid value)
                # so WE control the ensemble. outer_bags=0 is invalid in interpret.
                # 'sequential' backend avoids joblib worker-pool teardown race
                # that causes spurious resource_tracker KeyError tracebacks.
                ebm_kwargs_bag = {**ebm_kwargs, "outer_bags": 1}
                bag_models = []
                for _ in range(n_outer_bags):
                    idx = _block_bootstrap_indices(
                        len(train_data), block_size, rng)
                    m = ExplainableBoostingRegressor(**ebm_kwargs_bag)
                    with parallel_backend("sequential"):
                        m.fit(X_train[idx], y_train[idx])
                    bag_models.append(m)
                models = bag_models
            else:
                m = ExplainableBoostingRegressor(**ebm_kwargs)
                with parallel_backend("sequential"):
                    m.fit(X_train, y_train)
                models = [m]

            # ── In-sample metrics (averaged ensemble predictions) ─────────────
            y_pred_is = _predict_ensemble(X_train)
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

            # ── Feature importance: align by term name, average across bags ───
            # Different bags may discover different interaction pairs.
            # Use skipna=True (default) so a term found in k of n bags is
            # averaged over those k bags only — not diluted by (n-k) zeros.
            imp_series = [
                pd.Series(m.term_importances(), index=list(m.term_names_))
                for m in models
            ]
            imp_avg = pd.concat(imp_series, axis=1).mean(axis=1)
            imp_avg.name = str(pred_date.date())
            all_importances.append(imp_avg)

            if save_models:
                fold_path = os.path.join(
                    model_dir, f"ebm_model_{pred_date.strftime('%Y%m%d')}.pkl")
                with open(fold_path, "wb") as f:
                    pickle.dump(models, f)

        # ── Predict ──────────────────────────────────────────────────────────
        if models is None:
            continue

        pred_data = panel[panel["ts"] == pred_date].copy()
        pred_data = _fill_features(pred_data)
        if pred_data.empty:
            continue

        scores = _predict_ensemble(pred_data[feature_cols].values)
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
    return predictions_wide, importance_df, fold_stats_df, is_rets


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
    is_rets: pd.Series = None,
):
    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    # 1. Feature importance (full width)
    ax1 = fig.add_subplot(gs[0, :])
    if not importance_df.empty:
        mean_imp = importance_df.mean().sort_values(ascending=True)
        bar_colors = ["#F44336" if v <
                      0 else "#2196F3" for v in mean_imp.values]
        ax1.barh(mean_imp.index, mean_imp.values, color=bar_colors, alpha=0.85)
        ax1.axvline(0, color="black", lw=0.5)
        ax1.set_xlabel("Mean Importance (across folds)")
        ax1.set_title(
            "EBM Feature Importances (avg across walk-forward folds)")
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

    # 4. IS vs OOS IC scatter (overfitting view)
    ax4 = fig.add_subplot(gs[2, 0])
    if not fold_stats_df.empty and not ic_series.empty:
        oos_mean_per_fold = [
            ic_series[ic_series.index >= fd].head(21).mean()
            for fd in fold_stats_df.index
        ]
        is_vals = fold_stats_df["is_ic"].values
        oos_vals = np.array(oos_mean_per_fold)
        valid = ~np.isnan(oos_vals)
        if valid.sum() >= 2:
            ax4.scatter(is_vals[valid], oos_vals[valid],
                        alpha=0.8, color="#9C27B0", s=60, zorder=3)
            lo = min(is_vals[valid].min(), oos_vals[valid].min()) - 0.02
            hi = max(is_vals[valid].max(), oos_vals[valid].max()) + 0.02
            ax4.plot([lo, hi], [lo, hi], "k--", lw=0.8, label="IS = OOS")
            ax4.axhline(0, color="gray", lw=0.5)
            ax4.axvline(0, color="gray", lw=0.5)
            overfit_ratio = float(np.nanmean(
                oos_vals[valid] / (is_vals[valid] + 1e-12)))
            ax4.set_xlabel("In-Sample IC")
            ax4.set_ylabel("OOS IC (next 21d avg)")
            ax4.set_title(
                f"IS vs OOS IC per Fold  |  OOS/IS ratio={overfit_ratio:.2f}")
            ax4.legend(fontsize=8)
        else:
            ax4.text(0.5, 0.5, "Not enough fold overlap",
                     ha="center", va="center")
    else:
        ax4.text(0.5, 0.5, "No data", ha="center", va="center")

    # 5. OOS + IS Cumulative PnL overlay
    ax5 = fig.add_subplot(gs[2, 1])
    port_rets = oos_perf.get("port_rets", pd.Series(dtype=float))
    if not port_rets.empty:
        cumulative = (1 + port_rets).cumprod()
        ax5.plot(cumulative.index, cumulative.values,
                 color="#FF5722", lw=1.5, label="OOS PnL")
        ax5.fill_between(cumulative.index, 1.0, cumulative.values,
                         where=cumulative.values >= 1.0, alpha=0.12, color="#4CAF50")
        ax5.fill_between(cumulative.index, 1.0, cumulative.values,
                         where=cumulative.values < 1.0,  alpha=0.12, color="#F44336")

        if is_rets is not None and not is_rets.empty:
            is_cum = (1 + is_rets).cumprod()
            ax5.plot(is_cum.index, is_cum.values,
                     color="#1565C0", lw=1.2, linestyle="--",
                     alpha=0.75, label="IS PnL (last fold per date)")

        ax5.axhline(1.0, color="black", lw=0.6, linestyle="--")
        ax5.set_ylabel("Cumulative Return")
        ax5.set_title("IS vs OOS Cumulative PnL", fontsize=9, pad=4)

        oos_sr = oos_perf.get("sharpe", np.nan)
        oos_ret = oos_perf.get("total_return", np.nan)
        oos_wr = oos_perf.get("win_rate", np.nan)
        mean_is_sr = (fold_stats_df["is_sharpe"].mean()
                      if not fold_stats_df.empty and "is_sharpe" in fold_stats_df.columns
                      else np.nan)
        is_total_ret = float((1 + is_rets).prod() - 1) if (
            is_rets is not None and not is_rets.empty) else np.nan
        stats_lines = [
            f"OOS  Sharpe : {oos_sr:+.2f}",
            f"OOS  Return : {oos_ret*100:+.1f}%",
            f"OOS  WinRate: {oos_wr*100:.0f}%",
            f"IS   Sharpe : {mean_is_sr:+.2f}" if not np.isnan(
                mean_is_sr) else "",
            f"IS   Return : {is_total_ret*100:+.1f}%" if not np.isnan(
                is_total_ret) else "",
        ]
        ax5.text(0.02, 0.97, "\n".join(l for l in stats_lines if l),
                 transform=ax5.transAxes, fontsize=7.5, family="monospace",
                 verticalalignment="top",
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                           alpha=0.85, edgecolor="#CCCCCC"))
        ax5.legend(fontsize=7, loc="lower right")
        ax5.tick_params(axis="x", labelrotation=30, labelsize=7)
        ax5.xaxis.set_major_locator(plt.MaxNLocator(6))
    else:
        ax5.text(0.5, 0.5, "No OOS returns data", ha="center", va="center")

    fig.suptitle("EBM Signal Report", fontsize=14, fontweight="bold", y=1.01)
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
                    help="Number of pairwise interaction terms (0 = none)")
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
        # Grab every numeric column, but drop:
        #   - meta cols (ts/symbol/regime strings — plus market_adx which is
        #     already included explicitly via regime features)
        #   - target return cols (prevent accidental label leakage when
        #     target_col is one of them)
        #   - strategy signal cols (handled by --include_signals opt-in,
        #     otherwise excluded to avoid circular training on the same
        #     weights the model is meant to replace)
        #   - the auto-generated 'y' column built by build_target
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

    # ── Walk-forward EBM training ─────────────────────────────────────────────
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
    print(f"\nStarting walk-forward EBM training...")
    print(f"  train_window={args.train_window}  retrain_freq={args.retrain_freq}  "
          f"min_train_periods={args.min_train_periods}")

    predictions_wide, importance_df, fold_stats_df, is_rets = walk_forward(
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
        # Mean over folds in which the term was actually selected (skipna),
        # plus the coverage (k/N folds) so zero-importance rows can be
        # read as "boosting gave it no weight" vs "only appeared once".
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
        # Flag interaction terms (contain ' & ') that were proposed by FAST
        # but received ~0 boosting updates — useful to decide whether to
        # lower --interactions.
        is_interaction = mean_imp.index.to_series().str.contains(" & ")
        zero_int = mean_imp[is_interaction & (mean_imp.abs() < 1e-4)]
        if len(zero_int):
            print(f"  {len(zero_int)} interaction term(s) averaged < 1e-4 "
                  f"(FAST-proposed but boosting-inert). "
                  f"Consider lowering --interactions.")

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
                is_rets=is_rets)

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
