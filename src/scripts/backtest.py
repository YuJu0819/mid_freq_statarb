"""
Unified backtest entry point.

Usage:
    python -m src.scripts.backtest --strategy momentum --start_date 2023-01-01 --end_date 2024-01-01 --run_id run1
    python -m src.scripts.backtest --strategy reversal  --start_date 2023-01-01 --end_date 2024-01-01 --run_id run1
    python -m src.scripts.backtest --strategy combo     --run_id run1 --start_date 2023-01-01 --end_date 2024-01-01 --method linear
"""
import argparse
import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm

from ..core.utils import load_config, ensure_dir
from ..data.loader import DataLoader, discover_symbols
from ..data.binance_futures_rest import fetch_futures_klines
from ..backtest.engine import run_vectorized_backtest
from ..backtest.reporting import (
    plot_equity_curve,
    plot_cross_sectional_analysis,
    generate_daily_regime_analysis,
    generate_predictive_regime_analysis,
    plot_daily_regime_pnl_ts,
    generate_regime_analysis_report,
    generate_weekday_analysis_report,
    generate_skew_analysis_report,
    analyze_factor_quantiles,
)
from ..portfolio.optimizer import PortfolioOptimizer
from .. import factors


# ---------------------------------------------------------------------------
# Momentum strategy backtest
# ---------------------------------------------------------------------------

def run_momentum(args, cfg):
    from ..strategy.ad_mom_spot_future import FinalStrategy

    loader = DataLoader(
        parquet_dir=cfg["general"]["parquet_dir"],
        tz=args.tz,
        local_oi_dir="./data/open_interest",
    )
    all_data = loader.load_momentum_universe(
        symbols=cfg["backtest"]["symbols"],
        start_date=args.start_date,
        end_date=args.end_date,
        no_cache=args.no_cache,
    )
    if not all_data:
        print("No data loaded. Exiting.")
        return

    strat = FinalStrategy(
        lookback=30, quantile=0.4, min_volume_usd=10_000_000,
        funding_lookback=180, funding_z_threshold=1.5, trend_ma_length=30,
        smooth_lookback=10, vol_lookback=30, vol_adj_factor=0.5,
        inverse_in_weak_regime=True,
    )

    res = run_vectorized_backtest(
        all_data, strat, cfg, run_id=args.run_id, file_name="momentum")

    _print_summary(res.summary)
    report_dir = ensure_dir(os.path.join(os.getcwd(), "reports"))

    weights = _load_weights_parquet(args.run_id, "momentum")
    if not weights.empty:
        _save_portfolio_analytics(weights, report_dir, label="momentum")

    if res.score_history is not None and not res.score_history.empty:
        res.score_history.to_csv(os.path.join(
            report_dir, "score_inspection.csv"), index=False)
        plot_cross_sectional_analysis(res.score_history, report_dir)

    if not res.equity_curve.empty:
        _save_equity_reports(res, report_dir)

    if not res.trades.empty:
        generate_regime_analysis_report(res.trades)
        if not res.equity_curve.empty:
            generate_weekday_analysis_report(res.equity_curve)
        generate_skew_analysis_report(res.trades)

    for factor_name in ["trend_score", "volatility", "funding_z_score",
                        "basis_momentum", "sentiment_score"]:
        analyze_factor_quantiles(res.score_history, factor_name, quantiles=3,
                                 report_dir=report_dir)


# ---------------------------------------------------------------------------
# Reversal strategy backtest
# ---------------------------------------------------------------------------

def run_reversal(args, cfg):
    from ..strategy.liquidation_reversal import LiquidationReversalStrategy

    loader = DataLoader(
        parquet_dir=cfg["general"]["parquet_dir"],
        tz=args.tz,
        local_metrics_dir="./data/metrics",
    )
    all_data = loader.load_reversal_universe(
        symbols=cfg["backtest"]["symbols"],
        start_date=args.start_date,
        end_date=args.end_date,
        no_cache=args.no_cache,
    )
    if not all_data:
        print("No valid data loaded. Exiting.")
        return

    strategy = LiquidationReversalStrategy(
        leverage_scale=1.0, oi_level_lookback=30,
        sentiment_ma_window=40, ts_lookback=80, half_life_decay=12,
    )

    print(f"\nRunning simulation on {len(all_data)} assets...")
    try:
        res = run_vectorized_backtest(all_data, strategy, cfg,
                                      run_id=args.run_id, file_name="reversal")
        print("\n" + "=" * 40)
        print("       BACKTEST RESULTS (REVERSAL)")
        print("=" * 40)
        _print_summary(res.summary)

        output_dir = ensure_dir("reports_reversal_daily")
        res.equity_curve.to_csv(os.path.join(output_dir, "equity_curve.csv"))
        plot_equity_curve(res.equity_curve,
                          os.path.join(output_dir, "equity_curve.png"))

        if res.score_history is not None and not res.score_history.empty:
            score_path = os.path.join(output_dir, "score_inspection.csv")
            res.score_history.to_csv(score_path, index=False)
            print(f"Reversal score history saved to: {score_path}")

        weights = _load_weights_parquet(args.run_id, "reversal")
        if not weights.empty:
            _save_portfolio_analytics(weights, output_dir, label="reversal")
    except Exception as e:
        import traceback
        print(f"\n[CRITICAL ERROR] {e}")
        traceback.print_exc()


# ---------------------------------------------------------------------------
# Combo optimizer
# ---------------------------------------------------------------------------

def run_combo(args, _cfg):
    base_dir = f"./reports/strategies/{args.run_id}"

    # 1. Load saved strategy weights
    files = glob.glob(os.path.join(base_dir, "*.parquet"))
    # Exclude previously saved optimized_weights files
    files = [
        f for f in files if "optimized_weights" not in os.path.basename(f)]
    if not files:
        raise FileNotFoundError(f"No weight parquet files found in {base_dir}")

    print(
        f"Found {len(files)} strategies: {[os.path.basename(f) for f in files]}")

    raw_strategies = {}
    for f in files:
        name = os.path.basename(f).replace(".parquet", "")
        df = pd.read_parquet(f)
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            df.index = pd.to_datetime(df.index)
        raw_strategies[name] = df

    master_ts = pd.Index([])
    master_cols = pd.Index([])
    for df in raw_strategies.values():
        master_ts = master_ts.union(df.index)
        master_cols = master_cols.union(df.columns)
    master_ts = master_ts.sort_values().unique()
    master_cols = master_cols.sort_values().unique()

    aligned = {name: df.reindex(index=master_ts, columns=master_cols).fillna(0.0)
               for name, df in raw_strategies.items()}

    # 2. Fetch prices for covariance
    fetch_start = (pd.to_datetime(args.start_date) -
                   pd.Timedelta(days=args.cov_lookback + 20)).strftime("%Y-%m-%d")
    loader = DataLoader(parquet_dir="./cache/parquet", tz=args.tz)
    prices_df = loader.load_combo_prices(
        list(master_cols), fetch_start, args.end_date
    ).reindex(master_ts).ffill()
    returns_df = prices_df.pct_change()

    # 3. Composite alpha (equal-weight average of all strategies)
    composite_alpha = sum(aligned.values()) / len(aligned)

    # 4. Walk-forward optimizer
    optimizer = PortfolioOptimizer(
        max_leverage=args.max_leverage,
        max_position=args.max_position,
        lambda_risk=args.risk_aversion,
    )

    final_weights_list = []
    active_counts = []

    for t in tqdm(master_ts, desc="Optimizing"):
        alpha_t = composite_alpha.loc[t]
        active = alpha_t[alpha_t != 0]

        if active.empty:
            final_weights_list.append(
                pd.Series(0.0, index=master_cols, name=t))
            continue

        if args.method == "equal_weight":
            w = alpha_t.copy()
            lev = w.abs().sum()
            if lev > args.max_leverage:
                w = w * (args.max_leverage / lev)

        elif args.method == "linear":
            w = optimizer.optimize_linear(alpha_t)

        elif args.method == "mean_variance":
            past_returns = returns_df.loc[:t].iloc[-(args.cov_lookback + 1):-1]
            valid_cols = past_returns.columns[past_returns.isnull(
            ).mean() < 0.2]
            valid_returns = past_returns[valid_cols].fillna(0.0)
            valid_assets = valid_returns.columns.intersection(active.index)

            if len(valid_assets) < 5:
                print("===== In mv mode but switch to linear =====")
                w = optimizer.optimize_linear(alpha_t)
            else:
                cov = valid_returns[valid_assets].cov()
                valid_mask = np.diag(cov) > 1e-8
                final_assets = valid_assets[valid_mask]
                if len(final_assets) < 2:
                    w = optimizer.optimize_linear(alpha_t)
                else:
                    w_opt = optimizer.optimize_mean_variance(
                        _zscore_alpha(alpha_t[final_assets]),
                        valid_returns[final_assets].cov())
                    w = w_opt.reindex(master_cols).fillna(0.0)
        else:
            w = pd.Series(0.0, index=master_cols, name=t)

        final_weights_list.append(w)
        active_counts.append((w != 0).sum())

    final_weights = pd.DataFrame(final_weights_list, index=master_ts)
    print(f"Average active assets: {np.mean(active_counts):.1f}")

    # 5. Portfolio analytics
    _save_portfolio_analytics(final_weights, base_dir, label=args.method)

    # Save composite alpha for score inspection
    alpha_long = composite_alpha.stack().reset_index()
    alpha_long.columns = ['ts', 'symbol', 'composite_alpha']
    alpha_long.to_csv(os.path.join(base_dir, "score_inspection.csv"), index=False)
    print(f"Combo score history saved to: {base_dir}/score_inspection.csv")

    # 6. Save & report
    save_path = os.path.join(
        base_dir, f"optimized_weights_{args.method}.parquet")
    final_weights.to_parquet(save_path)
    print(f"Optimized weights saved to: {save_path}")

    lagged = final_weights.shift(1).fillna(0.0)
    aligned_returns = returns_df.reindex(master_ts).fillna(0.0)
    port_rets = (lagged * aligned_returns).sum(axis=1)

    initial_cash = 10_000
    equity_curve = initial_cash * (1 + port_rets).cumprod()
    eq_df = pd.DataFrame({"equity": equity_curve})

    plot_equity_curve(eq_df, os.path.join(
        base_dir, f"equity_{args.method}.png"))
    eq_df.to_csv(os.path.join(base_dir, f"equity_{args.method}.csv"))

    total_ret = equity_curve.iloc[-1] / initial_cash - 1
    sharpe = (port_rets.mean() / port_rets.std() * 365**0.5
              if port_rets.std() > 0 else 0)
    print(f"\n==== COMBO RESULTS ({args.method}) ====")
    print(f"Final Equity: ${equity_curve.iloc[-1]:,.2f}")
    print(f"Total Return: {total_ret*100:.2f}%")
    print(f"Sharpe Ratio: {sharpe:.2f}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_portfolio_analytics(weights_df: pd.DataFrame, save_dir: str, label: str = ""):
    """
    Generates two charts:
      1. coverage.png  — which symbols are traded over time + active count
      2. weight_dist.png — max single-symbol weight over time + weight histogram
    Also prints a summary to stdout.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    prefix = f"{label}_" if label else ""
    active = (weights_df.abs() > 1e-6)
    active_per_day = active.sum(axis=1)
    ever_traded = active.columns[active.any(axis=0)]

    # ---- 1. Coverage -------------------------------------------------------
    _fig, axes = plt.subplots(2, 1, figsize=(14, 10),
                              gridspec_kw={"height_ratios": [1, 3]})

    # Top: active symbol count over time
    axes[0].fill_between(active_per_day.index, active_per_day.values,
                         alpha=0.7, color="steelblue")
    axes[0].set_title("Active Symbol Count Over Time")
    axes[0].set_ylabel("# Symbols")
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].grid(True, alpha=0.3)

    # Bottom: heatmap — rows = symbols sorted by total active days (most → top)
    traded_filtered = active[ever_traded].astype(float)
    days_per_sym = traded_filtered.sum(axis=0).sort_values(ascending=False)
    sorted_syms = days_per_sym.index
    heatmap_data = traded_filtered[sorted_syms].T.values

    axes[1].imshow(heatmap_data, aspect="auto", cmap="Blues",
                   interpolation="none", vmin=0, vmax=1)
    axes[1].set_yticks(range(len(sorted_syms)))
    axes[1].set_yticklabels(sorted_syms, fontsize=6)
    axes[1].set_title(
        f"Symbol Coverage Heatmap ({len(ever_traded)} traded symbols, sorted by activity)")
    axes[1].set_xlabel("Time")
    n_ticks = min(10, len(weights_df))
    tick_pos = [int(i * len(weights_df) / n_ticks) for i in range(n_ticks)]
    axes[1].set_xticks(tick_pos)
    axes[1].set_xticklabels(
        [weights_df.index[i].strftime("%Y-%m") for i in tick_pos],
        rotation=45, fontsize=8)

    plt.tight_layout()
    cov_path = os.path.join(save_dir, f"{prefix}coverage.png")
    plt.savefig(cov_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Coverage chart saved: {cov_path}")

    # ---- 2. Weight distribution --------------------------------------------
    _fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: max single-symbol weight over time
    max_weight = weights_df.abs().max(axis=1)
    axes[0].plot(max_weight.index, max_weight.values,
                 color="tomato", linewidth=1)
    axes[0].axhline(y=0.10, color="gray", linestyle="--",
                    alpha=0.6, label="10% cap")
    axes[0].set_title("Max Single-Symbol Weight Over Time")
    axes[0].set_ylabel("Max |Weight|")
    axes[0].legend()
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].grid(True, alpha=0.3)

    # Right: histogram of all non-zero weights
    all_weights = weights_df.values.flatten()
    nonzero = all_weights[np.abs(all_weights) > 1e-6]
    axes[1].hist(nonzero, bins=50, color="steelblue",
                 alpha=0.7, edgecolor="white")
    axes[1].set_title("Weight Distribution (Non-Zero)")
    axes[1].set_xlabel("Weight")
    axes[1].set_ylabel("Frequency")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    dist_path = os.path.join(save_dir, f"{prefix}weight_dist.png")
    plt.savefig(dist_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Weight distribution saved: {dist_path}")

    # ---- Summary -----------------------------------------------------------
    print(f"\n==== Portfolio Analytics ({label or 'portfolio'}) ====")
    print(f"  Symbols ever traded:      {len(ever_traded)}")
    print(f"  Avg active per day:       {active_per_day.mean():.1f}")
    print(f"  Max active in one day:    {int(active_per_day.max())}")
    print(f"  Max single-symbol weight: {weights_df.abs().max().max():.4f}")
    print(
        f"  Avg |weight| (non-zero):  {np.abs(nonzero).mean():.4f}" if len(nonzero) else "")


def _load_weights_parquet(run_id: str, file_name: str) -> pd.DataFrame:
    """Loads the weights parquet saved by run_vectorized_backtest."""
    path = os.path.join("./reports/strategies", run_id, f"{file_name}.parquet")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        df.index = pd.to_datetime(df.index)
    return df


def _zscore_alpha(alpha: pd.Series) -> pd.Series:
    """
    Cross-sectionally z-score alpha scores before MV optimisation.
    Without this, pre-capped weights (all ±0.10) give the optimizer no
    differentiation between assets, so it defaults to equal-weighting.
    """
    std = alpha.std()
    if std < 1e-8:
        return alpha
    return (alpha - alpha.mean()) / std


def _print_summary(summary: dict):
    print("\n==== Summary ====")
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


def _save_equity_reports(res, report_dir: str):
    start_str = res.equity_curve.index[0].strftime("%Y-%m-%d")
    end_str = res.equity_curve.index[-1].strftime("%Y-%m-%d")
    plot_equity_curve(
        res.equity_curve,
        os.path.join(report_dir, f"equity_curve_{start_str}_to_{end_str}.png"),
    )
    generate_daily_regime_analysis(res.equity_curve)
    generate_predictive_regime_analysis(res.equity_curve)
    plot_daily_regime_pnl_ts(res.equity_curve, report_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Unified backtest runner")
    ap.add_argument("--strategy", default="combo",
                    choices=["momentum", "reversal", "combo"],
                    help="Which strategy / mode to run (default: combo)")
    ap.add_argument(
        "--start_date", help="YYYY-MM-DD (required for momentum/reversal)")
    ap.add_argument("--end_date",   help="YYYY-MM-DD (required for all modes)")
    ap.add_argument("--run_id",     default="default_run",
                    help="ID for this run — used for saving/loading weight files")
    ap.add_argument("--config",     default="config.yaml")
    ap.add_argument("--no-cache", action="store_true",
                    help="Ignore all cached parquet files and re-fetch from API")
    ap.add_argument("--tz", default=None,
                    help="Output timezone for timestamps, e.g. 'UTC', 'Asia/Singapore' "
                         "(default: UTC-naive)")

    # Combo-specific
    ap.add_argument("--method",       default="linear",
                    choices=["linear", "mean_variance", "equal_weight"])
    ap.add_argument("--cov_lookback", type=int,   default=30)
    ap.add_argument("--risk_aversion", type=float, default=1.0)
    ap.add_argument("--max_leverage",  type=float, default=1.0)
    ap.add_argument("--max_position",  type=float, default=0.10)

    args = ap.parse_args()
    cfg = load_config(args.config)

    if args.strategy == "momentum":
        if not args.start_date or not args.end_date:
            ap.error("--start_date and --end_date are required for momentum")
        run_momentum(args, cfg)

    elif args.strategy == "reversal":
        if not args.start_date or not args.end_date:
            ap.error("--start_date and --end_date are required for reversal")
        run_reversal(args, cfg)

    elif args.strategy == "combo":
        if not args.end_date:
            ap.error("--end_date is required for combo")
        if not args.start_date:
            ap.error(
                "--start_date is required for combo (used to fetch covariance data)")
        run_combo(args, cfg)


if __name__ == "__main__":
    main()
