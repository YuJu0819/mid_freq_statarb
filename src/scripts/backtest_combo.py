import argparse
import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
from ..core.utils import load_config, ensure_dir
from ..data.binance_futures_rest import fetch_futures_klines
from ..backtest.reporting import plot_equity_curve
from ..portfolio.optimizer import PortfolioOptimizer


# Files saved by the backtest engine and train_ebm_signal that are NOT
# strategy weight matrices (exclude these from auto-discovery).
_EXCLUDE_PREFIXES = ("optimized_weights_",)
_EXCLUDE_EXACT    = {"ebm_predictions.parquet"}


def load_and_align_strategies(run_dir: str,
                               strategies: list[str] | None = None):
    """
    Loads strategy weight parquets from run_dir and aligns them to a master
    timeline.

    Parameters
    ----------
    run_dir    : directory containing *.parquet weight files
    strategies : explicit list of base names (e.g. ["momentum","reversal","ebm"]).
                 If None, all *.parquet files are auto-discovered (excluding
                 optimized_weights_* and ebm_predictions.parquet).
    """
    if strategies:
        files = []
        for name in strategies:
            p = os.path.join(run_dir, f"{name}.parquet")
            if not os.path.exists(p):
                print(f"  [warn] {p} not found — skipping '{name}'.")
            else:
                files.append(p)
    else:
        all_files = glob.glob(os.path.join(run_dir, "*.parquet"))
        files = [
            f for f in all_files
            if os.path.basename(f) not in _EXCLUDE_EXACT
            and not any(os.path.basename(f).startswith(pfx)
                        for pfx in _EXCLUDE_PREFIXES)
        ]

    if not files:
        raise FileNotFoundError(f"No weight files found in {run_dir}")

    print(
        f"Found {len(files)} strategies: {[os.path.basename(f) for f in files]}")

    raw_strategies = {}
    for f in files:
        name = os.path.basename(f).replace(".parquet", "")
        df = pd.read_parquet(f)
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            df.index = pd.to_datetime(df.index)
        raw_strategies[name] = df

    # Create Union Index/Columns
    master_ts = pd.Index([])
    master_cols = pd.Index([])
    for df in raw_strategies.values():
        master_ts = master_ts.union(df.index)
        master_cols = master_cols.union(df.columns)

    master_ts = master_ts.sort_values().unique()
    master_cols = master_cols.sort_values().unique()

    # Align
    aligned_strategies = {}
    for name, df in raw_strategies.items():
        aligned_df = df.reindex(
            index=master_ts, columns=master_cols).fillna(0.0)
        aligned_strategies[name] = aligned_df

    return aligned_strategies, master_ts, master_cols


def fetch_all_prices(symbols, start_date, end_date):
    """
    Fetches prices for ALL symbols involved to calculate Covariance.
    """
    print(
        f"Fetching price history for {len(symbols)} assets (for Covariance/PnL)...")
    price_frames = {}
    for sym in tqdm(symbols):
        try:
            df = fetch_futures_klines(sym, "1d", start_date, end_date)
            if not df.empty:
                if not pd.api.types.is_datetime64_any_dtype(df['ts']):
                    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
                df.set_index('ts', inplace=True)
                # Prefer futures_close
                price = df['futures_close'] if 'futures_close' in df.columns else df['close']
                price_frames[sym] = price
        except Exception as e:
            print(f"Error fetching {sym}: {e}")

    if not price_frames:
        return pd.DataFrame()

    prices_df = pd.DataFrame(price_frames).ffill()
    return prices_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", required=True,
                    help="Run ID where strategy weights are saved")
    ap.add_argument("--start_date", required=True)
    ap.add_argument("--end_date", required=True)
    ap.add_argument("--strategies", nargs="*", default=None,
                    help="Strategy names to combine (default: all *.parquet in run_dir). "
                         "Example: --strategies momentum reversal ebm")

    # Optimizer Settings
    ap.add_argument("--method", default="linear", choices=["linear", "mean_variance", "equal_weight"],
                    help="Combination method")
    ap.add_argument("--cov_lookback", type=int, default=30,
                    help="Days for covariance calculation")
    ap.add_argument("--risk_aversion", type=float, default=1.0,
                    help="Lambda for Mean-Variance (Higher = Safer)")
    ap.add_argument("--max_leverage", type=float,
                    default=1.0, help="Max Global Leverage")

    # --- FIX 1: Allow user to control concentration ---
    ap.add_argument("--max_position", type=float, default=0.10,
                    help="Max weight per asset (e.g. 0.05 = 20 assets)")

    args = ap.parse_args()

    base_dir = f"./reports/strategies/{args.run_id}"

    # 1. Load Strategy Signals
    strategies, all_ts, all_syms = load_and_align_strategies(
        base_dir, args.strategies)

    # 2. Fetch Market Data
    fetch_start = (pd.to_datetime(args.start_date) -
                   pd.Timedelta(days=args.cov_lookback+20)).strftime('%Y-%m-%d')
    prices_df = fetch_all_prices(all_syms, fetch_start, args.end_date)

    prices_df = prices_df.reindex(all_ts).ffill()
    returns_df = prices_df.pct_change()

    # 3. Prepare Composite Alpha Score
    print("\n--- Computing Composite Signal ---")
    n_strategies = len(strategies)
    composite_alpha = pd.DataFrame(0.0, index=all_ts, columns=all_syms)

    for name, df in strategies.items():
        composite_alpha = composite_alpha.add(df)

    composite_alpha = composite_alpha / n_strategies

    # 4. Run Optimizer (Walk-Forward)
    print(
        f"\n--- Running Optimizer (Method: {args.method}, MaxPos: {args.max_position}) ---")
    optimizer = PortfolioOptimizer(
        max_leverage=args.max_leverage,
        max_position=args.max_position,
        lambda_risk=args.risk_aversion
    )

    final_weights_list = []

    # Track stats
    active_asset_counts = []

    for t in tqdm(all_ts):
        if t not in composite_alpha.index:
            final_weights_list.append(pd.Series(0.0, index=all_syms, name=t))
            continue

        alpha_t = composite_alpha.loc[t]
        # Filter strictly 0 signals to speed up optimization
        active_signals = alpha_t[alpha_t != 0]

        if active_signals.empty:
            final_weights_list.append(pd.Series(0.0, index=all_syms, name=t))
            continue

        w = pd.Series(0.0, index=all_syms)

        if args.method == "equal_weight":
            w = composite_alpha.loc[t]
            lev = w.abs().sum()
            if lev > args.max_leverage:
                w = w * (args.max_leverage / lev)

        elif args.method == "linear":
            w = optimizer.optimize_linear(alpha_t)

        elif args.method == "mean_variance":
            # Slice returns strictly past data
            past_returns = returns_df.loc[:t].iloc[-(args.cov_lookback+1):-1]

            # --- FIX 2: Relaxed Data Cleaning ---
            # Instead of dropping any column with NaN, we only drop columns that are mostly NaN
            missing_pct = past_returns.isnull().mean()
            valid_cols = missing_pct[missing_pct <
                                     0.2].index  # Allow 20% missing

            # Fill remaining small gaps with 0.0 (neutral return assumption)
            valid_returns = past_returns[valid_cols].fillna(0.0)

            # Intersection with Active Signals
            valid_assets = valid_returns.columns.intersection(
                active_signals.index)

            # --- FIX 3: Fallback Logic ---
            if len(valid_assets) < 5:
                # If we don't have enough data for covariance, use Linear optimizer
                # This prevents the "8-9 asset" trap when data is spotty
                w = optimizer.optimize_linear(alpha_t)
            else:
                cov_matrix = valid_returns[valid_assets].cov()

                # Filter Zero Variance (Zombie Assets)
                variances = np.diag(cov_matrix)
                valid_mask = variances > 1e-8
                final_valid_assets = valid_assets[valid_mask]

                if len(final_valid_assets) < 2:
                    w = optimizer.optimize_linear(alpha_t)
                else:
                    cov_matrix = valid_returns[final_valid_assets].cov()
                    w_opt = optimizer.optimize_mean_variance(
                        alpha_t[final_valid_assets], cov_matrix)
                    w = w_opt.reindex(all_syms).fillna(0.0)

        final_weights_list.append(w)
        active_asset_counts.append((w != 0).sum())

    final_weights = pd.DataFrame(final_weights_list)
    final_weights.index = all_ts

    avg_assets = np.mean(active_asset_counts) if active_asset_counts else 0
    print(f"\nAverage Active Assets: {avg_assets:.1f}")

    # 5. Save & Report
    save_path = os.path.join(
        base_dir, f"optimized_weights_{args.method}.parquet")
    final_weights.to_parquet(save_path)
    print(f"Optimized weights saved to: {save_path}")

    # Calculate Performance
    lagged_weights = final_weights.shift(1).fillna(0.0)
    aligned_returns = returns_df.reindex(all_ts).fillna(0.0)

    port_rets = (lagged_weights * aligned_returns).sum(axis=1)

    initial_cash = 10_000
    equity_curve = initial_cash * (1 + port_rets).cumprod()

    # Plot
    eq_df = pd.DataFrame({'equity': equity_curve})
    plot_path = os.path.join(base_dir, f"equity_{args.method}.png")
    plot_equity_curve(eq_df, plot_path)

    total_ret = (equity_curve.iloc[-1] / initial_cash) - 1
    sharpe = (port_rets.mean() / port_rets.std()) * \
        (365**0.5) if port_rets.std() > 0 else 0

    print("\n==== OPTIMIZED PORTFOLIO RESULTS ====")
    print(f"Method:       {args.method}")
    print(f"Final Equity: ${equity_curve.iloc[-1]:,.2f}")
    print(f"Total Return: {total_ret*100:.2f}%")
    print(f"Sharpe Ratio: {sharpe:.2f}")

    eq_df.to_csv(os.path.join(base_dir, f"equity_{args.method}.csv"))


if __name__ == "__main__":
    main()
