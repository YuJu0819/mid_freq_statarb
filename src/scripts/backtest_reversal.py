from ..backtest.engine import run_vectorized_backtest, trim_backtest_result
from ..strategy.liquidation_reversal import LiquidationReversalStrategy
from ..data.universe import load_validated_universe
from ..data.rolling_universe import RollingUniverse, build_symbol_active_mask
from ..data.binance_futures_rest import fetch_futures_klines, fetch_top_long_short_ratio
from ..data.storage import parquet_path, save_bars, load_bars
from ..core.utils import load_config, ensure_dir
from ..backtest.reporting import *
import argparse
import os
import pandas as pd
import glob
import numpy as np


def load_metrics_from_csv_folder(symbol: str, folder: str) -> pd.DataFrame:
    if not os.path.exists(folder):
        return pd.DataFrame()

    pattern = os.path.join(folder, f"{symbol}_metrics_*.csv")
    files = glob.glob(pattern)

    if not files:
        return pd.DataFrame()

    print(f"  > Found {len(files)} local CSVs...", end=" ", flush=True)

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            # Handle potential string vs int timestamps in CSV
            if 'ts' in df.columns:
                # First, force numeric to handle cases where it might be string "1704..."
                # If it's "2024-01-01", this will fail, so we check type or catch error
                try:
                    # If it looks like a float/int string
                    if pd.api.types.is_numeric_dtype(df['ts']):
                        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
                    else:
                        df['ts'] = pd.to_datetime(df['ts'])
                except:
                    # Fallback for mixed formats
                    df['ts'] = pd.to_datetime(df['ts'])

                if 'ls_ratio' not in df.columns:
                    df['ls_ratio'] = 1.0
                dfs.append(df)
        except:
            continue

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs).sort_values(
        'ts').drop_duplicates('ts').reset_index(drop=True)
    return combined


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_date", help="YYYY-MM-DD", required=True)
    parser.add_argument("--end_date", help="YYYY-MM-DD", required=True)
    parser.add_argument(
        "--config", help="Path to config.yaml", default="config.yaml")
    parser.add_argument(
        "--run_id", help="ID for this backtest run", default="default_run")
    parser.add_argument(
        "--no_rolling_universe", action="store_true",
        help="Force fixed universe from config (bypasses rolling universe). "
             "Use this to baseline-test strategy correctness.")
    parser.add_argument(
        "--perf_start_date", default=None,
        help="Optional. Trim performance reporting to start from this date "
             "(YYYY-MM-DD). The saved weights parquet still covers the full "
             "--start_date / --end_date range so downstream consumers (EBM "
             "factor panel) get full warmup. Only equity curve, Sharpe, "
             "summary metrics, score-history factor analysis, and plots are "
             "computed over the trimmed range.")
    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.no_rolling_universe:
        symbols = cfg["backtest"]["symbols"]
        print(f"[FIXED UNIVERSE] Using {len(symbols)} symbols from config "
              f"(rolling universe disabled).")
    else:
        validated = load_validated_universe(args.start_date, args.end_date)
        if validated is not None:
            symbols = validated
            print(f"Loaded validated universe: {len(symbols)} symbols  "
                  f"(run prepare_universe to refresh)")
        else:
            symbols = cfg["backtest"]["symbols"]
            print(f"WARNING: No validated universe found for {args.start_date}→{args.end_date}. "
                  f"Run 'python -m src.scripts.prepare_universe' first. "
                  f"Falling back to all {len(symbols)} config symbols.")
    interval = "1d"

    print(f"\n--- Starting Liquidation Reversal Backtest ---")

    all_data = {}

    for i, sym in enumerate(symbols):
        print(f"[{i+1}/{len(symbols)}] {sym}...", end=" ", flush=True)

        # 1. Load Price
        price_path = parquet_path(
            cfg["general"]["parquet_dir"], sym, f"{interval}_{args.start_date}_to_{args.end_date}")
        df_price = load_bars(price_path)

        if df_price is None or df_price.empty:
            print("Fetching Price...", end=" ")
            df_price = fetch_futures_klines(
                sym, interval, args.start_date, args.end_date)
            if not df_price.empty:
                save_bars(df_price, price_path)

        if df_price.empty:
            print("Skipping (No Price).")
            continue

        # 2. Load L/S Ratio
        ls_suffix = f"{interval}_{args.start_date}_to_{args.end_date}_ls_ratio"
        ls_path = parquet_path(cfg["general"]["parquet_dir"], sym, ls_suffix)
        df_ls = load_bars(ls_path)

        # Zombie Check: Force reload if ls_ratio column is missing
        if df_ls is not None and not df_ls.empty:
            if 'ls_ratio' not in df_ls.columns:
                df_ls = None  # Force reload

        # 3. Load from CSVs
        if df_ls is None or df_ls.empty:
            metrics_dir = "./data/metrics"
            df_ls = load_metrics_from_csv_folder(sym, metrics_dir)
            if not df_ls.empty and 'ls_ratio' in df_ls.columns:
                save_bars(df_ls, ls_path)

        # 4. Fallback API
        if df_ls is None or df_ls.empty:
            print("Fetching API...", end=" ")
            try:
                df_ls = fetch_top_long_short_ratio(
                    sym, interval, args.start_date, args.end_date)
                if not df_ls.empty:
                    save_bars(df_ls, ls_path)
            except:
                pass

        # 5. Merge & Standardize
        if df_ls is not None and not df_ls.empty and 'ls_ratio' in df_ls.columns:
            # Type safety
            if pd.api.types.is_numeric_dtype(df_price['ts']):
                df_price['ts'] = pd.to_datetime(df_price['ts'], unit='ms')
            else:
                df_price['ts'] = pd.to_datetime(df_price['ts'])

            if pd.api.types.is_numeric_dtype(df_ls['ts']):
                df_ls['ts'] = pd.to_datetime(df_ls['ts'], unit='ms')
            else:
                df_ls['ts'] = pd.to_datetime(df_ls['ts'])

            # --- CRITICAL FIX IS HERE ---
            # We must merge BOTH 'ls_ratio' AND 'open_interest'
            merge_cols = ['ts', 'ls_ratio', 'open_interest']

            # 1. Drop 'open_interest' from price if it exists (usually 0 or empty) to avoid conflict
            if 'open_interest' in df_price.columns:
                df_price = df_price.drop(columns=['open_interest'])

            # 2. Merge
            df_merged = pd.merge(
                df_price, df_ls[merge_cols], on='ts', how='left')

            # 3. Fill Gaps
            df_merged['ls_ratio'] = df_merged['ls_ratio'].ffill().fillna(1.0)
            df_merged['open_interest'] = df_merged['open_interest'].ffill().fillna(
                0.0)

            print(f"OK (Records: {len(df_ls)})")

            # DEBUG: Prove we have data now
            # valid_oi = df_merged['open_interest'].sum()
            # print(f"   [DEBUG] Total OI Sum: {valid_oi:,.0f}")

        else:
            print("OK (Default 1.0 - Data Missing)")
            df_merged = df_price.copy()
            df_merged['ls_ratio'] = 1.0
            df_merged['open_interest'] = 0.0  # No data

        # --- FINAL DATA CLEANUP ---
        # 1. Alias 'close' to 'futures_close'
        if 'close' in df_merged.columns:
            df_merged['futures_close'] = df_merged['close']
        if 'open' in df_merged.columns:
            df_merged['futures_open'] = df_merged['open']

        # 2. Final Timestamp Check (The "1970" Fix)
        if pd.api.types.is_numeric_dtype(df_merged['ts']):
            # If it's still an integer at this point, it MUST be ms
            df_merged['ts'] = pd.to_datetime(df_merged['ts'], unit='ms')
        else:
            df_merged['ts'] = pd.to_datetime(df_merged['ts'])

        # Remove timezone to prevent crashes
        if df_merged['ts'].dt.tz is not None:
            df_merged['ts'] = df_merged['ts'].dt.tz_localize(None)

        all_data[sym] = df_merged

    if not all_data:
        print("No valid data loaded.")
        return

    # --- Rolling Universe Epoch Mask -------------------------------------
    # Skipped when --no_rolling_universe is passed (epoch_mask_df stays None,
    # which the engine and strategy treat as "all symbols always active").
    import pandas as _pd
    epoch_mask_df = None
    if not args.no_rolling_universe:
        ru = RollingUniverse()
        if not ru.is_empty():
            ru_epochs = ru.get_epochs(args.start_date, args.end_date)
            if ru_epochs:
                print(
                    f"\nBuilding rolling universe epoch mask ({len(ru_epochs)} epochs)...")
                mask_cols = {}
                for sym, df in all_data.items():
                    ts_idx = _pd.DatetimeIndex(_pd.to_datetime(df["ts"]))
                    active = build_symbol_active_mask(sym, df["ts"], ru_epochs)
                    mask_cols[sym] = _pd.Series(active.values, index=ts_idx)
                epoch_mask_df = _pd.DataFrame(mask_cols)
                active_pairs = int(epoch_mask_df.sum().sum())
                total_pairs = epoch_mask_df.size
                print(
                    f"  Active (date, symbol) pairs: {active_pairs:,} / {total_pairs:,}")
    # ----------------------------------------------------------------------

    # --- Strategy Execution ---
    strategy = LiquidationReversalStrategy(
        leverage_scale=1.0,
        oi_level_lookback=90,
        sentiment_ma_window=90,
        ts_lookback=90,
        half_life_decay=3
    )

    print(f"\nRunning Simulation on {len(all_data)} assets...")
    try:
        res = run_vectorized_backtest(
            all_data, strategy, cfg, run_id=args.run_id, file_name='reversal',
            epoch_mask_df=epoch_mask_df)

        # Trim reporting window if requested. Weights parquet is already saved
        # with the full date range inside run_vectorized_backtest.
        if args.perf_start_date:
            print(f"\n[perf_start_date={args.perf_start_date}] "
                  f"trimming reporting; full weights parquet preserved on disk.")
            res = trim_backtest_result(res, args.perf_start_date)

        print("\n" + "="*40)
        print("           BACKTEST RESULTS           ")
        print("="*40)
        print(f"Final Equity:   ${res.summary['final_equity']:,.2f}")
        print(f"Total Return:   {res.summary['return_pct']:.2f}%")
        print(f"Daily Sharpe:   {res.summary['sharpe_daily']:.2f}")
        print(f"PSR:            {res.summary['prob_sharpe_ratio']:.4f}")
        print("="*40)

        report_dir = ensure_dir(f"./reports/strategies/{args.run_id}")
        res.equity_curve.to_csv(os.path.join(
            report_dir, "equity_curve_reversal.csv"))
        plot_equity_curve(res.equity_curve, os.path.join(
            report_dir, "equity_curve_reversal.png"))

        # analyze_factor_quantiles needs 'close_price', which the reversal strategy
        # doesn't include in score_history. Join it in from all_data here.
        if res.score_history is not None and not res.score_history.empty:
            closes = pd.concat(
                [df.set_index('ts')['futures_close'].rename(sym)
                 for sym, df in all_data.items()],
                axis=1
            ).stack().reset_index()
            closes.columns = ['ts', 'symbol', 'close_price']
            score_df = res.score_history.merge(
                closes, on=['ts', 'symbol'], how='left')

            print("\n==== Cross-Sectional Factor Analysis (Reversal) ====")
            for factor in ['oi_z_score', 'liquidation_shock', 'regime_score', 'interaction_alpha']:
                analyze_factor_quantiles(
                    score_df=score_df,
                    factor_name=factor,
                    quantiles=3,
                    report_dir=report_dir,
                )
    except Exception as e:
        print(f"\n[CRITICAL ERROR] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
