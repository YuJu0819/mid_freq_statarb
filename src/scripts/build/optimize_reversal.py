from src.scripts.backtest.backtest_reversal import load_metrics_from_csv_folder
from src.core.utils import load_config
from src.data.storage import parquet_path, save_bars, load_bars
from src.data.binance_futures_rest import fetch_futures_klines, fetch_top_long_short_ratio
from src.strategy.liquidation_reversal import LiquidationReversalStrategy
from src.backtest.engine import run_vectorized_backtest
import sys
import os
import argparse
import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

# Ensure module path is correct
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../")))


def load_all_data(cfg, start_date, end_date):
    """
    Loads all data into memory ONCE so we don't reload for every parameter set.
    """
    symbols = cfg["backtest"]["symbols"]
    interval = "1d"
    all_data = {}

    print(f"--- Loading Data for {len(symbols)} Symbols ---")

    for i, sym in enumerate(symbols):
        # 1. Load Price
        price_path = parquet_path(
            cfg["general"]["parquet_dir"], sym, f"{interval}_{start_date}_to_{end_date}")
        df_price = load_bars(price_path)

        if df_price is None or df_price.empty:
            df_price = fetch_futures_klines(
                sym, interval, start_date, end_date)
            if not df_price.empty:
                save_bars(df_price, price_path)

        if df_price.empty:
            continue

        # --- FIX STEP A: Standardize Price Timestamp BEFORE Merge ---
        if pd.api.types.is_numeric_dtype(df_price['ts']):
            df_price['ts'] = pd.to_datetime(df_price['ts'], unit='ms')
        else:
            df_price['ts'] = pd.to_datetime(df_price['ts'])
        # ------------------------------------------------------------

        # 2. Load L/S Ratio
        ls_suffix = f"{interval}_{start_date}_to_{end_date}_ls_ratio"
        ls_path = parquet_path(cfg["general"]["parquet_dir"], sym, ls_suffix)
        df_ls = load_bars(ls_path)

        if df_ls is None or df_ls.empty:
            metrics_dir = "./data/metrics"
            df_ls = load_metrics_from_csv_folder(sym, metrics_dir)
            if not df_ls.empty and 'ls_ratio' in df_ls.columns:
                save_bars(df_ls, ls_path)

        if df_ls is None or df_ls.empty:
            try:
                df_ls = fetch_top_long_short_ratio(
                    sym, interval, start_date, end_date)
                if not df_ls.empty:
                    save_bars(df_ls, ls_path)
            except:
                pass

        # --- FIX STEP B: Standardize LS Timestamp BEFORE Merge ---
        if df_ls is not None and not df_ls.empty:
            if pd.api.types.is_numeric_dtype(df_ls['ts']):
                df_ls['ts'] = pd.to_datetime(df_ls['ts'], unit='ms')
            else:
                df_ls['ts'] = pd.to_datetime(df_ls['ts'])
        # ---------------------------------------------------------

        # 3. Merge
        if df_ls is not None and not df_ls.empty and 'ls_ratio' in df_ls.columns:
            if 'open_interest' in df_price.columns:
                df_price = df_price.drop(columns=['open_interest'])

            # Now both 'ts' columns are guaranteed to be datetime64[ns]
            df_merged = pd.merge(
                df_price, df_ls[['ts', 'ls_ratio', 'open_interest']], on='ts', how='left')

            df_merged['ls_ratio'] = df_merged['ls_ratio'].ffill().fillna(1.0)
            df_merged['open_interest'] = df_merged['open_interest'].ffill().fillna(
                0.0)
        else:
            df_merged = df_price.copy()
            df_merged['ls_ratio'] = 1.0
            df_merged['open_interest'] = 0.0

        # 4. Clean Types (Remaining cleanup)
        if 'close' in df_merged.columns:
            df_merged['futures_close'] = df_merged['close']

        # Ensure 'ts' is clean (just in case of timezone issues)
        if df_merged['ts'].dt.tz is not None:
            df_merged['ts'] = df_merged['ts'].dt.tz_localize(None)

        all_data[sym] = df_merged

    print(f"Loaded {len(all_data)} valid assets.\n")
    return all_data


def run_grid_search(all_data, cfg):
    """
    Executes the loop over parameter combinations.
    """
    # 2. Define Parameter Grid
    param_grid = {
        # 'half_life_decay': [3 * i for i in range(1, 5)],
        # 'ts_lookback': [10 * i for i in range(1, 10)],
        # 'sentiment_ma_window': [10 * i for i in range(1, 10)],
        # 'beta_lookback': [60],
        # 'oi_level_lookback': [30],
        'half_life_decay': [12],
        'ts_lookback': [80],
        'sentiment_ma_window': [40],
        'beta_lookback': [60],
        'oi_level_lookback': [30],
        'regime_filter_threshold': [0.2 * i for i in range(1, 6)],
        'regime_window': [5 * i for i in range(1, 6)]
    }

    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    print(f"--- Starting Grid Search ({len(combinations)} combinations) ---")

    results = []

    for i, params in enumerate(combinations):
        print(f"[{i+1}/{len(combinations)}] Testing: {params} ...", end=" ")

        strategy = LiquidationReversalStrategy(
            half_life_decay=params['half_life_decay'],
            ts_lookback=params['ts_lookback'],
            sentiment_ma_window=params['sentiment_ma_window'],
            beta_lookback=params['beta_lookback'],
            leverage_scale=1.0,
            oi_level_lookback=30,
            regime_filter_threshold=params['regime_filter_threshold'],
            regime_window=params['regime_window']
        )

        try:
            res = run_vectorized_backtest(all_data, strategy, cfg)
            summary = res.summary
            results.append({
                **params,
                'final_equity': summary['final_equity'],
                'return_pct': summary['return_pct'],
                'sharpe': summary['sharpe_daily'],
                'turnover': summary.get('turnover_avg', 0.0)
            })
            print(
                f"Sharpe: {summary['sharpe_daily']:.2f} | Ret: {summary['return_pct']:.1f}%")
        except Exception as e:
            print(f"FAILED: {e}")

    return pd.DataFrame(results)


def visualize_results(df):
    """
    Generates plots from the results DataFrame.
    """
    print("\n--- Generating Visualization ---")

    # 1. Setup
    sns.set_theme(style="whitegrid")

    # 2. Heatmap: Half-Life vs TS Lookback
    # Group by the two main params and average the Sharpe (in case other params vary)
    pivot_df = df.groupby(['half_life_decay', 'ts_lookback'])[
        'sharpe'].mean().unstack()

    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_df, annot=True, cmap="RdYlGn", fmt=".2f", linewidths=.5)
    plt.title("Sharpe Ratio: Half-Life Decay vs TS Lookback")
    plt.ylabel("Half-Life (Days)")
    plt.xlabel("TS Lookback (Days)")
    plt.tight_layout()
    plt.show()

    # 3. Boxplot: Sentiment MA Window Impact
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='sentiment_ma_window', y='sharpe', data=df, palette="Set2")
    plt.title("Stability Check: Sentiment MA Window")
    plt.ylabel("Sharpe Ratio")
    plt.xlabel("MA Window")
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_date", required=True)
    parser.add_argument("--end_date", required=True)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # 1. Load Data
    all_data = load_all_data(cfg, args.start_date, args.end_date)

    # 2. Run Optimization
    df_res = run_grid_search(all_data, cfg)

    # 3. Save & Print
    df_res = df_res.sort_values('sharpe', ascending=False)

    print("\n" + "="*60)
    print("                 OPTIMIZATION RESULTS                 ")
    print("="*60)

    # Dynamic headers based on what columns exist
    cols = [c for c in df_res.columns if c not in ['final_equity', 'turnover']]
    print(tabulate(df_res[cols].head(10), headers=cols,
          tablefmt="grid", floatfmt=".2f"))

    df_res.to_csv("optimization_results.csv", index=False)
    print(f"\nFull results saved to optimization_results.csv")

    # 4. Visualize
    try:
        visualize_results(df_res)
    except Exception as e:
        print(f"Visualization Skipped (Error: {e})")


if __name__ == "__main__":
    main()
