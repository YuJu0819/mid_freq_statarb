import argparse
import os
import pandas as pd
from binance.exceptions import BinanceAPIException
from ..core.utils import load_config
from ..data.binance_rest import fetch_klines as fetch_spot_klines, fetch_futures_klines
from ..data.storage import parquet_path, save_bars, load_bars
from ..backtest.engine import run_multi_asset
from ..backtest.reporting import plot_equity_curve
from ..data.binance_futures_rest import fetch_funding_rate
from ..strategy.ad_mom_spot_future import FinalStrategy


def load_local_oi_data(symbol, start_date, end_date, data_dir="./data/open_interest"):
    """
    Loads and combines downloaded daily open interest CSVs for a given symbol and date range.
    """
    all_oi_df = []
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)

    date_range = pd.date_range(start=start_dt, end=end_dt)

    for single_date in date_range:
        date_str = single_date.strftime('%Y-%m-%d')
        csv_path = os.path.join(data_dir, f"{symbol}-metrics-{date_str}.csv")

        if os.path.exists(csv_path):
            try:
                daily_df = pd.read_csv(csv_path)
                all_oi_df.append(daily_df)
            except Exception as e:
                print(f"Could not read {csv_path}: {e}")

    if not all_oi_df:
        return pd.DataFrame()

    combined_df = pd.concat(all_oi_df, ignore_index=True)

    # --- DEFINITIVE FIX: The 'create_time' column is a date STRING, not milliseconds ---
    combined_df.rename(columns={
                       'sum_open_interest_value': 'open_interest', 'create_time': 'ts'}, inplace=True)
    # Correctly parse the date string
    combined_df['ts'] = pd.to_datetime(combined_df['ts'])

    oi_df = combined_df[['ts', 'open_interest']].copy()
    oi_df.drop_duplicates(subset=['ts'], inplace=True)
    oi_df.sort_values('ts', inplace=True)

    return oi_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--start_date", help="Start date in YYYY-MM-DD format", required=True)
    ap.add_argument(
        "--end_date", help="End date in YYYY-MM-DD format", required=True)
    args = ap.parse_args()

    cfg = load_config()
    symbols = cfg["backtest"]["symbols"]
    interval = "1d"

    all_data = {}
    for symbol in symbols:
        try:
            # Incrementing suffix to ensure new data is processed
            fname_suffix = f"{interval}_{args.start_date}_to_{args.end_date}_final_v15"
            ppath = parquet_path(
                cfg["general"]["parquet_dir"], symbol, fname_suffix)
            df = load_bars(ppath)

            if df is None or len(df) == 0:
                print(f"Processing data for {symbol}...")

                spot_df = fetch_spot_klines(
                    symbol, interval, start_date=args.start_date, end_date=args.end_date)
                futures_df = fetch_futures_klines(
                    symbol, interval, start_date=args.start_date, end_date=args.end_date)

                if spot_df.empty or futures_df.empty:
                    print(
                        f"Missing spot or futures data for {symbol}. Skipping.")
                    continue

                print(f"Loading local Open Interest data for {symbol}...")
                oi_df = load_local_oi_data(
                    symbol, args.start_date, args.end_date)
                if oi_df.empty:
                    print(
                        f"Warning: No local Open Interest data found for {symbol}. OI will be 0.")

                fr_df = fetch_funding_rate(symbol, limit=1000)

                # Timestamps from klines are milliseconds (numeric)
                spot_df['ts'] = pd.to_datetime(spot_df['ts'], unit='ms')
                spot_df.sort_values('ts', inplace=True)
                futures_df['ts'] = pd.to_datetime(futures_df['ts'], unit='ms')
                futures_df.sort_values('ts', inplace=True)

                if not fr_df.empty:
                    fr_df['ts'] = pd.to_datetime(fr_df['ts'], unit='ms')
                    fr_df.sort_values('ts', inplace=True)

                merged_df = pd.merge_asof(
                    left=futures_df, right=spot_df, on='ts', suffixes=('_futures', '_spot'))

                if not oi_df.empty:
                    merged_df = pd.merge_asof(
                        left=merged_df, right=oi_df, on='ts')
                if not fr_df.empty:
                    merged_df = pd.merge_asof(
                        left=merged_df, right=fr_df, on='ts')

                merged_df.rename(columns={
                                 'close_futures': 'futures_close', 'volume_futures': 'futures_volume'}, inplace=True)
                merged_df['basis'] = merged_df['futures_close'] - \
                    merged_df['close_spot']
                merged_df['volume_ratio'] = merged_df['futures_volume'] / \
                    (merged_df['volume_spot'] + 1e-12)

                for col in ['open_interest', 'funding_rate', 'basis', 'volume_ratio']:
                    if col not in merged_df.columns:
                        merged_df[col] = 0.0

                merged_df.ffill(inplace=True)
                merged_df.bfill(inplace=True)
                merged_df.fillna(0, inplace=True)

                df = merged_df
                save_bars(df, ppath)

            all_data[symbol] = df
        except Exception as e:
            print(
                f"An unexpected error occurred while processing {symbol}: {e}. Skipping.")

    if not all_data:
        print("No data was successfully loaded. Exiting backtest.")
        return

    strat = FinalStrategy(lookback=30, quantile=0.2, min_volume_usd=10_000_000,
                          funding_lookback=7, funding_threshold=0.0003)
    res = run_multi_asset(all_data, strat, cfg)

    print("\n==== Summary ====")
    for k, v in res.summary.items():
        print(f"{k}: {v}")

    if res.score_history is not None and not res.score_history.empty:
        score_save_path = os.path.join(
            os.getcwd(), "reports", "score_inspection.csv")
        res.score_history.to_csv(score_save_path, index=False)
        print(f"Score component breakdown saved to: {score_save_path}")


if __name__ == "__main__":
    main()
