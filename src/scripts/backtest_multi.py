import argparse
import os
import pandas as pd
from binance.exceptions import BinanceAPIException
from ..core.utils import load_config
from ..data.binance_rest import fetch_klines as fetch_spot_klines, fetch_futures_klines
from ..data.storage import parquet_path, save_bars, load_bars
from ..backtest.engine import run_multi_asset
from ..backtest.reporting import (
    plot_equity_curve, generate_regime_analysis_report,
    generate_weekday_analysis_report, generate_skew_analysis_report,
    generate_daily_regime_analysis,  # <-- IMPORTED
    plot_cross_sectional_analysis, plot_daily_regime_pnl_ts
)
from ..data.binance_futures_rest import fetch_funding_rate
from ..strategy.ad_mom_spot_future import FinalStrategy
from .. import factors


def load_local_oi_data(symbol, start_date, end_date, data_dir="./data/open_interest"):
    # ... (Keep existing implementation) ...
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
    combined_df.rename(columns={
                       'sum_open_interest_value': 'open_interest', 'create_time': 'ts'}, inplace=True)
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

    SKEW_LOOKBACK = 45
    SKEW_POSITIVE_THRESHOLD = 0.4
    SKEW_NEGATIVE_THRESHOLD = -0.2

    try:
        print("Preparing BTC data for regime analysis...")
        spot_btc = fetch_spot_klines(
            "BTCUSDT", interval, args.start_date, args.end_date)
        futures_btc = fetch_futures_klines(
            "BTCUSDT", interval, args.start_date, args.end_date)
        if spot_btc.empty or futures_btc.empty:
            raise ValueError("Could not fetch BTC data for regime analysis.")
        btc_df = pd.merge(spot_btc, futures_btc, on='ts',
                          suffixes=('_spot', '_futures'))
        btc_df['ts'] = pd.to_datetime(btc_df['ts'], unit='ms')
        btc_df.rename(columns={'close_futures': 'futures_close',
                               'high_futures': 'high', 'low_futures': 'low'}, inplace=True)

        market_regimes_df = factors.calc_btc_regimes(btc_df)

    except Exception as e:
        print(
            f"CRITICAL: Failed to generate market regimes. Exiting. Error: {e}")
        return

    all_data = {}
    for symbol in symbols:
        try:
            fname_suffix = f"{interval}_{args.start_date}_to_{args.end_date}_skew_analysis_long_factor3"
            ppath = parquet_path(
                cfg["general"]["parquet_dir"], symbol, fname_suffix)
            df = load_bars(ppath)
            if df is None or len(df) == 0:
                print(f"Processing data for {symbol}...")
                spot_df = fetch_spot_klines(
                    symbol, interval, args.start_date, args.end_date)
                futures_df = fetch_futures_klines(
                    symbol, interval, args.start_date, args.end_date)
                if spot_df.empty or futures_df.empty:
                    print(f"Missing data for {symbol}. Skipping.")
                    continue

                oi_df = load_local_oi_data(
                    symbol, args.start_date, args.end_date)
                fr_df = fetch_funding_rate(
                    symbol, args.start_date, args.end_date, limit=1000)

                spot_df['ts'] = pd.to_datetime(spot_df['ts'], unit='ms')
                futures_df['ts'] = pd.to_datetime(futures_df['ts'], unit='ms')
                if not fr_df.empty:
                    fr_df['ts'] = pd.to_datetime(fr_df['ts'], unit='ms')

                merged_df = pd.merge_asof(futures_df.sort_values('ts'), spot_df.sort_values(
                    'ts'), on='ts', suffixes=('_futures', '_spot'))
                if not oi_df.empty:
                    merged_df = pd.merge_asof(
                        merged_df, oi_df.sort_values('ts'), on='ts')
                if not fr_df.empty:
                    merged_df = pd.merge_asof(
                        merged_df, fr_df.sort_values('ts'), on='ts')
                merged_df = pd.merge_asof(merged_df.sort_values(
                    'ts'), market_regimes_df.sort_values('ts'), on='ts')

                merged_df.rename(columns={
                                 'close_futures': 'futures_close', 'volume_futures': 'futures_volume'}, inplace=True)
                merged_df['basis'] = merged_df['futures_close'] - \
                    merged_df['close_spot']
                merged_df['volume_ratio'] = merged_df['futures_volume'] / \
                    (merged_df['volume_spot'].replace(0, 1e-12))

                # --- MODIFICATION: Updated to pass the Series directly ---
                merged_df['skewness'] = factors.calc_skewness(
                    merged_df['futures_close'], lookback=SKEW_LOOKBACK)

                merged_df['skew_regime'] = 'Neutral Skew'
                merged_df.loc[merged_df['skewness'] <
                              SKEW_NEGATIVE_THRESHOLD, 'skew_regime'] = 'Negative Skew'
                merged_df.loc[merged_df['skewness'] >
                              SKEW_POSITIVE_THRESHOLD, 'skew_regime'] = 'Positive Skew'

                for col in ['open_interest', 'funding_rate', 'basis', 'volume_ratio', 'volatility_regime', 'trend_regime', 'adx', 'skew_regime']:
                    if col not in merged_df.columns:
                        if 'regime' in col:
                            merged_df[col] = 'Unknown'
                        elif col == 'adx':
                            merged_df[col] = 0.0
                        else:
                            merged_df[col] = 0.0

                merged_df.ffill(inplace=True)
                merged_df.bfill(inplace=True)
                merged_df.fillna(0, inplace=True)
                df = merged_df
                save_bars(df, ppath)

            all_data[symbol] = df
        except Exception as e:
            print(f"An unexpected error occurred for {symbol}: {e}. Skipping.")

    if not all_data:
        print("No data was successfully loaded. Exiting backtest.")
        return

    strat = FinalStrategy(lookback=30, quantile=0.1, min_volume_usd=10_000_000,
                          funding_lookback=180, funding_z_threshold=1, trend_ma_length=30,
                          smooth_lookback=10, vol_lookback=30, vol_adj_factor=0.5,
                          inverse_in_weak_regime=True)  # Using our new inverse setting

    res = run_multi_asset(all_data, strat, cfg)

    print("\n==== Summary ====")
    for k, v in res.summary.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    if res.score_history is not None and not res.score_history.empty:
        score_path = os.path.join(
            os.getcwd(), "reports", "score_inspection.csv")
        res.score_history.to_csv(score_path, index=False)
        print(f"Score breakdown saved to: {score_path}")

        report_dir = os.path.join(os.getcwd(), "reports")
        if not os.path.exists(report_dir):
            os.makedirs(report_dir)

        plot_cross_sectional_analysis(res.score_history, report_dir)

    if not res.equity_curve.empty:
        report_dir = os.path.join(os.getcwd(), "reports")
        if not os.path.exists(report_dir):
            os.makedirs(report_dir)
        start_str = res.equity_curve.index[0].strftime('%Y-%m-%d')
        end_str = res.equity_curve.index[-1].strftime('%Y-%m-%d')
        save_path = os.path.join(
            report_dir, f"equity_curve_{start_str}_to_{end_str}.png")
        plot_equity_curve(res.equity_curve, save_path)

        # --- NEW CALL: Generate Daily Regime Analysis ---
        generate_daily_regime_analysis(res.equity_curve)
        generate_weekday_analysis_report(res.equity_curve)
        plot_daily_regime_pnl_ts(res.equity_curve, report_dir)
    if not res.trades.empty:
        # Keep the trade-based ones too, for comparison
        generate_regime_analysis_report(res.trades)
        # generate_weekday_analysis_report(res.trades)
        generate_skew_analysis_report(res.trades)


if __name__ == "__main__":
    main()
