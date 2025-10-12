import argparse
import os
import pandas as pd
from ..core.utils import load_config
from ..data.binance_rest import fetch_klines
from ..data.storage import parquet_path, save_bars, load_bars
from ..strategy.sma_cross import SMACross
from ..backtest.engine import run_single_symbol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--lookback_days", type=int, default=120)
    ap.add_argument("--use_testnet", type=lambda s: s.lower()
                    == "true", default=True)
    args = ap.parse_args()

    cfg = load_config()
    ppath = parquet_path(cfg["general"]["parquet_dir"],
                         args.symbol, args.interval)
    df = load_bars(ppath)
    if df is None or len(df) == 0:
        df = fetch_klines(args.symbol, args.interval,
                          args.lookback_days, use_testnet=args.use_testnet)
        save_bars(df, ppath)

    strat = SMACross(fast=10, slow=30, cfg=cfg)
    res = run_single_symbol(df, args.symbol, args.interval, strat, cfg)

    print("==== Summary ====")
    for k, v in res.summary.items():
        print(f"{k}: {v}")
    print("Last 5 equity points:")
    print(res.equity_curve.tail())
    if len(res.trades):
        print("Last 5 trades:")
        print(res.trades.tail())


if __name__ == "__main__":
    main()
