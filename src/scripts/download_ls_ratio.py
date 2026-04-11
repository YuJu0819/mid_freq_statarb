"""
Incremental downloader for Long/Short Ratio data.

The Binance top L/S ratio API retains only the last 30 days of history.
The workaround is to run this script periodically (every ≤25 days) so that
successive downloads overlap and a full historical dataset is built up in
./data/ls_ratio/{SYMBOL}_ls_ratio.parquet.

The loader (src/data/loader.py) checks this accumulation store before
falling back to the live API.

Usage:
    # First time (or after a long gap — still capped to 30 days by API):
    python -m src.scripts.download_ls_ratio

    # Scheduled run (e.g. weekly cron):
    python -m src.scripts.download_ls_ratio --days 7

    # Custom config:
    python -m src.scripts.download_ls_ratio --config config.yaml --days 14
"""
import argparse
import os
import time
from datetime import datetime, timedelta, timezone

import pandas as pd

from ..core.utils import load_config, ensure_dir
from ..data.binance_futures_rest import fetch_top_long_short_ratio

# Where accumulated ls_ratio parquet files are stored (one file per symbol).
LS_RATIO_DIR = "./data/ls_ratio"


def _accumulated_path(symbol: str) -> str:
    return os.path.join(LS_RATIO_DIR, f"{symbol}_ls_ratio.parquet")


def _load_accumulated(symbol: str) -> pd.DataFrame:
    path = _accumulated_path(symbol)
    if not os.path.exists(path):
        return pd.DataFrame(columns=["ts", "ls_ratio"])
    return pd.read_parquet(path)


def _save_accumulated(df: pd.DataFrame, symbol: str) -> None:
    ensure_dir(LS_RATIO_DIR)
    df = (
        df[["ts", "ls_ratio"]]
        .drop_duplicates("ts")
        .sort_values("ts")
        .reset_index(drop=True)
    )
    df.to_parquet(_accumulated_path(symbol), index=False)


def main():
    ap = argparse.ArgumentParser(
        description="Incrementally download and accumulate L/S ratio history."
    )
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument(
        "--days",
        type=int,
        default=30,
        help="Days of history to request from the API (max 30, Binance limit).",
    )
    ap.add_argument("--interval", default="1d", choices=["5m", "1h", "1d"])
    args = ap.parse_args()

    cfg = load_config(args.config)
    symbols = cfg.get("backtest", {}).get("symbols", [])
    if not symbols:
        print("No symbols found in config['backtest']['symbols']. Exiting.")
        return

    days = min(args.days, 29)   # cap at 29 days: date truncation to midnight
                                # means "30 days ago" formatted as YYYY-MM-DD
                                # is always a few hours behind the exact cutoff,
                                # causing false "older than 30 days" warnings.
                                # 29 days keeps every request safely inside the
                                # Binance 30-day retention window.
    end_dt = datetime.now(tz=timezone.utc)
    start_dt = end_dt - timedelta(days=days)
    # Pass full ISO-8601 timestamps (not date-only strings) so pd.to_datetime
    # preserves the exact time and the start is never accidentally pushed
    # behind the 30-day cutoff by midnight truncation.
    start_str = start_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    end_str   = end_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    print("=" * 55)
    print("  Incremental L/S Ratio Download")
    print("=" * 55)
    print(f"  Symbols  : {len(symbols)}")
    print(f"  Interval : {args.interval}")
    print(f"  Fetch    : {start_str}  →  {end_str}  (API max 30 days)")
    print(f"  Store    : {LS_RATIO_DIR}/")
    print("=" * 55)

    new_count = skipped = failed = 0

    for i, sym in enumerate(symbols):
        print(f"[{i+1}/{len(symbols)}] {sym}...", end=" ", flush=True)

        try:
            df_new = fetch_top_long_short_ratio(
                symbol=sym,
                interval=args.interval,
                start_str=start_str,
                end_str=end_str,
            )
        except Exception as e:
            print(f"FAILED: {e}")
            failed += 1
            continue

        if df_new is None or df_new.empty:
            print("No data returned (symbol may be invalid or too new).")
            skipped += 1
            continue

        # Normalise ts to UTC-naive datetime so it merges cleanly
        if pd.api.types.is_numeric_dtype(df_new["ts"]):
            df_new["ts"] = pd.to_datetime(df_new["ts"], unit="ms", utc=True).dt.tz_localize(None)
        elif df_new["ts"].dt.tz is not None:
            df_new["ts"] = df_new["ts"].dt.tz_convert(None)

        # Merge with existing accumulated data
        df_old = _load_accumulated(sym)
        if not df_old.empty and df_old["ts"].dt.tz is not None:
            df_old["ts"] = df_old["ts"].dt.tz_convert(None)

        # Filter out empty frames before concat to avoid pandas FutureWarning
        # about dtype inference with all-NA entries (triggered on first-run
        # when df_old is an empty skeleton DataFrame).
        frames = [df for df in [df_old, df_new] if not df.empty]
        df_combined = pd.concat(frames, ignore_index=True) if frames else df_new.copy()
        before = len(df_old)
        _save_accumulated(df_combined, sym)
        after = len(_load_accumulated(sym))

        added = after - before
        new_count += added
        print(f"{added:+d} new records  →  {after} total.")

        time.sleep(0.2)

    print()
    print(f"Done. Added {new_count} records across {len(symbols)} symbols "
          f"({failed} failed, {skipped} skipped).")
    print(f"Tip: schedule this script every ≤25 days to maintain full history.")


if __name__ == "__main__":
    main()
