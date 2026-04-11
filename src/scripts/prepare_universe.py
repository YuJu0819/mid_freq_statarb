"""
Pre-download and validate the trading universe for backtesting.

Run this script BEFORE any backtest. It:
  1. Pre-fetches and caches futures + spot price data for all config symbols.
  2. Validates price coverage (must have >= min_coverage fraction of expected days).
  3. Checks metrics coverage (OI + ls_ratio from ./data/metrics/ or ./data/ls_ratio/).
  4. Saves ./data/universe_{start}_{end}.yaml with the agreed symbol list so that
     BOTH the momentum and reversal backtest scripts operate on the exact same universe.

Full pre-load pipeline:
    # Step 1 — historical OI + ls_ratio archives (run once, or monthly)
    python -m src.scripts.download_metrics

    # Step 2 — recent l/s ratio accumulation (run every <=25 days)
    python -m src.scripts.download_ls_ratio

    # Step 3 — validate universe + pre-cache price data  (run per backtest window)
    python -m src.scripts.prepare_universe --start_date 2024-01-01 --end_date 2025-12-31

    # Step 4 — backtests (both read the validated universe automatically)
    python -m src.scripts.backtest_multi    --start_date 2024-01-01 --end_date 2025-12-31
    python -m src.scripts.backtest_reversal --start_date 2024-01-01 --end_date 2025-12-31
"""
import argparse
import os
import time
from datetime import datetime, timezone

import pandas as pd

from ..core.utils import load_config, ensure_dir
from ..data.binance_rest import fetch_klines as fetch_spot_klines
from ..data.binance_futures_rest import fetch_futures_klines
from ..data.storage import parquet_path, save_bars, load_bars
from ..data.universe import save_validated_universe

METRICS_DIR = "./data/metrics"
LS_RATIO_DIR = "./data/ls_ratio"
MIN_COVERAGE = 0.80   # fraction of expected trading days


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_spot_symbol(futures_symbol: str) -> str:
    if futures_symbol.startswith("1000"):
        return futures_symbol[4:]
    return futures_symbol


def _expected_trading_days(start_date: str, end_date: str) -> int:
    """Business-day count as a proxy for expected trading days."""
    return len(pd.bdate_range(start_date, end_date))


def _price_coverage(df: pd.DataFrame, expected: int) -> float:
    if df is None or df.empty:
        return 0.0
    return min(len(df) / max(expected, 1), 1.0)


def _metrics_coverage(symbol: str, start_date: str, end_date: str) -> float:
    """
    Return fraction of calendar days in range that have metrics (OI / ls_ratio).
    Checks ./data/metrics/ CSVs first, then ./data/ls_ratio/ parquet.
    """
    expected = len(pd.date_range(start_date, end_date))
    t0 = pd.to_datetime(start_date)
    t1 = pd.to_datetime(end_date)

    # --- Local metrics CSVs (download_metrics.py output) ---
    import glob
    csv_files = glob.glob(os.path.join(METRICS_DIR, f"{symbol}_metrics_*.csv"))
    if csv_files:
        rows = []
        for f in csv_files:
            try:
                df = pd.read_csv(f, usecols=["ts"])
                rows.append(df)
            except Exception:
                continue
        if rows:
            df_ts = pd.concat(rows, ignore_index=True)
            df_ts["ts"] = pd.to_datetime(df_ts["ts"], errors="coerce")
            in_range = df_ts[(df_ts["ts"] >= t0) & (df_ts["ts"] <= t1)]
            return min(len(in_range) / max(expected, 1), 1.0)

    # --- Accumulated ls_ratio parquet (download_ls_ratio.py output) ---
    acc_path = os.path.join(LS_RATIO_DIR, f"{symbol}_ls_ratio.parquet")
    if os.path.exists(acc_path):
        try:
            df_acc = pd.read_parquet(acc_path, columns=["ts"])
            if not df_acc.empty:
                df_acc["ts"] = pd.to_datetime(df_acc["ts"], utc=True).dt.tz_localize(None)
                in_range = df_acc[(df_acc["ts"] >= t0) & (df_acc["ts"] <= t1)]
                return min(len(in_range) / max(expected, 1), 1.0)
        except Exception:
            pass

    return 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Pre-download prices and build a validated backtest universe."
    )
    ap.add_argument("--start_date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end_date",   required=True, help="YYYY-MM-DD")
    ap.add_argument("--config",     default="config.yaml")
    ap.add_argument(
        "--min_coverage", type=float, default=MIN_COVERAGE,
        help=f"Min fraction of expected trading days required (default {MIN_COVERAGE:.0%}).",
    )
    ap.add_argument(
        "--no_cache", action="store_true",
        help="Force re-download of all price data ignoring existing cache.",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    symbols      = cfg["backtest"]["symbols"]
    parquet_dir  = cfg["general"]["parquet_dir"]
    interval     = "1d"

    ensure_dir(parquet_dir)
    expected_days = _expected_trading_days(args.start_date, args.end_date)

    print("=" * 60)
    print("  Universe Preparation")
    print("=" * 60)
    print(f"  Period   : {args.start_date} → {args.end_date}  ({expected_days} trading days)")
    print(f"  Symbols  : {len(symbols)} from config")
    print(f"  Min cov  : {args.min_coverage:.0%}")
    print("=" * 60)

    results = []   # (symbol, fut_cov, spot_ok, metrics_cov, accepted)

    for i, sym in enumerate(symbols):
        print(f"[{i+1}/{len(symbols)}] {sym} ...", end=" ", flush=True)

        # ---- Futures price ----
        # Use the same cache key as DataLoader (reversal_price) so both
        # prepare_universe and the DataLoader share one cached file.
        fut_key   = f"{interval}_{args.start_date}_to_{args.end_date}_reversal_price"
        fut_cache = parquet_path(parquet_dir, sym, fut_key)
        df_fut    = None if args.no_cache else load_bars(fut_cache)
        if df_fut is None or df_fut.empty:
            df_fut = fetch_futures_klines(sym, interval, args.start_date, args.end_date)
            if df_fut is not None and not df_fut.empty:
                save_bars(df_fut, fut_cache)
        fut_cov = _price_coverage(df_fut, expected_days)

        # ---- Spot price (needed by momentum) ----
        spot_sym   = _normalize_spot_symbol(sym)
        spot_key   = f"{interval}_{args.start_date}_to_{args.end_date}_spot"
        spot_cache = parquet_path(parquet_dir, spot_sym, spot_key)
        df_spot    = None if args.no_cache else load_bars(spot_cache)
        if df_spot is None or df_spot.empty:
            df_spot = fetch_spot_klines(spot_sym, interval, args.start_date, args.end_date)
            if df_spot is not None and not df_spot.empty:
                save_bars(df_spot, spot_cache)
        spot_ok = df_spot is not None and not df_spot.empty

        # ---- Metrics (OI + ls_ratio) ----
        metrics_cov = _metrics_coverage(sym, args.start_date, args.end_date)

        accepted = fut_cov >= args.min_coverage
        results.append((sym, fut_cov, spot_ok, metrics_cov, accepted))

        flag = "✓" if accepted else "✗"
        spot_tag = "spot✓" if spot_ok else "spot✗"
        print(f"{flag}  futures={fut_cov:.0%}  {spot_tag}  metrics={metrics_cov:.0%}")

        time.sleep(0.3)   # light throttle between symbols

    # ---- Summary ----
    accepted  = [sym for sym, *_, ok in results if ok]
    rejected  = [sym for sym, *_, ok in results if not ok]
    low_metrics = [
        sym for sym, fc, _, mc, ok in results
        if ok and mc < args.min_coverage
    ]

    print()
    print("=" * 60)
    print(f"  Accepted : {len(accepted)}")
    if rejected:
        print(f"  Rejected (insufficient futures price data): {len(rejected)}")
        for sym in rejected:
            print(f"    - {sym}")
    if low_metrics:
        print(f"\n  Accepted but low metrics coverage (<{args.min_coverage:.0%}):")
        for sym in low_metrics:
            mc = next(mc for s, _, _, mc, _ in results if s == sym)
            print(f"    - {sym}: metrics={mc:.0%}  (ls_ratio will default to 1.0)")
    print("=" * 60)

    # ---- Write shared universe file ----
    generated = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    path = save_validated_universe(
        start_date=args.start_date,
        end_date=args.end_date,
        symbols=accepted,
        rejected=rejected,
        low_metrics_warning=low_metrics,
        min_coverage=args.min_coverage,
        generated=generated,
    )

    print(f"\nSaved → {path}")
    print(f"Both backtest scripts will load this universe automatically.\n")


if __name__ == "__main__":
    main()
