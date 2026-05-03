"""
Reconstruct historical top-N universe snapshots for the 2024-2025 backtest period.

Uses Binance Data Vision permanent archives to discover ALL symbols (including
delisted coins) and fetch their 24h quote volume on each target date. This is the
only reliable method because the Binance Futures REST API returns errors for
delisted symbols.

Data source: https://data.binance.vision/data/futures/um/monthly/klines/{SYM}/1d/
Each monthly zip contains daily 1d OHLCV bars. We read the quote_asset_volume
column for each target date to rank symbols by actual trading volume.

Default snapshot dates (semi-annual rebalance):
    2024-01-01, 2024-07-01, 2025-01-01, 2025-07-01

These create 4 epochs covering 2024-2025 with genuinely different symbol sets,
eliminating survivorship bias from using only today's top-150.

Usage:
    # Build default semi-annual snapshots for 2024-2025, top-150
    python -m src.scripts.build_historical_snapshots

    # Custom dates and top-N
    python -m src.scripts.build_historical_snapshots \\
        --dates 2024-01-01 2024-07-01 2025-01-01 2025-07-01 \\
        --top_n 200

    # Preview only, do not write files
    python -m src.scripts.build_historical_snapshots --dry_run

    # Skip re-downloading symbols already cached
    python -m src.scripts.build_historical_snapshots --no_overwrite

After running, continue with:
    python -m src.scripts.download_metrics
    python -m src.scripts.download_ls_ratio
    python -m src.scripts.prepare_universe --rolling \\
        --start_date 2024-01-01 --end_date 2025-12-31
"""

import argparse
import io
import os
import time
import xml.etree.ElementTree as ET
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timezone
from typing import Optional

import pandas as pd
import requests

from ..data.rolling_universe import save_snapshot

_SNAPSHOTS_DIR  = "./data/universe_snapshots"
_VISION_S3      = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
_KLINES_PREFIX  = "data/futures/um/daily/klines/"    # for symbol enumeration
_MONTHLY_KLINES = "https://data.binance.vision/data/futures/um/monthly/klines"
_DAILY_KLINES   = "https://data.binance.vision/data/futures/um/daily/klines"
_MAX_WORKERS    = 24

DEFAULT_DATES = [
    "2024-01-01",
    "2024-07-01",
    "2025-01-01",
    "2025-07-01",
]


# ---------------------------------------------------------------------------
# Symbol discovery via Data Vision directory listing
# ---------------------------------------------------------------------------

def _list_vision_symbols() -> list[str]:
    """
    Page through the Binance Data Vision S3 bucket to discover every symbol
    that ever had USDT-margined perpetual futures daily klines (including delisted).

    Returns a sorted list of symbol strings like ["BTCUSDT", "ETHUSDT", ...].
    """
    symbols = []
    marker   = ""   # S3 v1 listing uses marker/NextMarker for pagination
    pages    = 0

    print("  Enumerating Binance Data Vision symbol directories ...", end=" ", flush=True)
    while True:
        params: dict = {
            "prefix":    _KLINES_PREFIX,
            "delimiter": "/",
            "max-keys":  "1000",
        }
        if marker:
            params["marker"] = marker

        try:
            r = requests.get(_VISION_S3, params=params, timeout=30)
            r.raise_for_status()
        except Exception as e:
            print(f"\n  [Warning] Data Vision listing failed: {e}")
            break

        ns   = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
        root = ET.fromstring(r.content)

        for el in root.findall("s3:CommonPrefixes/s3:Prefix", ns):
            path = (el.text or "").strip()
            sym  = path[len(_KLINES_PREFIX):].rstrip("/")
            if sym:
                symbols.append(sym)

        truncated = root.findtext("s3:IsTruncated", default="false", namespaces=ns)
        if truncated.lower() == "true":
            marker = root.findtext("s3:NextMarker", default="", namespaces=ns) or ""
            if not marker:
                break   # no marker returned despite truncated — stop to avoid infinite loop
        else:
            break

        pages += 1
        if pages > 100:   # safety guard against infinite pagination
            print("\n  [Warning] Stopped after 100 pages — symbol list may be incomplete.")
            break

    print(f"{len(symbols)} raw symbols found.")
    return sorted(set(symbols))


def _filter_symbols(raw: list[str]) -> list[str]:
    """Keep only USDT perpetuals; exclude leveraged tokens (UP/DOWN/BULL/BEAR)."""
    excluded_suffixes = {"UP", "DOWN", "BULL", "BEAR"}
    out = []
    for s in raw:
        if not s.endswith("USDT"):
            continue
        base = s[:-4]   # strip "USDT"
        if any(base.endswith(tag) for tag in excluded_suffixes):
            continue
        out.append(s)
    return sorted(out)


# ---------------------------------------------------------------------------
# Volume retrieval from Binance Data Vision archives
# ---------------------------------------------------------------------------

def _download_zip_bytes(url: str, timeout: int = 20) -> Optional[bytes]:
    """Download a zip from Data Vision. Returns raw bytes or None on 404/error."""
    try:
        r = requests.get(url, timeout=timeout, stream=True)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.content
    except Exception:
        return None


def _extract_quote_volume(zip_bytes: bytes, target_date: str) -> Optional[float]:
    """
    Given the raw bytes of a Binance Vision monthly (or daily) klines zip,
    return the quote_volume for `target_date` (YYYY-MM-DD), or None.

    Binance Data Vision CSVs have a header row:
        open_time,open,high,low,close,volume,close_time,quote_volume,count,
        taker_buy_volume,taker_buy_quote_volume,ignore
    open_time is in milliseconds UTC.
    """
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
            csv_name = z.namelist()[0]
            with z.open(csv_name) as f:
                df = pd.read_csv(f)   # header=True (default) — CSV has a header row

        if "open_time" not in df.columns or "quote_volume" not in df.columns:
            return None

        df["date_str"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.strftime("%Y-%m-%d")

        row = df[df["date_str"] == target_date]
        if row.empty:
            return None

        return float(row.iloc[0]["quote_volume"])
    except Exception:
        return None


def _fetch_volume_from_vision(symbol: str, target_date: str) -> Optional[float]:
    """
    Fetch the 24h quote-asset volume for `symbol` on `target_date` from Data Vision.

    Strategy:
      1. Try the monthly zip for that month (contains all days of the month).
      2. Fall back to the daily zip for that specific date.

    Returns None if the symbol had no data on that date (not yet listed, or delisted
    before that date).
    """
    d      = date.fromisoformat(target_date)
    ym     = f"{d.year}-{d.month:02d}"

    # 1. Monthly zip
    monthly_url = f"{_MONTHLY_KLINES}/{symbol}/1d/{symbol}-1d-{ym}.zip"
    raw = _download_zip_bytes(monthly_url)
    if raw:
        vol = _extract_quote_volume(raw, target_date)
        if vol is not None:
            return vol

    # 2. Daily zip fallback
    daily_url = f"{_DAILY_KLINES}/{symbol}/1d/{symbol}-1d-{target_date}.zip"
    raw = _download_zip_bytes(daily_url)
    if raw:
        vol = _extract_quote_volume(raw, target_date)
        if vol is not None:
            return vol

    return None


def _fetch_volumes_for_date(
    symbols: list[str],
    target_date: str,
    max_workers: int = _MAX_WORKERS,
) -> dict[str, float]:
    """
    Concurrently fetch 24h quote-asset volume for all symbols on `target_date`
    from Binance Data Vision archives (works for delisted symbols).

    Returns {symbol: volume} for symbols that had data on that date.
    """
    volumes: dict[str, float] = {}
    total  = len(symbols)
    done   = 0

    def _task(sym: str):
        return sym, _fetch_volume_from_vision(sym, target_date)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_task, s): s for s in symbols}
        for fut in as_completed(futs):
            sym, vol = fut.result()
            done += 1
            if vol is not None and vol > 0:
                volumes[sym] = vol
            if done % 50 == 0 or done == total:
                print(f"    {done}/{total} checked, {len(volumes)} with data ...",
                      end="\r", flush=True)

    print()   # newline after \r progress
    return volumes


# ---------------------------------------------------------------------------
# Snapshot builder
# ---------------------------------------------------------------------------

def build_snapshot_for_date(
    target_date: str,
    symbols: list[str],
    top_n: int,
    dry_run: bool = False,
) -> Optional[str]:
    """
    Fetch volumes from Data Vision for `target_date`, rank symbols, and save
    a snapshot YAML.  Returns saved path or None (dry_run / error).
    """
    print(f"\n── {target_date}  ({len(symbols)} candidates)")
    volumes = _fetch_volumes_for_date(symbols, target_date)

    if not volumes:
        print(f"  [Skip] No volume data found for {target_date}. "
              f"All symbols may predate or postdate this date.")
        return None

    ranked      = sorted(volumes.items(), key=lambda x: x[1], reverse=True)
    top_symbols = [s for s, _ in ranked[:top_n]]
    actual_n    = len(top_symbols)

    print(f"  {len(volumes)} symbols had volume data.")
    print(f"  Top-{actual_n} selected (requested {top_n}).")
    if ranked:
        print(f"  #1  {ranked[0][0]}   volume={ranked[0][1]:>20,.0f} USDT")
    if len(ranked) >= top_n:
        print(f"  #{top_n:3d} {ranked[top_n-1][0]}   volume={ranked[top_n-1][1]:>20,.0f} USDT")

    if dry_run:
        print(f"  [DRY RUN] Would save snapshot_{target_date}.yaml — {actual_n} symbols.")
        return None

    generated = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    path = save_snapshot(
        snapshot_date=target_date,
        symbols=top_symbols,
        top_n=top_n,
        generated=generated,
        snapshots_dir=_SNAPSHOTS_DIR,
        source="binance_data_vision_1d_quote_volume_historical",
    )
    print(f"  Saved → {path}")
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Build historical universe snapshots from Binance Data Vision archives.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "--dates", nargs="+", default=DEFAULT_DATES,
        help="Snapshot dates (YYYY-MM-DD). Default: semi-annual 2024-2025.",
    )
    ap.add_argument(
        "--top_n", type=int, default=150,
        help="Number of symbols per snapshot (default 150).",
    )
    ap.add_argument(
        "--dry_run", action="store_true",
        help="Show rankings but do NOT write snapshot files.",
    )
    ap.add_argument(
        "--no_overwrite", action="store_true",
        help="Skip dates for which a snapshot file already exists.",
    )
    ap.add_argument(
        "--workers", type=int, default=_MAX_WORKERS,
        help=f"Parallel download threads (default {_MAX_WORKERS}). "
             f"Lower if rate-limited.",
    )
    args = ap.parse_args()

    os.makedirs(_SNAPSHOTS_DIR, exist_ok=True)

    # Validate date format
    for d in args.dates:
        try:
            date.fromisoformat(d)
        except ValueError:
            print(f"ERROR: Invalid date format '{d}'. Expected YYYY-MM-DD.")
            return

    print("=" * 65)
    print("  Historical Universe Snapshot Builder")
    print("  Source: Binance Data Vision archives (incl. delisted coins)")
    print("=" * 65)
    print(f"  Snapshot dates : {', '.join(sorted(args.dates))}")
    print(f"  Top-N          : {args.top_n}")
    print(f"  Workers        : {args.workers}")
    print(f"  Dry run        : {args.dry_run}")
    print("=" * 65)

    # ---- Step 1: discover all candidate symbols ----
    print("\n[1/2] Discovering candidate symbols from Data Vision ...")
    raw_symbols  = _list_vision_symbols()
    candidates   = _filter_symbols(raw_symbols)
    print(f"  {len(raw_symbols)} raw → {len(candidates)} USDT perpetuals (excl. leveraged tokens)")

    # ---- Step 2: build one snapshot per target date ----
    print(f"\n[2/2] Fetching volumes and ranking for each date ...")
    saved_paths = []
    skipped     = []

    for target_date in sorted(args.dates):
        snap_file = os.path.join(_SNAPSHOTS_DIR, f"snapshot_{target_date}.yaml")
        if args.no_overwrite and os.path.exists(snap_file):
            print(f"\n── {target_date}  [SKIP — file already exists]")
            skipped.append(target_date)
            continue

        path = build_snapshot_for_date(
            target_date=target_date,
            symbols=candidates,
            top_n=args.top_n,
            dry_run=args.dry_run,
        )
        if path:
            saved_paths.append(path)

        time.sleep(0.5)   # brief pause between dates

    # ---- Summary ----
    print()
    print("=" * 65)
    if saved_paths:
        print(f"  Saved {len(saved_paths)} snapshot(s):")
        for p in saved_paths:
            print(f"    {p}")
    if skipped:
        print(f"  Skipped (already existed): {skipped}")
    if args.dry_run:
        print("  DRY RUN — no files were written.")
    print("=" * 65)

    if saved_paths and not args.dry_run:
        print()
        print("Next steps:")
        print("  # Download OI + L/S ratio data for all snapshot symbols:")
        print("  python -m src.scripts.download_metrics")
        print("  python -m src.scripts.download_ls_ratio")
        print()
        print("  # Validate and cache price data, build rolling universe file:")
        print("  python -m src.scripts.prepare_universe --rolling \\")
        print("      --start_date 2024-01-01 --end_date 2025-12-31")
    print()


if __name__ == "__main__":
    main()
