"""
Refresh the trading universe by fetching the current top-N symbols from Binance Futures.

This script:
1. Fetches the top-N symbols by 24h volume from Binance Futures API.
2. Compares with the most recent saved snapshot (or config.yaml if no snapshot exists).
3. Shows a diff of added / removed symbols.
4. Downloads Binance Vision metrics (OI + L/S ratio) for newly added symbols.
5. Saves a dated snapshot to data/universe_snapshots/snapshot_YYYY-MM-DD.yaml.
6. Optionally updates config.yaml with the new symbol list (--apply flag).

When to run:
    - Every 6 months (e.g. Jan 1 and Jul 1) as a calendar-based schedule.
    - When prepare_universe shows many rejected symbols — use --check_coverage first.

Usage:
    # Dry-run: show diff and save snapshot, do NOT touch config.yaml
    python -m src.scripts.refresh_universe

    # Check how much of the last snapshot is still in the live top-N
    python -m src.scripts.refresh_universe --check_coverage

    # Full refresh: update config.yaml AND download data for new symbols
    python -m src.scripts.refresh_universe --apply --download_data

    # Custom top-N or earliest download date
    python -m src.scripts.refresh_universe --top_n 200 --since 2024-07-01 --apply

After running with --apply, complete the pipeline update with:
    python -m src.scripts.download_metrics
    python -m src.scripts.download_ls_ratio
    python -m src.scripts.prepare_universe --start_date YYYY-MM-DD --end_date YYYY-MM-DD
"""

import argparse
import os
import shutil
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone

import requests
import yaml
from dateutil.relativedelta import relativedelta

from ..core.logger import get_logger
from ..core.utils import load_config
from ..data.rolling_universe import RollingUniverse, save_snapshot
from ..scripts.get_top_futures_symbols import get_top_futures_symbols

logger = get_logger("refresh_universe")

_SNAPSHOTS_DIR = "./data/universe_snapshots"
_METRICS_DIR   = "./data/metrics"
_DAILY_URL     = "https://data.binance.vision/data/futures/um/daily/metrics"
_MONTHLY_URL   = "https://data.binance.vision/data/futures/um/monthly/metrics"
_MAX_WORKERS   = 16


# ---------------------------------------------------------------------------
# Binance Vision download helpers
# (self-contained copy; mirrors download_metrics.py to avoid global-state coupling)
# ---------------------------------------------------------------------------

def _get_listing_date(symbol: str) -> date:
    """Return the futures listing date for symbol (falls back to 2024-01-01)."""
    try:
        resp = requests.get("https://fapi.binance.com/fapi/v1/exchangeInfo", timeout=15)
        for s in resp.json().get("symbols", []):
            if s["symbol"] == symbol:
                ts = s.get("onboardDate")
                if ts:
                    return datetime.fromtimestamp(ts / 1000).date()
    except Exception as e:
        logger.warning(f"Could not fetch listing date for {symbol}: {e}")
    return date(2024, 1, 1)


def _extract_metrics(csv_path: str):
    """Parse a Binance Vision metrics CSV into a normalised DataFrame, or None."""
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        if "create_time" in df.columns:
            df["ts"] = pd.to_datetime(df["create_time"])
        elif "timestamp" in df.columns:
            df["ts"] = pd.to_datetime(df["timestamp"])
        else:
            return None

        if "sum_open_interest" in df.columns:
            df.rename(columns={"sum_open_interest": "open_interest"}, inplace=True)
        elif "openInterest" in df.columns:
            df.rename(columns={"openInterest": "open_interest"}, inplace=True)

        if "open_interest" not in df.columns:
            return None

        ls_candidates = [
            "count_toptrader_long_short_ratio", "longShortRatio",
            "long_short_ratio", "top_long_short_account_ratio",
            "count_top_trader_long_short_ratio", "sum_toptrader_long_short_ratio",
            "globalLongShortAccountRatio",
        ]
        for col in ls_candidates:
            if col in df.columns:
                df.rename(columns={col: "ls_ratio"}, inplace=True)
                break

        cols = ["ts", "open_interest"]
        if "ls_ratio" in df.columns:
            cols.append("ls_ratio")
        df = df[cols].copy()
        if "ls_ratio" not in df.columns:
            df["ls_ratio"] = 1.0
        return df.sort_values("ts")
    except Exception:
        return None


def _download_daily(symbol: str, date_obj: date):
    """Download one daily metrics zip. Returns status string or None (404/cached)."""
    date_str = date_obj.strftime("%Y-%m-%d")
    final_csv = os.path.join(_METRICS_DIR, f"{symbol}_metrics_{date_str}.csv")
    if os.path.exists(final_csv):
        return None   # already cached

    url      = f"{_DAILY_URL}/{symbol}/{symbol}-metrics-{date_str}.zip"
    temp_zip = os.path.join(_METRICS_DIR, f"temp_{symbol}_{date_str}.zip")
    try:
        with requests.get(url, stream=True, timeout=10) as r:
            if r.status_code == 200:
                with open(temp_zip, "wb") as f:
                    shutil.copyfileobj(r.raw, f)
                with zipfile.ZipFile(temp_zip, "r") as z:
                    csv_name = z.namelist()[0]
                    z.extractall(_METRICS_DIR)
                extracted = os.path.join(_METRICS_DIR, csv_name)
                df = _extract_metrics(extracted)
                status = "OK" if df is not None else "BAD_DATA"
                if df is not None:
                    df.to_csv(final_csv, index=False)
                for p in [extracted, temp_zip]:
                    if os.path.exists(p):
                        os.remove(p)
                return f"{status}: {symbol} {date_str}"
    except Exception:
        pass
    if os.path.exists(temp_zip):
        os.remove(temp_zip)
    return None   # silently skip 404s / network errors


def _download_monthly(symbol: str, year: int, month: int):
    """Download one monthly metrics zip. Returns status string or None (sentinel/404)."""
    # Sentinel: if the 15th already exists the whole month was already downloaded
    sentinel = os.path.join(_METRICS_DIR, f"{symbol}_metrics_{year}-{month:02d}-15.csv")
    if os.path.exists(sentinel):
        return None

    date_str = f"{year}-{month:02d}"
    url      = f"{_MONTHLY_URL}/{symbol}/{symbol}-metrics-{date_str}.zip"
    temp_zip = os.path.join(_METRICS_DIR, f"temp_month_{symbol}_{date_str}.zip")
    try:
        with requests.get(url, stream=True, timeout=20) as r:
            if r.status_code == 200:
                with open(temp_zip, "wb") as f:
                    shutil.copyfileobj(r.raw, f)
                with zipfile.ZipFile(temp_zip, "r") as z:
                    csv_name = z.namelist()[0]
                    z.extractall(_METRICS_DIR)
                extracted = os.path.join(_METRICS_DIR, csv_name)
                df = _extract_metrics(extracted)
                if df is not None:
                    count = 0
                    for gdate, gdf in df.groupby(df["ts"].dt.date):
                        day_s = gdate.strftime("%Y-%m-%d")
                        gdf.to_csv(
                            os.path.join(_METRICS_DIR, f"{symbol}_metrics_{day_s}.csv"),
                            index=False,
                        )
                        count += 1
                    status = f"OK ({count} days)"
                else:
                    status = "BAD_DATA"
                for p in [extracted, temp_zip]:
                    if os.path.exists(p):
                        os.remove(p)
                return f"MONTH_{status}: {symbol} {date_str}"
    except Exception:
        pass
    if os.path.exists(temp_zip):
        os.remove(temp_zip)
    return None


def _download_metrics_for_symbols(symbols: list, since: date):
    """
    Download Binance Vision metrics (monthly then daily) for the given symbols,
    starting from max(listing_date, since) for each symbol.
    """
    os.makedirs(_METRICS_DIR, exist_ok=True)
    today = date.today()

    # Per-symbol start date: listing date or `since`, whichever is later
    sym_starts = {}
    logger.info(f"Fetching listing dates for {len(symbols)} new symbols...")
    for sym in symbols:
        listing = _get_listing_date(sym)
        sym_starts[sym] = max(since, listing)

    # --- Phase 1: monthly archives ---
    monthly_tasks = []
    for sym in symbols:
        cur = sym_starts[sym]
        while cur <= today:
            if date(cur.year, cur.month, 1) < date(today.year, today.month, 1):
                monthly_tasks.append((sym, cur.year, cur.month))
            cur += relativedelta(months=1)
    monthly_tasks = sorted(set(monthly_tasks))

    if monthly_tasks:
        logger.info(f"Monthly archives: {len(monthly_tasks)} tasks ...")
        with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as ex:
            futs = {ex.submit(_download_monthly, *t): t for t in monthly_tasks}
            for i, fut in enumerate(as_completed(futs)):
                res = fut.result()
                if res:
                    print(f"  [{i+1}/{len(monthly_tasks)}] {res}")

    # --- Phase 2: daily gap-fill ---
    daily_tasks = []
    for sym in symbols:
        cur = sym_starts[sym]
        while cur < today:
            daily_tasks.append((sym, cur))
            cur += timedelta(days=1)

    if daily_tasks:
        logger.info(f"Daily gap-fill: {len(daily_tasks)} tasks ...")
        with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as ex:
            futs = {ex.submit(_download_daily, *t): t for t in daily_tasks}
            for i, fut in enumerate(as_completed(futs)):
                res = fut.result()
                if res:
                    print(f"  [Daily] {res}")


# ---------------------------------------------------------------------------
# Coverage check helper
# ---------------------------------------------------------------------------

def _coverage_overlap(snapshot_symbols: list, live_symbols: list) -> float:
    """Fraction of snapshot symbols still present in the live top-N list."""
    snap_set = set(snapshot_symbols)
    live_set = set(live_symbols)
    if not snap_set:
        return 0.0
    return len(snap_set & live_set) / len(snap_set)


# ---------------------------------------------------------------------------
# config.yaml update
# ---------------------------------------------------------------------------

def _update_config_symbols(config_path: str, new_symbols: list):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    cfg["backtest"]["symbols"] = new_symbols
    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    logger.info(f"Updated {config_path} with {len(new_symbols)} symbols.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Refresh the trading universe from Binance Futures.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--top_n", type=int, default=150,
                    help="Number of top symbols by 24h volume (default 150)")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--apply", action="store_true",
                    help="Update config.yaml with the new symbol list (default: dry-run)")
    ap.add_argument("--download_data", action="store_true",
                    help="Download Binance Vision metrics for newly added symbols")
    ap.add_argument("--since", default=None,
                    help="Earliest date to download data from (YYYY-MM-DD). "
                         "Defaults to the date of the previous snapshot.")
    ap.add_argument("--check_coverage", action="store_true",
                    help="Report overlap between last snapshot and live top-N, then exit")
    ap.add_argument("--coverage_threshold", type=float, default=0.85,
                    help="Overlap fraction below which refresh is recommended (default 0.85)")
    args = ap.parse_args()

    ru    = RollingUniverse(_SNAPSHOTS_DIR)
    snaps = ru.list_snapshots()

    # Determine reference (previous) symbol set
    if snaps:
        prev_date    = snaps[-1]
        prev_symbols = ru._load(prev_date)
        logger.info(f"Most recent snapshot: {prev_date} ({len(prev_symbols)} symbols)")
    else:
        cfg          = load_config(args.config)
        prev_symbols = cfg["backtest"]["symbols"]
        prev_date    = None
        logger.info("No previous snapshot found; comparing against config.yaml symbols.")

    # ------------------------------------------------------------------
    # --check_coverage mode: just report and exit
    # ------------------------------------------------------------------
    if args.check_coverage:
        logger.info(f"Fetching live top-{args.top_n} from Binance ...")
        live_syms = get_top_futures_symbols(args.top_n)
        cov = _coverage_overlap(prev_symbols, live_syms)
        ref = prev_date or "config.yaml"
        print(f"\n  Coverage check — {ref} vs live top-{args.top_n}")
        print(f"  Overlap : {cov:.1%}  "
              f"({int(cov * len(prev_symbols))}/{len(prev_symbols)} symbols still in top-{args.top_n})")
        if cov < args.coverage_threshold:
            print(f"  ⚠  Below threshold ({args.coverage_threshold:.0%}) — a refresh is recommended.")
            print(f"     Run: python -m src.scripts.refresh_universe --apply [--download_data]")
        else:
            print(f"  ✓  Above threshold ({args.coverage_threshold:.0%}) — universe is healthy.")
        return

    # ------------------------------------------------------------------
    # Fetch fresh top-N from Binance Futures
    # ------------------------------------------------------------------
    today_str = date.today().isoformat()
    logger.info(f"Fetching top {args.top_n} symbols from Binance Futures ...")
    new_symbols = get_top_futures_symbols(args.top_n)
    new_set     = set(new_symbols)
    prev_set    = set(prev_symbols)
    added       = sorted(new_set - prev_set)
    removed     = sorted(prev_set - new_set)

    print()
    print("=" * 60)
    print(f"  Universe Refresh   {today_str}")
    print("=" * 60)
    print(f"  Previous : {prev_date or 'config.yaml'} — {len(prev_set)} symbols")
    print(f"  New      : {today_str}          — {len(new_set)} symbols")
    print(f"  Added  (+{len(added):3d}) : {added  or '—'}")
    print(f"  Removed(-{len(removed):3d}) : {removed or '—'}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Save snapshot
    # ------------------------------------------------------------------
    generated = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    snap_path = save_snapshot(today_str, new_symbols, args.top_n, generated, _SNAPSHOTS_DIR)
    print(f"\nSnapshot saved → {snap_path}")

    # ------------------------------------------------------------------
    # Download Binance Vision data for new symbols not yet in config
    # ------------------------------------------------------------------
    cfg = load_config(args.config)
    current_config_set = set(cfg["backtest"]["symbols"])
    truly_new = [s for s in added if s not in current_config_set]

    if truly_new:
        print(f"\nSymbols new to config.yaml ({len(truly_new)}): {truly_new}")
        if args.download_data:
            since_date = (
                date.fromisoformat(args.since) if args.since
                else (date.fromisoformat(prev_date) if prev_date else date(2024, 1, 1))
            )
            print(f"Downloading Binance Vision metrics since {since_date} ...")
            _download_metrics_for_symbols(truly_new, since=since_date)
        else:
            print("  → Re-run with --download_data to fetch their historical metrics.")
            print("    Or run: python -m src.scripts.download_metrics  after updating config.yaml")
    else:
        print("\nNo symbols new to config.yaml — metrics download not needed.")

    # ------------------------------------------------------------------
    # Apply to config.yaml
    # ------------------------------------------------------------------
    if args.apply:
        _update_config_symbols(args.config, new_symbols)
        print(f"\nconfig.yaml updated with {len(new_symbols)} symbols.")
        print("\nNext steps:")
        print("  1. python -m src.scripts.download_metrics")
        print("  2. python -m src.scripts.download_ls_ratio")
        print("  3. python -m src.scripts.prepare_universe \\")
        print("        --start_date <PERIOD_START> --end_date <PERIOD_END>")
        print("     # or use rolling mode:")
        print("     python -m src.scripts.prepare_universe --rolling \\")
        print("        --start_date <PERIOD_START> --end_date <PERIOD_END>")
    else:
        print("\n[DRY RUN] config.yaml was NOT modified. Add --apply to update it.")
    print()


if __name__ == "__main__":
    main()
