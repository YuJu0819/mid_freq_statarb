from src.scripts.get_top_futures_symbols import get_top_futures_symbols
from src.data.binance_futures_rest import fetch_open_interest, fetch_top_long_short_ratio
import requests
import os
import zipfile
import pandas as pd
import shutil
import sys
import time
from datetime import date, timedelta, datetime
from dateutil.relativedelta import relativedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.core.utils import load_config  # Import your config loader
# Ensure module path is correct
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../")))


# --- CONFIG ---
# <--- The earliest possible date you want
GLOBAL_START_DATE = date(2024, 1, 1)
END_DATE = date.today()
OUTPUT_DIRECTORY = "./data/metrics"
DAILY_BASE_URL = "https://data.binance.vision/data/futures/um/daily/metrics"
MONTHLY_BASE_URL = "https://data.binance.vision/data/futures/um/monthly/metrics"
MAX_WORKERS = 16
# ---------------------


def get_symbol_listing_date(symbol):
    """
    Fetches the 'onboardDate' from Binance Exchange Info.
    Returns a datetime.date object.
    """
    try:
        # We use a direct request to avoid initializing the full Client just for this
        url = "https://fapi.binance.com/fapi/v1/exchangeInfo"
        resp = requests.get(url)
        data = resp.json()

        for s in data['symbols']:
            if s['symbol'] == symbol:
                # onboardDate is in ms
                ts = s.get('onboardDate')
                if ts:
                    return datetime.fromtimestamp(ts / 1000).date()
    except Exception as e:
        print(f"  [Warning] Could not fetch listing date for {symbol}: {e}")

    # Fallback: Just return global start if check fails
    return GLOBAL_START_DATE


def robust_extract_df(csv_path):
    # ... (Same extraction logic as before) ...
    try:
        df = pd.read_csv(csv_path)
        if 'create_time' in df.columns:
            df['ts'] = pd.to_datetime(df['create_time'])
        elif 'timestamp' in df.columns:
            df['ts'] = pd.to_datetime(df['timestamp'])
        else:
            return None

        if 'sum_open_interest' in df.columns:
            df.rename(
                columns={'sum_open_interest': 'open_interest'}, inplace=True)
        elif 'openInterest' in df.columns:
            df.rename(columns={'openInterest': 'open_interest'}, inplace=True)

        if 'open_interest' not in df.columns:
            return None

        ls_candidates = [
            'count_toptrader_long_short_ratio',
            'longShortRatio',
            'long_short_ratio',
            'top_long_short_account_ratio',
            'count_top_trader_long_short_ratio',
            'sum_toptrader_long_short_ratio',
            'globalLongShortAccountRatio'
        ]

        found_ls = False
        for col in ls_candidates:
            if col in df.columns:
                df.rename(columns={col: 'ls_ratio'}, inplace=True)
                found_ls = True
                break

        cols = ['ts', 'open_interest']
        if found_ls:
            cols.append('ls_ratio')

        df = df[cols].copy()
        if 'ls_ratio' not in df.columns:
            df['ls_ratio'] = 1.0

        return df.sort_values('ts')
    except:
        return None


def process_single_day(task):
    symbol, date_obj = task
    date_str = date_obj.strftime("%Y-%m-%d")
    final_csv_path = os.path.join(
        OUTPUT_DIRECTORY, f"{symbol}_metrics_{date_str}.csv")

    if os.path.exists(final_csv_path):
        return None

    url = f"{DAILY_BASE_URL}/{symbol}/{symbol}-metrics-{date_str}.zip"
    temp_zip = os.path.join(OUTPUT_DIRECTORY, f"temp_{symbol}_{date_str}.zip")

    try:
        with requests.get(url, stream=True, timeout=10) as r:
            if r.status_code == 200:
                with open(temp_zip, 'wb') as f:
                    shutil.copyfileobj(r.raw, f)
                with zipfile.ZipFile(temp_zip, 'r') as z:
                    csv_name = z.namelist()[0]
                    z.extractall(OUTPUT_DIRECTORY)

                extracted_path = os.path.join(OUTPUT_DIRECTORY, csv_name)
                df = robust_extract_df(extracted_path)

                if df is not None:
                    df.to_csv(final_csv_path, index=False)
                    status = "SUCCESS"
                else:
                    status = "BAD_DATA"

                if os.path.exists(extracted_path):
                    os.remove(extracted_path)
                if os.path.exists(temp_zip):
                    os.remove(temp_zip)
                return f"{status}: {symbol} {date_str}"

    except Exception:
        pass

    if os.path.exists(temp_zip):
        os.remove(temp_zip)
    # Don't return MISSING for everything, keeps logs clean
    return None


def process_monthly_zip(task):
    symbol, year, month = task
    date_str = f"{year}-{month:02d}"

    test_day = f"{year}-{month:02d}-15"
    if os.path.exists(os.path.join(OUTPUT_DIRECTORY, f"{symbol}_metrics_{test_day}.csv")):
        return None

    url = f"{MONTHLY_BASE_URL}/{symbol}/{symbol}-metrics-{date_str}.zip"
    temp_zip = os.path.join(
        OUTPUT_DIRECTORY, f"temp_month_{symbol}_{date_str}.zip")

    try:
        with requests.get(url, stream=True, timeout=20) as r:
            if r.status_code == 200:
                with open(temp_zip, 'wb') as f:
                    shutil.copyfileobj(r.raw, f)
                with zipfile.ZipFile(temp_zip, 'r') as z:
                    csv_name = z.namelist()[0]
                    z.extractall(OUTPUT_DIRECTORY)

                extracted_path = os.path.join(OUTPUT_DIRECTORY, csv_name)
                df = robust_extract_df(extracted_path)

                if df is not None:
                    count = 0
                    for group_date, group_df in df.groupby(df['ts'].dt.date):
                        day_s = group_date.strftime("%Y-%m-%d")
                        t_path = os.path.join(
                            OUTPUT_DIRECTORY, f"{symbol}_metrics_{day_s}.csv")
                        group_df.to_csv(t_path, index=False)
                        count += 1
                    status = f"SUCCESS ({count} days)"
                else:
                    status = "BAD_DATA"

                if os.path.exists(extracted_path):
                    os.remove(extracted_path)
                if os.path.exists(temp_zip):
                    os.remove(temp_zip)
                return f"MONTH_{status}: {symbol} {date_str}"

    except Exception:
        pass

    if os.path.exists(temp_zip):
        os.remove(temp_zip)
    return None


def main():
    if not os.path.exists(OUTPUT_DIRECTORY):
        os.makedirs(OUTPUT_DIRECTORY)

    print("1. Fetching Symbol List and Onboard Dates...")
    cfg = load_config("config.yaml")

    # Build the union of symbols across ALL rolling universe snapshots so that
    # metrics are downloaded for every coin that has ever been in any epoch,
    # not just the coins currently in config.yaml.
    from ..data.rolling_universe import RollingUniverse
    ru = RollingUniverse()
    snapshot_symbols: set = set()
    for snap_date in ru.list_snapshots():
        snapshot_symbols.update(ru._load(snap_date))

    config_symbols: list = cfg["backtest"]["symbols"]

    if snapshot_symbols:
        symbols = sorted(snapshot_symbols | set(config_symbols))
        print(f"1. Loaded {len(symbols)} symbols "
              f"(union of {len(ru.list_snapshots())} snapshots + config.yaml; "
              f"config alone has {len(config_symbols)})")
    else:
        symbols = config_symbols
        print(f"1. Loaded {len(symbols)} symbols from config.yaml "
              f"(no rolling snapshots found — using config only)")

    # -----------------------------------------------------
    # NEW STEP: Determine specific start date for each symbol
    # -----------------------------------------------------
    symbol_start_map = {}
    print(f"   > Checking listing dates for {len(symbols)} symbols...")

    for sym in symbols:
        listing_date = get_symbol_listing_date(sym)
        # We start from whichever is later: Global Start or Listing Date
        actual_start = max(GLOBAL_START_DATE, listing_date)
        symbol_start_map[sym] = actual_start
        # print(f"     {sym}: {actual_start}")
    # -----------------------------------------------------

    # --- PHASE 1: MONTHLY (Deep History) ---
    print(f"\n2. Phase 1: Monthly Archives (Threads: {MAX_WORKERS})...")

    monthly_tasks = []

    for sym in symbols:
        sym_start = symbol_start_map[sym]
        cur = sym_start

        while cur <= END_DATE:
            # Only add task if whole month is in range
            if date(cur.year, cur.month, 1) < date(date.today().year, date.today().month, 1):
                monthly_tasks.append((sym, cur.year, cur.month))
            cur += relativedelta(months=1)

    monthly_tasks = sorted(list(set(monthly_tasks)))

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(
            process_monthly_zip, t): t for t in monthly_tasks}
        for i, future in enumerate(as_completed(futures)):
            res = future.result()
            if res:
                print(f"[{i+1}/{len(monthly_tasks)}] {res}")

    # --- PHASE 2: DAILY (Gap Fill) ---
    print(f"\n3. Phase 2: Daily Archives (Smart Range Scan)...")

    daily_tasks = []

    for sym in symbols:
        sym_start = symbol_start_map[sym]
        cur = sym_start

        while cur < date.today():
            daily_tasks.append((sym, cur))
            cur += timedelta(days=1)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_single_day, t)
                                   : t for t in daily_tasks}
        for i, future in enumerate(as_completed(futures)):
            res = future.result()
            if res:
                print(f"  [Daily] {res}")

    print("\n--- Download Complete ---")


if __name__ == "__main__":
    main()
