import os
import pandas as pd
import time
from binance.client import Client
from binance.exceptions import BinanceAPIException
import requests  # <--- CRITICAL FIX: Added missing import
from dotenv import load_dotenv
from typing import Optional, List, Dict
load_dotenv()

api_key = os.getenv("BINANCE_API_KEY")
api_secret = os.getenv("BINANCE_API_SECRET")
use_testnet = os.getenv("BINANCE_USE_TESTNET", "false").lower() == "true"

client = Client(api_key, api_secret, testnet=use_testnet)

# --- V-- NEW: RATE LIMIT WRAPPER --V ---
BASE_URL = "https://fapi.binance.com"  # <--- Added missing parameter


def safe_api_call(func, *args, **kwargs):
    """
    Wraps API calls with automatic retry on Rate Limit (HTTP 429 / Code -1003).
    """
    MAX_RETRIES = 3
    for attempt in range(MAX_RETRIES):
        try:
            return func(*args, **kwargs)
        except BinanceAPIException as e:
            if e.code == -1003:  # Too many requests
                print(f"   [RATE LIMIT HIT] Cooling down for 60 seconds...")
                time.sleep(60)
                continue
            raise e
    return None
# ---------------------------------------


def fetch_futures_klines(symbol: str, interval: str, start_date: str, end_date: str, limit: int = 1500) -> pd.DataFrame:
    try:
        start_ts = int(pd.to_datetime(start_date).timestamp() * 1000)
        end_ts = int(pd.to_datetime(end_date).timestamp() * 1000)

        all_klines = []
        current_start = start_ts

        while True:
            # Use safe_api_call instead of direct client call
            klines = safe_api_call(
                client.futures_klines,
                symbol=symbol,
                interval=interval,
                startTime=current_start,
                endTime=end_ts,
                limit=limit
            )

            if not klines:
                break

            all_klines.extend(klines)

            # Pagination: Move time forward
            last_close_time = klines[-1][6]
            current_start = last_close_time + 1

            if len(klines) < limit or current_start > end_ts:
                break

            # Short sleep between pages
            time.sleep(0.2)

        if not all_klines:
            return pd.DataFrame()

        df = pd.DataFrame(all_klines, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
        ])

        df["ts"] = pd.to_numeric(df["open_time"])
        df["open"] = pd.to_numeric(df["open"])
        df["high"] = pd.to_numeric(df["high"])
        df["low"] = pd.to_numeric(df["low"])
        df["close"] = pd.to_numeric(df["close"])
        df["volume"] = pd.to_numeric(df["volume"])

        # Deduplicate
        df = df.drop_duplicates(subset=['ts']).sort_values('ts')

        return df[["ts", "open", "high", "low", "close", "volume"]]

    except BinanceAPIException as e:
        print(f"Error fetching futures klines for {symbol}: {e}")
        return pd.DataFrame()


def fetch_open_interest(symbol: str, interval: str = "1d", start_date: str = None, end_date: str = None, limit: int = 500) -> pd.DataFrame:
    try:
        start_ts = int(pd.to_datetime(start_date).timestamp()
                       * 1000) if start_date else None
        end_ts = int(pd.to_datetime(end_date).timestamp()
                     * 1000) if end_date else int(time.time() * 1000)

        all_oi = []
        current_start = start_ts

        while True:
            oi_data = safe_api_call(
                client.futures_open_interest_hist,
                symbol=symbol,
                period=interval,
                limit=limit,
                startTime=current_start,
                endTime=end_ts,
            )

            if not oi_data:
                break

            all_oi.extend(oi_data)

            last_ts = oi_data[-1]["timestamp"]
            current_start = last_ts + 1

            if len(oi_data) < limit or current_start > end_ts:
                break

            time.sleep(0.1)

        if not all_oi:
            return pd.DataFrame()

        df = pd.DataFrame(all_oi)
        df["sumOpenInterestValue"] = pd.to_numeric(
            df["sumOpenInterestValue"], errors="coerce")
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")

        df.rename(columns={"sumOpenInterestValue": "open_interest",
                  "timestamp": "ts"}, inplace=True)
        df = df.drop_duplicates(subset=["ts"]).sort_values("ts").reset_index(drop=True)
        return df[["ts", "open_interest"]]

    except BinanceAPIException as e:
        # 1130 error is handled by caller or ignored
        return pd.DataFrame()


def fetch_funding_rate(symbol: str, start_date: str = None, end_date: str = None, limit: int = 1000) -> pd.DataFrame:
    """
    Fetches historical funding rates with PAGINATION (Fixes the missing 2024 data).
    """
    try:
        start_ts = int(pd.to_datetime(start_date).timestamp()
                       * 1000) if start_date else None
        end_ts = int(pd.to_datetime(end_date).timestamp() *
                     1000) if end_date else int(time.time() * 1000)

        all_rates = []
        current_start = start_ts

        # print(f"Fetching funding rates for {symbol}...")

        while True:
            fr_data = safe_api_call(
                client.futures_funding_rate,
                symbol=symbol,
                startTime=current_start,
                endTime=end_ts,
                limit=limit
            )

            if not fr_data:
                break

            all_rates.extend(fr_data)

            last_timestamp = fr_data[-1]['fundingTime']
            current_start = last_timestamp + 1

            if len(fr_data) < limit or current_start > end_ts:
                break

            time.sleep(0.1)

        if not all_rates:
            return pd.DataFrame()

        df = pd.DataFrame(all_rates)
        df["fundingTime"] = pd.to_numeric(df["fundingTime"], errors="coerce")
        df["fundingRate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
        df.rename(columns={"fundingTime": "ts",
                  "fundingRate": "funding_rate"}, inplace=True)
        df = df.drop_duplicates(subset=['ts']).sort_values(
            'ts').reset_index(drop=True)

        return df[["ts", "funding_rate"]]

    except BinanceAPIException as e:
        print(f"Error fetching funding rate for {symbol}: {e}")
        return pd.DataFrame()


def fetch_top_long_short_ratio(
    symbol: str,
    interval: str,
    start_str: Optional[str] = None,
    end_str: Optional[str] = None,
    limit: int = 500,
) -> pd.DataFrame:

    # URL for "Top Trader Account Ratio".
    # Switch to /globalLongShortAccountRatio if you want the Global ratio.
    url = "https://fapi.binance.com/futures/data/topLongShortAccountRatio"

    # 1. Map interval to milliseconds for dynamic chunking
    # This prevents the "Limit vs Chunk" bug
    interval_map = {
        "5m": 5 * 60 * 1000,
        "15m": 15 * 60 * 1000,
        "30m": 30 * 60 * 1000,
        "1h": 60 * 60 * 1000,
        "2h": 2 * 60 * 60 * 1000,
        "4h": 4 * 60 * 60 * 1000,
        "1d": 24 * 60 * 60 * 1000,
    }

    if interval not in interval_map:
        raise ValueError(f"Unsupported interval: {interval}")

    interval_ms = interval_map[interval]

    # 2. Parse Timestamps
    # If start_str is None, default to 30 days ago (API max retention)
    now_ms = int(time.time() * 1000)

    if start_str:
        start_ts = int(pd.to_datetime(start_str, utc=True).timestamp() * 1000)
    else:
        start_ts = now_ms - (30 * 24 * 60 * 60 * 1000)

    if end_str:
        end_ts = int(pd.to_datetime(end_str, utc=True).timestamp() * 1000)
    else:
        end_ts = now_ms

    # 3. Dynamic Chunk Calculation
    # We request (limit * interval) duration to maximize throughput without hitting the limit
    # We subtract 1 interval to be safe against boundary overlaps
    chunk_duration_ms = (limit * interval_ms)

    all_data = []
    current_start = start_ts

    print(f"Fetching {symbol} [{interval}] from {start_ts} to {end_ts}")

    while current_start < end_ts:
        # Calculate end of this chunk
        current_end = min(current_start + chunk_duration_ms, end_ts)

        params = {
            "symbol": symbol,
            "period": interval,
            "limit": limit,
            "startTime": current_start,
            "endTime": current_end,
        }

        try:
            resp = requests.get(url, params=params, timeout=10)

            # Handle the 400 error gracefully
            if resp.status_code == 400:
                print(
                    f"Skipping chunk {current_start}-{current_end}: Data likely too old (Max 30 days).")
                current_start = current_end
                continue

            resp.raise_for_status()
            data = resp.json()

            if isinstance(data, list) and data:
                all_data.extend(data)

                # OPTIMIZATION: Update current_start based on the *actual* last data point received
                # This handles gaps in data (e.g., maintenance) gracefully
                last_data_ts = data[-1]['timestamp']
                current_start = last_data_ts + interval_ms
            else:
                # If no data returned (empty list), move pointer forward
                current_start = current_end

        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            break

        time.sleep(0.1)

    df = pd.DataFrame(all_data)
    if df.empty:
        return pd.DataFrame()

    df["timestamp"] = df["timestamp"].astype("int64")
    df["ts"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["ls_ratio"] = df["longShortRatio"].astype(float)

    # Dedup and Sort
    df = df.drop_duplicates(subset=["timestamp"]).sort_values(
        "timestamp").reset_index(drop=True)

    return df[["ts", "ls_ratio"]]
