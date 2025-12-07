import os
import pandas as pd
import time
from binance.client import Client
from binance.exceptions import BinanceAPIException
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("BINANCE_API_KEY")
api_secret = os.getenv("BINANCE_API_SECRET")
use_testnet = os.getenv("BINANCE_USE_TESTNET", "false").lower() == "true"

client = Client(api_key, api_secret, testnet=use_testnet)

# --- V-- NEW: RATE LIMIT WRAPPER --V ---


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
    # (Same as before, simplified for brevity - assumes start_date logic we added)
    try:
        start_ts = int(pd.to_datetime(start_date).timestamp()
                       * 1000) if start_date else None
        end_ts = int(pd.to_datetime(end_date).timestamp()
                     * 1000) if end_date else None

        oi_data = safe_api_call(
            client.futures_open_interest_hist,
            symbol=symbol,
            period=interval,
            limit=limit,
            startTime=start_ts,
            endTime=end_ts
        )

        if not oi_data:
            return pd.DataFrame()

        df = pd.DataFrame(oi_data)
        df["sumOpenInterestValue"] = pd.to_numeric(
            df["sumOpenInterestValue"], errors="coerce")
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")

        df.rename(columns={"sumOpenInterestValue": "open_interest",
                  "timestamp": "ts"}, inplace=True)
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
