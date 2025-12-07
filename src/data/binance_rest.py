import os
import pandas as pd
import time
from binance.client import Client
from binance.exceptions import BinanceAPIException
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("BINANCE_API_KEY")
api_secret = os.getenv("BINANCE_API_SECRET")
client = Client(api_key, api_secret)


def safe_api_call(func, *args, **kwargs):
    MAX_RETRIES = 3
    for attempt in range(MAX_RETRIES):
        try:
            return func(*args, **kwargs)
        except BinanceAPIException as e:
            if e.code == -1003:
                print(f"   [RATE LIMIT HIT] Cooling down for 60 seconds...")
                time.sleep(60)
                continue
            raise e
    return None


def fetch_klines(symbol: str, interval: str, start_date: str, end_date: str, limit: int = 1000) -> pd.DataFrame:
    try:
        start_ts = int(pd.to_datetime(start_date).timestamp() * 1000)
        end_ts = int(pd.to_datetime(end_date).timestamp() * 1000)

        all_klines = []
        current_start = start_ts

        while True:
            klines = safe_api_call(
                client.get_klines,
                symbol=symbol,
                interval=interval,
                startTime=current_start,
                endTime=end_ts,
                limit=limit
            )

            if not klines:
                break

            all_klines.extend(klines)
            current_start = klines[-1][6] + 1

            if len(klines) < limit or current_start > end_ts:
                break
            time.sleep(0.2)

        if not all_klines:
            return pd.DataFrame()

        # ... (Same column processing as before) ...
        df = pd.DataFrame(all_klines, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
        ])

        df["ts"] = pd.to_numeric(df["open_time"])
        df["close"] = pd.to_numeric(df["close"])
        df["volume"] = pd.to_numeric(df["volume"])
        # We also need High/Low for some calculations if used
        df["high"] = pd.to_numeric(df["high"])
        df["low"] = pd.to_numeric(df["low"])

        # Return relevant cols
        return df[["ts", "close", "volume", "high", "low"]]

    except BinanceAPIException as e:
        # print(f"Error fetching spot klines for {symbol}: {e}")
        return pd.DataFrame()
