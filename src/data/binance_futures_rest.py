import os
import pandas as pd
from binance.client import Client
from binance.exceptions import BinanceAPIException
from dotenv import load_dotenv

load_dotenv()

# --- FIX: Initialize the client properly ---
api_key = os.getenv("BINANCE_API_KEY")
api_secret = os.getenv("BINANCE_API_SECRET")
use_testnet = os.getenv("BINANCE_USE_TESTNET", "false").lower() == "true"

client = Client(api_key, api_secret, testnet=use_testnet)


def fetch_open_interest(symbol: str, interval: str = "1d", limit: int = 500) -> pd.DataFrame:
    """
    Fetches historical open interest for a given symbol.
    """
    try:
        oi_data = client.futures_open_interest_hist(
            symbol=symbol, period=interval, limit=limit
        )

        if not oi_data:
            return pd.DataFrame()

        df = pd.DataFrame(oi_data)
        df["sumOpenInterestValue"] = pd.to_numeric(
            df["sumOpenInterestValue"], errors="coerce"
        )
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")

        df.rename(
            columns={"sumOpenInterestValue": "open_interest", "timestamp": "ts"},
            inplace=True,
        )

        df = df[["ts", "open_interest"]]
        # print(df)
        return df

    except BinanceAPIException as e:
        print(f"Error fetching open interest for {symbol}: {e}")
        return pd.DataFrame()


def fetch_funding_rate(symbol: str, limit: int = 500) -> pd.DataFrame:
    """
    Fetches historical funding rates.
    """
    try:
        fr_data = client.futures_funding_rate(symbol=symbol, limit=limit)

        if not fr_data:
            return pd.DataFrame()

        df = pd.DataFrame(fr_data)
        df["fundingTime"] = pd.to_numeric(df["fundingTime"], errors="coerce")
        df["fundingRate"] = pd.to_numeric(df["fundingRate"], errors="coerce")

        df.rename(
            columns={"fundingTime": "ts", "fundingRate": "funding_rate"},
            inplace=True,
        )

        return df[["ts", "funding_rate"]]

    except BinanceAPIException as e:
        print(f"Error fetching funding rate for {symbol}: {e}")
        return pd.DataFrame()
