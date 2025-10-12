import os
import pandas as pd
from binance.client import Client
from binance.exceptions import BinanceAPIException
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("BINANCE_API_KEY")
api_secret = os.getenv("BINANCE_API_SECRET")
use_testnet = os.getenv("BINANCE_USE_TESTNET", "false").lower() == "true"

client = Client(api_key, api_secret, testnet=use_testnet)


def fetch_klines(symbol: str, interval: str, start_date: str, end_date: str, use_testnet: bool = False) -> pd.DataFrame:
    try:
        klines = client.get_historical_klines(
            symbol, interval, start_str=start_date, end_str=end_date
        )
        if not klines:
            return pd.DataFrame()

        df = pd.DataFrame(klines, columns=[
            "ts", "open", "high", "low", "close", "volume", "close_time",
            "quote_asset_volume", "number_of_trades", "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume", "ignore"
        ])

        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        return df
    except BinanceAPIException as e:
        print(f"Error fetching klines for {symbol}: {e}")
        return pd.DataFrame()


def fetch_futures_klines(symbol: str, interval: str, start_date: str, end_date: str, use_testnet: bool = False) -> pd.DataFrame:
    try:
        klines = client.futures_historical_klines(
            symbol, interval, start_str=start_date, end_str=end_date
        )
        if not klines:
            return pd.DataFrame()

        df = pd.DataFrame(klines, columns=[
            "ts", "open", "high", "low", "close", "volume", "close_time",
            "quote_asset_volume", "number_of_trades", "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume", "ignore"
        ])

        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        return df
    except BinanceAPIException as e:
        print(f"Error fetching futures klines for {symbol}: {e}")
        return pd.DataFrame()
