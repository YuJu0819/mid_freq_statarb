import time
import pandas as pd
from binance.exceptions import BinanceAPIException
from .binance_futures_rest import client, safe_api_call


def fetch_klines(
    symbol: str,
    interval: str,
    start_date: str,
    end_date: str,
    limit: int = 1000,
) -> pd.DataFrame:
    """Fetch spot market OHLCV from Binance with pagination."""
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
                limit=limit,
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

        df = pd.DataFrame(
            all_klines,
            columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_asset_volume", "number_of_trades",
                "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore",
            ],
        )

        df["ts"] = pd.to_numeric(df["open_time"])
        df["close"] = pd.to_numeric(df["close"])
        df["volume"] = pd.to_numeric(df["volume"])
        df["high"] = pd.to_numeric(df["high"])
        df["low"] = pd.to_numeric(df["low"])

        return df[["ts", "close", "volume", "high", "low"]]

    except BinanceAPIException:
        return pd.DataFrame()
