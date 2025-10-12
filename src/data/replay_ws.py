import pandas as pd
import asyncio
from typing import AsyncGenerator


async def stream_replay_klines(path: str) -> AsyncGenerator[pd.DataFrame, None]:
    """
    Reads a historical Parquet file and yields each row as a single-row DataFrame,
    simulating a live kline WebSocket stream.
    """
    try:
        df = pd.read_parquet(path)
        df = df.sort_values("ts").reset_index(drop=True)
    except FileNotFoundError:
        print(f"Error: Data file not found at {path}")
        print("Please run a backtest first to generate the historical data file.")
        return

    for i in range(len(df)):
        # Yield a DataFrame with a single row to match the WebSocket stream's format
        yield df.iloc[i:i+1]
        # Allow other async tasks to run, preventing the event loop from blocking
        await asyncio.sleep(0)
