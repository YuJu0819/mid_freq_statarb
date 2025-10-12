import asyncio
import json
import pandas as pd
from typing import AsyncGenerator
from binance import AsyncClient, BinanceSocketManager
from ..core.logger import get_logger

logger = get_logger("binance_ws")


async def stream_klines(symbol: str, interval: str, use_testnet: bool = True) -> AsyncGenerator[pd.DataFrame, None]:
    client = await AsyncClient.create(testnet=use_testnet)
    bm = BinanceSocketManager(client)
    # kline socket
    async with bm.kline_socket(symbol=symbol, interval=interval) as stream:
        # V-- CHANGE IS HERE --V
        # The library now requires calling recv() in a loop instead of using 'async for'
        while True:
            msg = await stream.recv()
            # ^-- CHANGE IS HERE --^

            if msg.get("e") != "kline":
                continue
            k = msg["k"]
            # we emit only when bar is closed to avoid repaint
            if not k["x"]:
                continue
            row = {
                "ts": int(k["t"]),
                "open": float(k["o"]),
                "high": float(k["h"]),
                "low": float(k["l"]),
                "close": float(k["c"]),
                "volume": float(k["v"]),
            }
            yield pd.DataFrame([row])

    await client.close_connection()
