import os
import pandas as pd
from binance.client import Client
from ..core.logger import get_logger

logger = get_logger("get_symbols")


def get_top_symbols_by_volume(top_n: int = 100) -> list[str]:
    """
    Fetches all symbols from Binance, filters for USDT pairs, and returns
    the top N symbols ranked by their 24-hour trading volume.
    """
    # Use the real Binance API for market data, not the testnet
    client = Client(
        api_key=os.getenv("BINANCE_API_KEY"),
        api_secret=os.getenv("BINANCE_API_SECRET"),
        testnet=False
    )

    logger.info("Fetching 24-hour ticker data for all symbols...")
    # 1. Fetch data for all tickers
    all_tickers = client.get_ticker()

    # 2. Filter for USDT pairs and process the data
    usdt_pairs = []
    for ticker in all_tickers:
        symbol = ticker['symbol']
        # We only want pairs that are quoted in USDT
        if not symbol.endswith("USDT"):
            continue
        # Exclude leveraged UP/DOWN tokens and stablecoin pairs
        if any(x in symbol for x in ["UP", "DOWN", "USDC"]):
            continue

        usdt_pairs.append({
            "symbol": symbol,
            "volume": float(ticker["quoteVolume"])
        })

    if not usdt_pairs:
        logger.error("No USDT pairs found. Check API connection.")
        return []

    # 3. Create a DataFrame, sort by volume, and get the top N
    df = pd.DataFrame(usdt_pairs)
    df = df.sort_values(by="volume", ascending=False)
    top_symbols = df.head(top_n)["symbol"].tolist()

    logger.info(f"Found {len(top_symbols)} top symbols by 24h volume.")
    return top_symbols


def main():
    top_100_symbols = get_top_symbols_by_volume(100)

    if top_100_symbols:
        print("\n--- Top 100 Symbols by 24h Trading Volume (USDT) ---")
        # Print in a format that's easy to copy into config.yaml
        print("symbols: [", end="")
        for i, symbol in enumerate(top_100_symbols):
            if i % 10 == 0:
                print("\n  ", end="")
            print(f'"{symbol}", ', end="")
        print("\n]")


if __name__ == "__main__":
    main()
