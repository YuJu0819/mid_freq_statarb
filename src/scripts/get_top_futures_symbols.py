import os
import pandas as pd
from binance.client import Client
from ..core.logger import get_logger

logger = get_logger("get_futures_symbols")


def get_top_futures_symbols(top_n: int = 100) -> list[str]:
    """
    Identifies all USDT-margined futures contracts, ranks them by 24h volume,
    and returns the top N symbols. This ensures every symbol has a futures market.
    """
    client = Client(
        api_key=os.getenv("BINANCE_API_KEY"),
        api_secret=os.getenv("BINANCE_API_SECRET"),
        testnet=False
    )

    logger.info("Fetching all futures symbols from exchange info...")
    # --- 1. Get all symbols that have a futures market ---
    exchange_info = client.futures_exchange_info()
    futures_symbols = {
        item['symbol'] for item in exchange_info['symbols']
        if item['quoteAsset'] == 'USDT' and item['contractType'] == 'PERPETUAL'
    }

    if not futures_symbols:
        logger.error("Could not fetch any futures symbols.")
        return []

    logger.info(
        f"Found {len(futures_symbols)} USDT perpetual futures contracts. Fetching tickers...")

    # --- 2. Fetch 24-hour ticker data for ONLY the futures symbols ---
    all_tickers = client.futures_ticker()

    futures_tickers = []
    for ticker in all_tickers:
        symbol = ticker['symbol']
        if symbol in futures_symbols:
            # Exclude leveraged UP/DOWN tokens which are not suitable for this strategy
            if "UP" in symbol or "DOWN" in symbol:
                continue

            futures_tickers.append({
                "symbol": symbol,
                "volume": float(ticker["quoteVolume"])
            })

    # --- 3. Rank by volume and select the top N ---
    df = pd.DataFrame(futures_tickers)
    df = df.sort_values(by="volume", ascending=False)
    top_symbols = df.head(top_n)["symbol"].tolist()

    logger.info(
        f"Selected top {len(top_symbols)} futures symbols by 24h volume.")
    return top_symbols


def main():
    K = 150
    top_100_symbols = get_top_futures_symbols(K)

    if top_100_symbols:
        print(
            f"\n--- Top {K} Futures Symbols by 24h Trading Volume (USDT) ---")
        # Print in a format that's easy to copy into config.yaml
        print("symbols: [", end="")
        for i, symbol in enumerate(top_100_symbols):
            if i % 10 == 0:
                print("\n  ", end="")
            print(f'"{symbol}", ', end="")
        print("\n]")


if __name__ == "__main__":
    main()
