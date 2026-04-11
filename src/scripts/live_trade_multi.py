import asyncio
import pandas as pd
from datetime import datetime, timezone
from ..core.utils import load_config
from ..data.binance_rest import fetch_klines
# <-- Import the new broker
from ..portfolio.binance_futures_broker import BinanceFuturesBroker
from ..strategy.ad_mom_spot_future import FinalStrategy
from ..core.logger import get_logger

logger = get_logger("live_multi_futures")


async def run_live_multi_futures(cfg: dict, strategy):
    broker = BinanceFuturesBroker()
    symbols = cfg["backtest"]["symbols"]
    leverage = cfg["live"]["leverage"]
    rebalance_period = pd.to_timedelta(cfg["backtest"]["rebalance_period"])
    last_rebalance_ts = pd.Timestamp(0, tz='UTC')

    logger.info(f"Initializing live futures trader for universe: {symbols}")
    logger.info("Setting leverage for all symbols...")
    for symbol in symbols:
        await asyncio.to_thread(broker.set_leverage, symbol, leverage)

    while True:
        now = datetime.now(timezone.utc)

        if now >= last_rebalance_ts + rebalance_period:
            logger.info("Rebalance period triggered. Evaluating strategy...")

            try:
                # --- 1. Fetch data & get signals ---
                lookback_days = strategy.lookback + 5
                strategy_data = {}
                for symbol in symbols:
                    df = fetch_klines(
                        symbol, "1d", lookback_days, use_testnet=False)
                    if not df.empty:
                        strategy_data[symbol] = df

                if not strategy_data:
                    logger.warning("Could not fetch data. Skipping rebalance.")
                    await asyncio.sleep(60)
                    continue

                signals = strategy.on_rebalance(strategy_data)

                # --- 2. Get current portfolio state ---
                equity = await asyncio.to_thread(broker.get_equity_usdt)
                logger.info(
                    f"Current portfolio equity (collateral): ${equity:.2f}")

                # --- 3. Execute rebalancing trades ---
                for symbol in symbols:
                    target_weight = signals.get(symbol, {"weight": 0.0}).weight
                    last_price = strategy_data.get(symbol)["close"].iloc[-1]

                    # Calculate target position size in base asset
                    target_position_size = (
                        equity * leverage * target_weight) / last_price

                    # Get current position size from futures
                    current_position_size = await asyncio.to_thread(broker.get_position_size, symbol)

                    delta_qty = target_position_size - current_position_size

                    if abs(delta_qty) < float(broker.get_symbol_info(symbol)['filters'][1]['stepSize']):
                        continue

                    side = "BUY" if delta_qty > 0 else "SELL"
                    logger.info(
                        f"Rebalancing {symbol}: Target Size={target_position_size:.4f}, Current Size={current_position_size:.4f}, Action: {side} {abs(delta_qty):.4f}")
                    await asyncio.to_thread(broker.market_order, {"symbol": symbol, "side": side, "qty": abs(delta_qty)})

                last_rebalance_ts = pd.Timestamp(now)
                logger.info("Rebalance complete.")

            except Exception as e:
                logger.error(f"An error occurred during rebalance: {e}")

        await asyncio.sleep(60)


def main():
    cfg = load_config()
    strategy = FinalStrategy(
        lookback=90,
        quantile=0.2,
        min_volume_usd=10_000_000,
        funding_lookback=14,
        funding_z_threshold=0.0025
    )
    asyncio.run(run_live_multi_futures(cfg, strategy))


if __name__ == "__main__":
    main()
