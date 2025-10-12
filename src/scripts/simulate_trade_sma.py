import argparse
import asyncio
import os
import pandas as pd
from dataclasses import dataclass

from ..core.utils import load_config
from ..data.storage import parquet_path
# <-- Import the new replay streamer
from ..data.replay_ws import stream_replay_klines
from ..strategy.sma_cross import SMACross
from ..portfolio.paperbroker import PaperBroker
from ..portfolio.risk import risk_checks
from ..core.types import Order
from ..core.logger import get_logger

logger = get_logger("simulation")


@dataclass
class SimulationContext:
    """ A context object to hold all simulation parameters and objects. """
    symbol: str
    interval: str
    strategy: any
    cfg: dict
    broker: PaperBroker


async def run_simulation(ctx: SimulationContext):
    """
    Runs the main simulation loop, feeding replayed historical data to the
    live trading logic.
    """
    # This DataFrame will hold the rolling window of data for the strategy
    df = pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])
    trades = []

    # Construct the path to the historical data file
    data_path = parquet_path(
        ctx.cfg["general"]["parquet_dir"], ctx.symbol, ctx.interval)
    logger.info(f"Starting simulation from data file: {data_path}")

    # Use the replay kline streamer
    async for new_df in stream_replay_klines(data_path):
        # Append new bar and maintain a rolling window of data
        df = pd.concat([df, new_df], ignore_index=True).drop_duplicates(
            "ts").sort_values("ts")
        last_price = float(df["close"].iloc[-1])

        # 1. Mark-to-market the portfolio with the latest price
        snap = ctx.broker.mark_to_market({ctx.symbol: last_price})

        # 2. Get a signal from the strategy
        sig = ctx.strategy.on_bar(ctx.symbol, ctx.interval, df)
        if sig is None:
            continue

        # 3. Size the order based on the signal and portfolio equity
        target_weight = max(0.0, min(1.0, float(sig.weight)))
        notional = target_weight * snap["equity"]
        current_qty = ctx.broker.positions.get(ctx.symbol, {"qty": 0.0})["qty"]
        target_qty = notional / last_price
        delta = target_qty - current_qty

        if abs(delta) < 1e-9:
            continue

        # 4. Perform risk checks
        r_events = risk_checks(snap, ctx.symbol, last_price, target_weight,
                               # Use backtest config for sim
                               ctx.cfg["backtest"]["max_position_notional"])
        if r_events:
            logger.info(
                f"Risk blocked at ts={int(df['ts'].iloc[-1])}: {[r.name for r in r_events]}")
            continue

        # 5. Execute the order on the PaperBroker
        side = "BUY" if delta > 0 else "SELL"
        order: Order = {"symbol": ctx.symbol, "side": side, "qty": abs(delta), "order_type": "MARKET",
                        "price": None, "tif": None}
        fill = ctx.broker.execute(order, last_price)
        trades.append({"ts": fill["ts"], "symbol": ctx.symbol,
                      "side": side, "qty": fill["qty"], "price": fill["price"]})
        logger.info(
            f"(SIM) Executed {side} {abs(delta):.6f} {ctx.symbol} at ~{last_price:.2f}")

    # --- Print Summary ---
    final_snap = ctx.broker.mark_to_market(
        {ctx.symbol: float(df["close"].iloc[-1])})
    initial_cash = ctx.cfg["backtest"]["initial_cash"]
    final_equity = final_snap["equity"]
    pnl_pct = (final_equity / initial_cash - 1) * 100

    print("\n==== Simulation Summary ====")
    print(f"Initial Cash: {initial_cash:.2f}")
    print(f"Final Equity: {final_equity:.2f}")
    print(f"Total Return: {pnl_pct:.2f}%")
    print(f"Total Trades: {len(trades)}")
    print("==========================")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--interval", default="1h")
    args = ap.parse_args()

    cfg = load_config()

    # V-- CHANGE IS HERE --V
    # Pass the config and specify the mode for simulation.
    # We use 'backtest' mode to ensure the simulation and backtest are 1-to-1 comparable.
    strat = SMACross(fast=10, slow=30, cfg=cfg, mode="backtest")
    # ^-- CHANGE IS HERE --^

    broker = PaperBroker(
        fee_bps=cfg["backtest"]["fee_bps"],
        slippage_bps=cfg["backtest"]["slippage_bps"],
        cash=cfg["backtest"]["initial_cash"]
    )

    ctx = SimulationContext(
        symbol=args.symbol, interval=args.interval, strategy=strat, cfg=cfg, broker=broker)
    asyncio.run(run_simulation(ctx))


if __name__ == "__main__":
    main()
