import asyncio
import os
import pandas as pd
from dataclasses import dataclass
from ..data.binance_ws import stream_klines
from ..portfolio.binance_broker import BinanceBroker
from ..portfolio.paperbroker import PaperBroker
from ..core.logger import get_logger
from ..core.types import Order
from ..portfolio.risk import risk_checks

logger = get_logger("live")


@dataclass
class LiveContext:
    symbol: str
    interval: str
    strategy: any
    cfg: dict
    use_testnet: bool = True


async def run_live(ctx: LiveContext):

    # Decide broker by env (BINANCE_USE_TESTNET)
    use_testnet = os.getenv("BINANCE_USE_TESTNET", "true").lower() == "true"
    if use_testnet:
        logger.info("Using PaperBroker (testnet mode ON).")
        broker = PaperBroker(
            fee_bps=ctx.cfg["backtest"]["fee_bps"],
            slippage_bps=ctx.cfg["backtest"]["slippage_bps"],
            cash=ctx.cfg["backtest"]["initial_cash"]
        )
    else:
        logger.info(
            "Using BinanceBroker (LIVE). MAKE SURE YOU UNDERSTAND THE RISKS.")
        broker = BinanceBroker()

    df = pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])
    required_rows = ctx.strategy.slow + 5  # Add a small buffer
    if len(df) > required_rows:
        df = df.iloc[-required_rows:]
    async for new_df in stream_klines(ctx.symbol, ctx.interval, use_testnet=use_testnet):
        df = pd.concat([df, new_df], ignore_index=True).drop_duplicates(
            "ts").sort_values("ts")
        last_price = float(df["close"].iloc[-1])

        # Hint the broker about last price (for tick caches & faster pricing)
        if isinstance(broker, BinanceBroker):
            broker.update_last_price(ctx.symbol, last_price)

        sig = ctx.strategy.on_bar(ctx.symbol, ctx.interval, df)
        if sig is None:
            continue
        target_weight = max(0.0, min(1.0, float(sig.weight)))

        if isinstance(broker, PaperBroker):
            # Paper sizing: use snapshot equity
            snap = broker.mark_to_market({ctx.symbol: last_price})
            notional = target_weight * snap["equity"]
            current_qty = broker.positions.get(ctx.symbol, {"qty": 0.0})["qty"]
            target_qty = notional / last_price
            delta = target_qty - current_qty
            if abs(delta) < 1e-9:
                continue
            r_events = risk_checks(snap, ctx.symbol, last_price, target_weight,
                                   ctx.cfg["live"]["max_position_notional"])
            if r_events:
                logger.info(f"Risk blocked: {[r.name for r in r_events]}")
                continue
            side = "BUY" if delta > 0 else "SELL"
            order: Order = {"symbol": ctx.symbol, "side": side, "qty": abs(delta), "order_type": "MARKET",
                            "price": None, "tif": None}
            broker.execute(order, last_price)
            logger.info(
                f"(Paper) Executed {side} {abs(delta)} {ctx.symbol} at ~{last_price}")
        else:
            # LIVE sizing: compute equity & current position from account
            try:
                equity, balances = broker.equity_usdt()  # total portfolio equity in USDT
                current_qty = broker.position_qty_spot(
                    ctx.symbol, balances=balances)  # base asset qty
                target_notional = target_weight * equity
                target_qty = target_notional / max(1e-12, last_price)
                delta = target_qty - current_qty
                if abs(delta) <= 1e-12:
                    continue

                # Basic risk: cap notional
                # minimal shim for risk function
                snap_like = {"equity": equity}
                r_events = risk_checks(snap_like, ctx.symbol, last_price, target_weight,
                                       ctx.cfg["live"]["max_position_notional"])
                if r_events:
                    logger.info(f"Risk blocked: {[r.name for r in r_events]}")
                    continue

                side = "BUY" if delta > 0 else "SELL"
                order: Order = {"symbol": ctx.symbol, "side": side, "qty": abs(delta), "order_type": "MARKET",
                                "price": None, "tif": None}
                fill = broker.market_order(order)
                if fill:
                    logger.info(
                        f"(LIVE) Executed {side} {fill['qty']} {ctx.symbol} avg_px={fill['price']}")
            except Exception as e:
                logger.error(f"Live sizing/execute error: {e}")
