import argparse
import asyncio
import os
from ..core.utils import load_config
from ..strategy.sma_cross import SMACross
from ..live.trader import LiveContext, run_live


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--interval", default="1m")
    args = ap.parse_args()

    cfg = load_config()

    # V-- CHANGE IS HERE --V
    # Pass the config and specify 'live' mode
    strat = SMACross(fast=10, slow=30, cfg=cfg, mode="live")
    # ^-- CHANGE IS HERE --^

    ctx = LiveContext(symbol=args.symbol,
                      interval=args.interval, strategy=strat, cfg=cfg)
    asyncio.run(run_live(ctx))


if __name__ == "__main__":
    main()
