import os
import math
import time
from typing import Optional, Dict, Tuple
from binance.client import Client
from ..core.types import Order, Fill
from ..core.logger import get_logger

logger = get_logger("binance_broker")


def client_from_env() -> Client:
    key = os.getenv("BINANCE_API_KEY")
    secret = os.getenv("BINANCE_API_SECRET")
    use_testnet = os.getenv("BINANCE_USE_TESTNET", "true").lower() == "true"
    return Client(api_key=key, api_secret=secret, testnet=use_testnet)


class BinanceBroker:
    """Spot broker wrapper with:
    - market order execution
    - lot size / tick size rounding
    - balances / equity in USDT
    - position reconciliation helpers
    """

    def __init__(self):
        self.client = client_from_env()
        self._symbol_info_cache: Dict[str, dict] = {}
        # e.g., BTCUSDT -> 60234.1
        self._last_price_cache: Dict[str, float] = {}
        # e.g., BTC -> 60234.1 (USDT terms)
        self._last_asset_price_cache: Dict[str, float] = {}

    # ---------- Symbol / filters ----------
    def get_symbol_info(self, symbol: str) -> dict:
        if symbol not in self._symbol_info_cache:
            self._symbol_info_cache[symbol] = self.client.get_symbol_info(
                symbol)
        return self._symbol_info_cache[symbol]

    def lot_size(self, symbol: str) -> float:
        ex = self.get_symbol_info(symbol)
        for f in ex["filters"]:
            if f["filterType"] == "LOT_SIZE":
                return float(f["stepSize"])
        return 1e-6

    def price_tick(self, symbol: str) -> float:
        ex = self.get_symbol_info(symbol)
        for f in ex["filters"]:
            if f["filterType"] == "PRICE_FILTER":
                return float(f["tickSize"])
        return 1e-6

    def round_qty(self, symbol: str, qty: float) -> float:
        step = self.lot_size(symbol)
        if step <= 0:
            return qty
        return math.floor(qty / step) * step

    # ---------- Prices ----------
    def update_last_price(self, symbol: str, price: float):
        self._last_price_cache[symbol] = float(price)
        base, quote = symbol[:-4], symbol[-4:]  # naive, assumes *USDT
        if quote == "USDT":
            self._last_asset_price_cache[base] = float(price)

    def get_price(self, symbol: str) -> float:
        # Prefer cache, fall back to REST
        if symbol in self._last_price_cache:
            return self._last_price_cache[symbol]
        px = float(self.client.get_symbol_ticker(symbol=symbol)["price"])
        self.update_last_price(symbol, px)
        return px

    def get_asset_usdt_price(self, asset: str) -> float:
        if asset == "USDT":
            return 1.0
        if asset in self._last_asset_price_cache:
            return self._last_asset_price_cache[asset]
        sym = f"{asset}USDT"
        try:
            px = float(self.client.get_symbol_ticker(symbol=sym)["price"])
            self.update_last_price(sym, px)
            return px
        except Exception as e:
            logger.warning(
                f"No direct {asset}USDT price; treating as 0 in equity. err={e}")
            return 0.0

    # ---------- Balances / equity ----------
    def get_balances(self) -> Dict[str, float]:
        """Return free balances (spot)."""
        acct = self.client.get_account()
        out = {}
        for b in acct.get("balances", []):
            free = float(b.get("free", 0.0))
            if free > 0:
                out[b["asset"]] = free
        return out

    def equity_usdt(self) -> Tuple[float, Dict[str, float]]:
        """Compute total equity in USDT using cached/rest prices. Returns (equity, balances)."""
        bals = self.get_balances()
        eq = 0.0
        for asset, qty in bals.items():
            px = self.get_asset_usdt_price(asset)
            eq += qty * px
        return eq, bals

    # ---------- Position (spot) ----------
    def position_qty_spot(self, symbol: str, balances: Optional[Dict[str, float]] = None) -> float:
        """For spot, position in {BASE}{QUOTE} is simply how much BASE asset you hold."""
        base, quote = symbol[:-4], symbol[-4:]  # assumes *USDT symbol naming
        bals = balances if balances is not None else self.get_balances()
        return float(bals.get(base, 0.0))

    # ---------- Orders ----------
    def market_order(self, order: Order) -> Optional[Fill]:
        symbol = order["symbol"]
        side = order["side"]

        # For spot, you can't short more than you have.
        # This logic ensures we don't try to sell more of an asset than is available.
        if side == "SELL":
            current_qty = self.position_qty_spot(symbol)
            if float(order["qty"]) > current_qty:
                logger.warning(
                    f"SELL order for {order['qty']} {symbol} exceeds available balance of {current_qty}. Adjusting to sell all.")
                order["qty"] = current_qty

        qty = self.round_qty(symbol, float(order["qty"]))
        if qty <= 0:
            logger.warning(
                f"Order for {symbol} has quantity 0 after rounding/adjustment; skipping.")
            return None

        # Execute the order using the Binance API
        resp = self.client.create_order(
            symbol=symbol, side=side, type="MARKET", quantity=qty)

        # Process the fill information
        fills = resp.get("fills", [])
        if fills:
            # Calculate the average fill price
            total_qty = sum(float(f["qty"]) for f in fills)
            total_notional = sum(
                float(f["price"]) * float(f["qty"]) for f in fills)
            px = total_notional / max(1e-12, total_qty)
        else:
            # Fallback to the last known price if fill data isn't available
            px = self.get_price(symbol)

        logger.info(
            f"(LIVE) Executed MARKET {side} {qty} {symbol} @ avg_px={px}")
        return {"symbol": symbol, "qty": qty, "price": px, "ts": resp["transactTime"]}
