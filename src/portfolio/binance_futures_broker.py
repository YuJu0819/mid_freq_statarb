import os
import math
from binance.client import Client
from ..core.types import Order, Fill
from ..core.logger import get_logger
from .binance_broker import client_from_env  # We can reuse the client creation

logger = get_logger("binance_futures_broker")


class BinanceFuturesBroker:
    """ A broker specifically for trading USDT-margined futures. """

    def __init__(self):
        self.client = client_from_env()
        self._symbol_info_cache = {}

    def get_symbol_info(self, symbol: str) -> dict:
        if not self._symbol_info_cache:
            info = self.client.futures_exchange_info()
            for item in info['symbols']:
                self._symbol_info_cache[item['symbol']] = item
        return self._symbol_info_cache.get(symbol)

    def round_qty(self, symbol: str, qty: float) -> float:
        info = self.get_symbol_info(symbol)
        for f in info['filters']:
            if f['filterType'] == 'LOT_SIZE':
                step = float(f['stepSize'])
                precision = int(round(-math.log(step, 10), 0))
                return round(qty, precision)
        return qty

    def set_leverage(self, symbol: str, leverage: int):
        """ Sets the leverage for a given symbol. """
        try:
            self.client.futures_change_leverage(
                symbol=symbol, leverage=leverage)
            logger.info(f"Set leverage for {symbol} to {leverage}x")
        except Exception as e:
            logger.error(f"Failed to set leverage for {symbol}: {e}")

    def get_equity_usdt(self) -> float:
        """ Gets the total collateral value of the futures account in USDT. """
        account_info = self.client.futures_account()
        return float(account_info['totalWalletBalance'])

    def get_position_size(self, symbol: str) -> float:
        """ Gets the current position size for a single symbol. """
        positions = self.client.futures_position_information()
        for pos in positions:
            if pos['symbol'] == symbol:
                return float(pos['positionAmt'])
        return 0.0

    def market_order(self, order: Order) -> Fill:
        """ Places a market order on the futures market. """
        symbol = order["symbol"]
        side = order["side"]
        qty = self.round_qty(symbol, abs(float(order["qty"])))

        if qty <= 0:
            logger.warning(
                f"Order for {symbol} has quantity 0 after rounding; skipping.")
            return None

        try:
            resp = self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type="MARKET",
                quantity=qty
            )
            # Fetch the order to get the average fill price
            order_info = self.client.futures_get_order(
                symbol=symbol, orderId=resp['orderId'])
            avg_price = float(order_info['avgPrice'])

            logger.info(
                f"(LIVE FUTURES) Executed MARKET {side} {qty} {symbol} @ avg_px={avg_price}")
            return {"symbol": symbol, "qty": qty, "price": avg_price, "ts": order_info["updateTime"]}
        except Exception as e:
            logger.error(f"Failed to execute futures order for {symbol}: {e}")
            return None
