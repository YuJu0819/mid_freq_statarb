from dataclasses import dataclass
from ..core.types import Order, Fill, PortfolioSnapshot


@dataclass
class PaperBroker:
    """ A paper broker that supports both long and short positions. """
    fee_bps: float = 10.0
    slippage_bps: float = 5.0
    cash: float = 100000.0
    positions: dict = None

    def __post_init__(self):
        self.positions = {} if self.positions is None else self.positions

    def mark_to_market(self, price_map: dict[str, float]) -> PortfolioSnapshot:
        equity = self.cash
        pos_out = {}
        for sym, pos in self.positions.items():
            if sym in price_map:
                notional = pos["qty"] * price_map[sym]
                equity += notional
                pos_out[sym] = pos
        return {"equity": equity, "cash": self.cash, "positions": pos_out}

    def _apply_fee_slip(self, side: str, px: float):
        slip = px * (self.slippage_bps / 1e4)
        eff_px = px + slip if side == "BUY" else px - slip
        fee_rate = self.fee_bps / 1e4
        return eff_px, fee_rate

    def execute(self, order: Order, last_price: float) -> Fill:
        qty = float(order["qty"])
        side = order["side"]
        symbol = order["symbol"]
        eff_px, fee_rate = self._apply_fee_slip(side, last_price)
        notional = eff_px * qty
        fee = abs(notional) * fee_rate

        pos = self.positions.get(
            symbol, {"symbol": symbol, "qty": 0.0, "avg_price": 0.0})
        current_qty = pos["qty"]

        if side == "BUY":
            # Buying to open a new long or cover a short
            self.cash -= notional + fee
            new_qty = current_qty + qty
            if new_qty > current_qty and current_qty >= 0:  # Averaging up a long
                pos["avg_price"] = (
                    pos["avg_price"]*current_qty + eff_px*qty) / new_qty
            pos["qty"] = new_qty
        else:  # SELL
            # Selling to close a long or open a new short
            self.cash += notional - fee
            pos["qty"] = current_qty - qty

        # If position is closed, reset avg_price
        if abs(pos["qty"]) < 1e-12:
            pos["qty"] = 0.0
            pos["avg_price"] = 0.0

        self.positions[symbol] = pos
        return {"symbol": symbol, "qty": qty, "price": eff_px}
