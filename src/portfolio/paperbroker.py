import pandas as pd
from ..core.types import Order, Fill, BacktestResult


class PaperBroker:
    def __init__(self, cash: float, fee_bps: float, slippage_bps: float):
        self.initial_cash = cash
        self.cash = cash
        self.fee_bps = fee_bps
        self.slippage_bps = slippage_bps
        # Add skew_regime to position tracking
        self.positions = {}  # symbol -> {'qty': float, 'avg_price': float, 'vol_regime': str, 'trend_regime': str, 'skew_regime': str}
        self.trades = []
        self.equity_curve = []

    def execute(self, order: Order, last_price: float, regimes: dict) -> Fill:
        symbol = order["symbol"]
        qty_to_trade = order["qty"]
        side = order["side"]

        slippage_mult = 1 + self.slippage_bps / 10000
        fill_price = last_price * slippage_mult if side == "BUY" else last_price / slippage_mult

        notional = qty_to_trade * fill_price
        fee = notional * self.fee_bps / 10000
        self.cash -= fee

        realized_pnl = 0.0
        entry_vol_regime = 'Unknown'
        entry_trend_regime = 'Unknown'
        entry_skew_regime = 'Unknown'  # New variable

        if symbol not in self.positions:
            self.positions[symbol] = {"qty": 0.0, "avg_price": 0.0,
                                      "vol_regime": "Unknown", "trend_regime": "Unknown", "skew_regime": "Unknown"}

        current_qty = self.positions[symbol]["qty"]
        avg_price = self.positions[symbol]["avg_price"]

        is_new_position = abs(current_qty) < 1e-9
        if is_new_position:
            self.positions[symbol]['vol_regime'] = regimes['volatility_regime']
            self.positions[symbol]['trend_regime'] = regimes['trend_regime']
            # Store skew regime on entry
            self.positions[symbol]['skew_regime'] = regimes['skew_regime']

        entry_vol_regime = self.positions[symbol]['vol_regime']
        entry_trend_regime = self.positions[symbol]['trend_regime']
        # Get entry skew regime
        entry_skew_regime = self.positions[symbol]['skew_regime']

        if side == "BUY":
            if current_qty >= 0:
                new_avg_price = ((avg_price * current_qty) + (fill_price *
                                 qty_to_trade)) / (current_qty + qty_to_trade)
                self.positions[symbol]["avg_price"] = new_avg_price
            else:
                qty_to_close = min(qty_to_trade, abs(current_qty))
                realized_pnl = (avg_price - fill_price) * qty_to_close
                self.cash += realized_pnl

            self.positions[symbol]["qty"] += qty_to_trade
        else:  # SELL
            if current_qty <= 0:
                new_avg_price = ((avg_price * abs(current_qty)) + (fill_price *
                                 qty_to_trade)) / (abs(current_qty) + qty_to_trade)
                self.positions[symbol]["avg_price"] = new_avg_price
            else:
                qty_to_close = min(qty_to_trade, current_qty)
                realized_pnl = (fill_price - avg_price) * qty_to_close
                self.cash += realized_pnl
            self.positions[symbol]["qty"] -= qty_to_trade

        if abs(self.positions[symbol]["qty"]) < 1e-9:
            self.positions[symbol]["qty"] = 0.0
            self.positions[symbol]["avg_price"] = 0.0
            self.positions[symbol]['vol_regime'] = 'Unknown'
            self.positions[symbol]['trend_regime'] = 'Unknown'
            self.positions[symbol]['skew_regime'] = 'Unknown'  # Reset

        fill = {
            "symbol": symbol, "qty": qty_to_trade, "price": fill_price, "fee": fee, "pnl": realized_pnl,
            "volatility_regime": entry_vol_regime,
            "trend_regime": entry_trend_regime,
            "skew_regime": entry_skew_regime  # Return entry skew regime
        }
        return fill

    def mark_to_market(self, prices: dict[str, float]) -> dict:
        market_value = 0.0
        for symbol, pos in self.positions.items():
            qty = pos["qty"]
            if abs(qty) > 1e-9 and symbol in prices:
                market_value += prices[symbol] * qty
        equity = self.cash + market_value
        return {"equity": equity, "cash": self.cash, "market_value": market_value}

    def get_results(self):
        return BacktestResult(
            equity_curve=pd.DataFrame(),
            trades=pd.DataFrame(self.trades),
            summary={},
            score_history=pd.DataFrame()
        )
