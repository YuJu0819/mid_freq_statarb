import pandas as pd
from typing import Optional, Dict, Any
from ..core.event import SignalEvent
from .base import Strategy


class SMACross(Strategy):
    # Add a 'mode' parameter to know if we are in 'backtest' or 'live'
    def __init__(self, fast: int = 10, slow: int = 30, cfg: Optional[Dict[str, Any]] = None, mode: str = "backtest"):
        if fast >= slow:
            raise ValueError("fast must be < slow")
        self.fast = fast
        self.slow = slow
        self.cfg = cfg if cfg is not None else {}
        self.mode = mode  # 'backtest' or 'live'
        self._last_state = 0  # -1 short, 0 flat, 1 long

    def on_bar(self, symbol: str, interval: str, df: pd.DataFrame) -> SignalEvent | None:
        if len(df) < self.slow + 2:
            return None
        sma_f = df["close"].rolling(self.fast).mean()
        sma_s = df["close"].rolling(self.slow).mean()
        cross_up = sma_f.iloc[-2] <= sma_s.iloc[-2] and sma_f.iloc[-1] > sma_s.iloc[-1]
        cross_dn = sma_f.iloc[-2] >= sma_s.iloc[-2] and sma_f.iloc[-1] < sma_s.iloc[-1]

        # V-- CHANGE IS HERE --V
        # Look in the correct section ('live' or 'backtest') of the config
        config_section = self.cfg.get(self.mode, {})
        target_weight = config_section.get("max_position_notional", 1.0)

        if cross_up and self._last_state != 1:
            self._last_state = 1
            return SignalEvent(symbol=symbol, weight=target_weight)
        # ^-- CHANGE IS HERE --^

        if cross_dn and self._last_state != -1:
            self._last_state = -1
            return SignalEvent(symbol=symbol, weight=0.0)  # flat for spot
        return None
