from dataclasses import dataclass
from typing import Any, Optional, Dict
from .types import Bar, Order, Fill, PortfolioSnapshot

@dataclass
class Event: ...
@dataclass
class BarEvent(Event):
    symbol: str
    interval: str
    bar: Bar

@dataclass
class SignalEvent(Event):
    symbol: str
    weight: float  # target weight in [-1, 1]

@dataclass
class OrderEvent(Event):
    order: Order

@dataclass
class FillEvent(Event):
    fill: Fill

@dataclass
class PortfolioEvent(Event):
    snapshot: PortfolioSnapshot

@dataclass
class RiskEvent(Event):
    name: str
    detail: Dict[str, Any]
