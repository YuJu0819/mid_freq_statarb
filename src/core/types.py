from typing import Literal, TypedDict, Optional, Dict, Any

Side = Literal["BUY", "SELL"]
OrderType = Literal["MARKET", "LIMIT"]
TimeInForce = Literal["GTC", "IOC", "FOK"]

class Bar(TypedDict):
    ts: int          # epoch ms (open time for kline)
    open: float
    high: float
    low: float
    close: float
    volume: float

class Order(TypedDict):
    symbol: str
    side: Side
    qty: float
    order_type: OrderType
    price: Optional[float]
    tif: Optional[TimeInForce]

class Fill(TypedDict):
    symbol: str
    qty: float
    price: float
    ts: int

class Position(TypedDict):
    symbol: str
    qty: float
    avg_price: float

class PortfolioSnapshot(TypedDict):
    equity: float
    cash: float
    positions: Dict[str, Position]
    ts: int
