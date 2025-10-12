from ..core.event import RiskEvent
from ..core.types import PortfolioSnapshot

def risk_checks(snapshot: PortfolioSnapshot, symbol: str, price: float,
                target_weight: float, max_position_notional: float) -> list[RiskEvent]:
    events = []
    equity = snapshot["equity"]
    target_notional = equity * abs(target_weight)
    if target_notional > equity * max_position_notional:
        events.append(RiskEvent(name="MAX_POSITION", detail={
            "allowed": equity*max_position_notional, "requested": target_notional
        }))
    return events
