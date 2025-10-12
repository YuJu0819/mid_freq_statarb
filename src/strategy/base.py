from abc import ABC, abstractmethod
import pandas as pd
from ..core.event import SignalEvent

class Strategy(ABC):
    @abstractmethod
    def on_bar(self, symbol: str, interval: str, df: pd.DataFrame) -> SignalEvent | None:
        ...

    def name(self) -> str:
        return self.__class__.__name__
