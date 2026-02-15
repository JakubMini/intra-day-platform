from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Sequence

from intraday_platform.domain.entities.candle import Candle
from intraday_platform.domain.value_objects.signal import Signal


class IntradayStrategy(ABC):
    id: str
    name: str

    @abstractmethod
    def generate_signals(self, symbol: str, candles: Sequence[Candle]) -> Sequence[Signal]:
        raise NotImplementedError


class DailySelectionStrategy(ABC):
    id: str
    name: str

    @abstractmethod
    def select(self, universe: Iterable[str], daily_candles: Sequence[Candle]) -> Sequence[str]:
        raise NotImplementedError
