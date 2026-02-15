from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date, datetime
from typing import Sequence

from intraday_platform.domain.entities.candle import Candle
from intraday_platform.domain.value_objects.timeframe import Timeframe


class UniverseRepository(ABC):
    @abstractmethod
    def list_symbols(self) -> Sequence[str]:
        raise NotImplementedError


class CandleRepository(ABC):
    @abstractmethod
    def get_intraday_candles(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: Timeframe,
    ) -> Sequence[Candle]:
        raise NotImplementedError

    @abstractmethod
    def get_daily_candles(self, symbol: str, start: date, end: date) -> Sequence[Candle]:
        raise NotImplementedError

    @abstractmethod
    def store_intraday_candles(
        self,
        symbol: str,
        timeframe: Timeframe,
        candles: Sequence[Candle],
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def store_daily_candles(self, symbol: str, candles: Sequence[Candle]) -> None:
        raise NotImplementedError
