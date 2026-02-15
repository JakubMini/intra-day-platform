from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date, datetime
from typing import Sequence

from intraday_platform.domain.entities.candle import Candle
from intraday_platform.domain.value_objects.timeframe import Timeframe


class MarketDataProvider(ABC):
    @abstractmethod
    def fetch_intraday_candles(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: Timeframe,
    ) -> Sequence[Candle]:
        raise NotImplementedError

    @abstractmethod
    def fetch_daily_candles(
        self,
        symbol: str,
        start: date,
        end: date,
    ) -> Sequence[Candle]:
        raise NotImplementedError

    @abstractmethod
    def fetch_latest_candle(self, symbol: str) -> Candle | None:
        raise NotImplementedError
