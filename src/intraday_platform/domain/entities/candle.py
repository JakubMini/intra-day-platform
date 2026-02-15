from __future__ import annotations

from datetime import datetime

from pydantic import Field

from intraday_platform.domain.base import DomainModel
from intraday_platform.domain.value_objects.timeframe import Timeframe


class Candle(DomainModel):
    symbol: str
    timestamp: datetime
    open: float = Field(..., gt=0.0)
    high: float = Field(..., gt=0.0)
    low: float = Field(..., gt=0.0)
    close: float = Field(..., gt=0.0)
    volume: float = Field(..., ge=0.0)
    timeframe: Timeframe = Timeframe.ONE_MINUTE
    vwap: float | None = Field(default=None, ge=0.0)
