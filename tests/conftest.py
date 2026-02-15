from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from intraday_platform.domain.entities.candle import Candle


@pytest.fixture
def base_time() -> datetime:
    return datetime(2024, 1, 2, 9, 0, 0)


def make_candle(symbol: str, timestamp: datetime, price: float, volume: float) -> Candle:
    return Candle(
        symbol=symbol,
        timestamp=timestamp,
        open=price,
        high=price,
        low=price,
        close=price,
        volume=volume,
    )


def make_candles(symbol: str, start: datetime, prices: list[float], volumes: list[float]) -> list[Candle]:
    candles: list[Candle] = []
    for idx, (price, volume) in enumerate(zip(prices, volumes, strict=True)):
        candles.append(make_candle(symbol, start + timedelta(minutes=idx), price, volume))
    return candles
