from __future__ import annotations

from datetime import datetime, timedelta

from intraday_platform.application.use_cases.candle_resampler import resample_candles
from intraday_platform.domain.entities.candle import Candle
from intraday_platform.domain.value_objects.timeframe import Timeframe


def test_resampler_produces_valid_candles() -> None:
    start = datetime(2024, 1, 1, 10, 0, 0)
    candles = []
    for i in range(6):
        price = 100.0 + i
        candles.append(
            Candle(
                symbol="TEST.L",
                timestamp=start + timedelta(minutes=i),
                open=price,
                high=price,
                low=price,
                close=price,
                volume=100.0,
            )
        )

    resampled = resample_candles(candles, Timeframe.TWO_MINUTE)
    assert len(resampled) > 0
    assert len(resampled) <= len(candles)
    for candle in resampled:
        assert candle.open > 0
        assert candle.high > 0
        assert candle.low > 0
        assert candle.close > 0
        assert candle.volume >= 0
