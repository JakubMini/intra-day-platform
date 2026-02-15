from __future__ import annotations

from datetime import datetime, timedelta

from intraday_platform.application.use_cases.daily_selection_strategy import MultiFactorDailySelectionStrategy
from intraday_platform.domain.entities.candle import Candle


def _daily_candles(symbol: str, start: datetime, closes: list[float], volumes: list[float]) -> list[Candle]:
    candles: list[Candle] = []
    for idx, (close, volume) in enumerate(zip(closes, volumes, strict=True)):
        ts = start + timedelta(days=idx)
        candles.append(
            Candle(
                symbol=symbol,
                timestamp=ts,
                open=close,
                high=close,
                low=close,
                close=close,
                volume=volume,
            )
        )
    return candles


def test_daily_selection_prefers_stronger_symbol() -> None:
    strategy = MultiFactorDailySelectionStrategy(
        lookback_days=10,
        recent_window=3,
        prior_window=3,
        breakout_window=5,
        max_symbols=1,
    )
    start = datetime(2024, 1, 1)
    win_closes = [100 + i for i in range(12)]
    lose_closes = [100 for _ in range(12)]
    win_vol = [200_000 for _ in range(12)]
    lose_vol = [50_000 for _ in range(12)]

    daily_candles = _daily_candles("WIN.L", start, win_closes, win_vol) + _daily_candles(
        "LOSE.L", start, lose_closes, lose_vol
    )
    selected = strategy.select(["WIN.L", "LOSE.L"], daily_candles)
    assert selected == ["WIN.L"]
