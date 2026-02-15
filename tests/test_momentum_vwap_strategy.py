from __future__ import annotations

from datetime import datetime, timedelta

from intraday_platform.application.use_cases.intraday_strategies.momentum_vwap import (
    MomentumVwapStrategy,
)
from intraday_platform.domain.entities.candle import Candle
from intraday_platform.domain.value_objects.side import SignalAction


def test_momentum_vwap_generates_buy_signal() -> None:
    start = datetime(2024, 1, 2, 9, 0, 0)
    candles: list[Candle] = []
    for i in range(24):
        candles.append(
            Candle(
                symbol="TEST.L",
                timestamp=start + timedelta(minutes=i),
                open=100.0,
                high=100.0,
                low=100.0,
                close=100.0,
                volume=100.0,
            )
        )
    candles.append(
        Candle(
            symbol="TEST.L",
            timestamp=start + timedelta(minutes=24),
            open=101.0,
            high=101.0,
            low=101.0,
            close=101.0,
            volume=500.0,
        )
    )

    strategy = MomentumVwapStrategy()
    signals = strategy.generate_signals("TEST.L", candles)
    assert signals
    assert signals[-1].action == SignalAction.BUY
