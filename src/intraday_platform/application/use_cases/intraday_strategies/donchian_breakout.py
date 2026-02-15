from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd

from intraday_platform.application.use_cases.intraday_strategies.strategy_utils import (
    candles_to_frame,
    generate_signal_id,
)
from intraday_platform.domain.entities.candle import Candle
from intraday_platform.domain.ports.strategy import IntradayStrategy
from intraday_platform.domain.value_objects.signal import Signal
from intraday_platform.domain.value_objects.side import SignalAction
from intraday_platform.infrastructure.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class DonchianBreakoutConfig:
    entry_window: int = 20
    exit_window: int = 10


class DonchianBreakoutStrategy(IntradayStrategy):
    """
    Donchian Channel breakout (trend-following):
    - Enter long when price breaks above the prior N-period high.
    - Exit when price breaks below the prior M-period low.
    """

    id = "donchian_breakout_v1"
    name = "Donchian Channel Breakout"

    def __init__(self, config: DonchianBreakoutConfig | None = None) -> None:
        self._config = config or DonchianBreakoutConfig()

    def generate_signals(self, symbol: str, candles: Sequence[Candle]) -> Sequence[Signal]:
        if len(candles) < max(self._config.entry_window, self._config.exit_window) + 2:
            return []

        df = candles_to_frame(candles)
        df["entry_high"] = df["high"].rolling(self._config.entry_window).max().shift(1)
        df["exit_low"] = df["low"].rolling(self._config.exit_window).min().shift(1)

        latest = df.iloc[-1]
        previous = df.iloc[-2]

        if pd.isna(latest["entry_high"]) or pd.isna(latest["exit_low"]):
            return []

        crossed_up = previous["close"] <= previous["entry_high"] and latest["close"] > latest["entry_high"]
        crossed_down = previous["close"] >= previous["exit_low"] and latest["close"] < latest["exit_low"]

        if crossed_up:
            return [self._build_signal(symbol, candles[-1].timestamp, SignalAction.BUY)]

        if crossed_down:
            return [self._build_signal(symbol, candles[-1].timestamp, SignalAction.EXIT)]

        return []

    def _build_signal(self, symbol: str, timestamp, action: SignalAction) -> Signal:
        confidence = 0.6 if action == SignalAction.BUY else 0.5
        return Signal(
            signal_id=generate_signal_id(self.id, symbol, timestamp),
            symbol=symbol,
            action=action,
            confidence=confidence,
            timestamp=timestamp,
            strategy_id=self.id,
            metadata={"strategy": self.name},
        )
