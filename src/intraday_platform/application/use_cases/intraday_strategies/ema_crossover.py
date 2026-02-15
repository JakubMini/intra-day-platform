from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from intraday_platform.application.use_cases.intraday_strategies.strategy_utils import (
    candles_to_frame,
    compute_ema,
    generate_signal_id,
)
from intraday_platform.domain.entities.candle import Candle
from intraday_platform.domain.ports.strategy import IntradayStrategy
from intraday_platform.domain.value_objects.signal import Signal
from intraday_platform.domain.value_objects.side import SignalAction
from intraday_platform.infrastructure.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class EmaCrossoverConfig:
    fast_period: int = 12
    slow_period: int = 26
    min_spread: float = 0.0005


class EmaCrossoverStrategy(IntradayStrategy):
    """
    EMA crossover strategy:
    - Enter long when fast EMA crosses above slow EMA.
    - Exit when fast EMA crosses below slow EMA.
    """

    id = "ema_crossover_v1"
    name = "EMA Crossover"

    def __init__(self, config: EmaCrossoverConfig | None = None) -> None:
        self._config = config or EmaCrossoverConfig()

    def generate_signals(self, symbol: str, candles: Sequence[Candle]) -> Sequence[Signal]:
        min_required = max(self._config.fast_period, self._config.slow_period) + 2
        if len(candles) < min_required:
            logger.debug("Not enough candles for EMA crossover", extra={"symbol": symbol})
            return []

        df = candles_to_frame(candles)
        df["fast"] = compute_ema(df["close"], self._config.fast_period)
        df["slow"] = compute_ema(df["close"], self._config.slow_period)

        latest = df.iloc[-1]
        previous = df.iloc[-2]

        spread_now = (latest["fast"] - latest["slow"]) / latest["slow"] if latest["slow"] != 0 else 0.0
        spread_prev = (previous["fast"] - previous["slow"]) / previous["slow"] if previous["slow"] != 0 else 0.0

        crossed_up = spread_prev <= self._config.min_spread and spread_now > self._config.min_spread
        crossed_down = spread_prev >= -self._config.min_spread and spread_now < -self._config.min_spread

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
