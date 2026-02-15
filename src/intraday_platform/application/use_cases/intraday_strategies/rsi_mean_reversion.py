from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from intraday_platform.application.use_cases.intraday_strategies.strategy_utils import (
    candles_to_frame,
    compute_rsi,
    generate_signal_id,
)
from intraday_platform.domain.entities.candle import Candle
from intraday_platform.domain.ports.strategy import IntradayStrategy
from intraday_platform.domain.value_objects.signal import Signal
from intraday_platform.domain.value_objects.side import SignalAction
from intraday_platform.infrastructure.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class RsiMeanReversionConfig:
    period: int = 14
    oversold: float = 30.0
    exit_level: float = 50.0


class RsiMeanReversionStrategy(IntradayStrategy):
    """
    RSI mean reversion strategy:
    - Enter long when RSI crosses back above the oversold threshold (bounce).
    - Exit when RSI crosses above the exit level.
    """

    id = "rsi_mean_reversion_v1"
    name = "RSI Mean Reversion"

    def __init__(self, config: RsiMeanReversionConfig | None = None) -> None:
        self._config = config or RsiMeanReversionConfig()

    def generate_signals(self, symbol: str, candles: Sequence[Candle]) -> Sequence[Signal]:
        if len(candles) < self._config.period + 2:
            logger.debug("Not enough candles for RSI", extra={"symbol": symbol})
            return []

        df = candles_to_frame(candles)
        df["rsi"] = compute_rsi(df["close"], self._config.period)

        latest = df.iloc[-1]
        previous = df.iloc[-2]

        crossed_oversold = previous["rsi"] < self._config.oversold and latest["rsi"] >= self._config.oversold
        crossed_exit = previous["rsi"] < self._config.exit_level and latest["rsi"] >= self._config.exit_level

        if crossed_oversold:
            return [self._build_signal(symbol, candles[-1].timestamp, SignalAction.BUY)]

        if crossed_exit:
            return [self._build_signal(symbol, candles[-1].timestamp, SignalAction.EXIT)]

        return []

    def _build_signal(self, symbol: str, timestamp, action: SignalAction) -> Signal:
        confidence = 0.55 if action == SignalAction.BUY else 0.5
        return Signal(
            signal_id=generate_signal_id(self.id, symbol, timestamp),
            symbol=symbol,
            action=action,
            confidence=confidence,
            timestamp=timestamp,
            strategy_id=self.id,
            metadata={"strategy": self.name},
        )
