from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd

from intraday_platform.application.use_cases.intraday_strategies.strategy_utils import (
    candles_to_frame,
    compute_sma,
    compute_std,
    generate_signal_id,
)
from intraday_platform.domain.entities.candle import Candle
from intraday_platform.domain.ports.strategy import IntradayStrategy
from intraday_platform.domain.value_objects.signal import Signal
from intraday_platform.domain.value_objects.side import SignalAction
from intraday_platform.infrastructure.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class BollingerReversionConfig:
    window: int = 20
    std_dev: float = 2.0


class BollingerReversionStrategy(IntradayStrategy):
    """
    Bollinger Band mean reversion:
    - Enter long when price crosses back above the lower band.
    - Exit when price crosses above the middle band.
    """

    id = "bollinger_reversion_v1"
    name = "Bollinger Band Reversion"

    def __init__(self, config: BollingerReversionConfig | None = None) -> None:
        self._config = config or BollingerReversionConfig()

    def generate_signals(self, symbol: str, candles: Sequence[Candle]) -> Sequence[Signal]:
        if len(candles) < self._config.window + 2:
            return []

        df = candles_to_frame(candles)
        df["sma"] = compute_sma(df["close"], self._config.window)
        df["std"] = compute_std(df["close"], self._config.window)
        df["lower"] = df["sma"] - self._config.std_dev * df["std"]
        df["upper"] = df["sma"] + self._config.std_dev * df["std"]

        latest = df.iloc[-1]
        previous = df.iloc[-2]

        if pd.isna(latest["lower"]) or pd.isna(previous["lower"]) or pd.isna(latest["sma"]):
            return []

        crossed_up = previous["close"] <= previous["lower"] and latest["close"] > latest["lower"]
        crossed_mid = previous["close"] <= previous["sma"] and latest["close"] > latest["sma"]

        if crossed_up:
            return [self._build_signal(symbol, candles[-1].timestamp, SignalAction.BUY)]

        if crossed_mid:
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
