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
class ZScoreReversionConfig:
    window: int = 50
    entry_z: float = 1.5
    exit_z: float = 0.2


class ZScoreReversionStrategy(IntradayStrategy):
    """
    Statistical mean reversion using z-score of price.
    - Enter long when price deviates below the rolling mean by entry_z standard deviations.
    - Exit when z-score reverts above -exit_z.
    """

    id = "zscore_reversion_v1"
    name = "Z-Score Mean Reversion"

    def __init__(self, config: ZScoreReversionConfig | None = None) -> None:
        self._config = config or ZScoreReversionConfig()

    def generate_signals(self, symbol: str, candles: Sequence[Candle]) -> Sequence[Signal]:
        if len(candles) < self._config.window + 2:
            return []

        df = candles_to_frame(candles)
        df["mean"] = df["close"].rolling(window=self._config.window, min_periods=self._config.window).mean()
        df["std"] = df["close"].rolling(window=self._config.window, min_periods=self._config.window).std()
        df["z"] = (df["close"] - df["mean"]) / df["std"].replace(0, pd.NA)

        latest = df.iloc[-1]
        previous = df.iloc[-2]
        if pd.isna(latest["z"]) or pd.isna(previous["z"]):
            return []

        z_now = float(latest["z"])
        z_prev = float(previous["z"])

        crossed_entry = z_prev > -self._config.entry_z and z_now <= -self._config.entry_z
        crossed_exit = z_now >= -self._config.exit_z

        if crossed_entry:
            confidence = min(1.0, abs(z_now) / self._config.entry_z)
            return [self._build_signal(symbol, candles[-1].timestamp, SignalAction.BUY, confidence)]

        if crossed_exit:
            return [self._build_signal(symbol, candles[-1].timestamp, SignalAction.EXIT, 0.5)]

        return []

    def _build_signal(self, symbol: str, timestamp, action: SignalAction, confidence: float) -> Signal:
        return Signal(
            signal_id=generate_signal_id(self.id, symbol, timestamp),
            symbol=symbol,
            action=action,
            confidence=confidence,
            timestamp=timestamp,
            strategy_id=self.id,
            metadata={"strategy": self.name},
        )
