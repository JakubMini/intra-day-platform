from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

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
class KalmanTrendConfig:
    process_variance: float = 1e-5
    measurement_variance: float = 1e-2
    entry_band: float = 0.001
    exit_band: float = 0.0
    slope_window: int = 5
    slope_threshold: float = 0.0


class KalmanTrendStrategy(IntradayStrategy):
    """
    Kalman-filter trend strategy:
    - Estimate latent price with a 1D Kalman filter.
    - Enter long on fresh cross above filtered price with positive slope.
    - Exit when price crosses below filtered price or slope turns negative.
    """

    id = "kalman_trend_v1"
    name = "Kalman Filter Trend"

    def __init__(self, config: KalmanTrendConfig | None = None) -> None:
        self._config = config or KalmanTrendConfig()

    def generate_signals(self, symbol: str, candles: Sequence[Candle]) -> Sequence[Signal]:
        if len(candles) < max(10, self._config.slope_window + 2):
            return []

        df = candles_to_frame(candles)
        close = df["close"].to_numpy(dtype=float)

        filtered = self._kalman_filter(close)
        if filtered.size < self._config.slope_window + 2:
            return []

        latest_price = close[-1]
        prev_price = close[-2]
        latest_filter = filtered[-1]
        prev_filter = filtered[-2]

        slope = (filtered[-1] - filtered[-1 - self._config.slope_window]) / float(self._config.slope_window)
        crossed_up = prev_price <= prev_filter * (1 + self._config.entry_band) and latest_price > latest_filter * (
            1 + self._config.entry_band
        )
        crossed_down = prev_price >= prev_filter * (1 - self._config.exit_band) and latest_price < latest_filter * (
            1 - self._config.exit_band
        )

        if crossed_up and slope > self._config.slope_threshold:
            return [self._build_signal(symbol, candles[-1].timestamp, SignalAction.BUY, 0.6)]

        if crossed_down or slope < 0:
            return [self._build_signal(symbol, candles[-1].timestamp, SignalAction.EXIT, 0.5)]

        return []

    def _kalman_filter(self, series: np.ndarray) -> np.ndarray:
        q = self._config.process_variance
        r = self._config.measurement_variance
        x = series[0]
        p = 1.0
        filtered = np.zeros_like(series)
        filtered[0] = x
        for i in range(1, len(series)):
            # Predict
            p = p + q
            # Update
            k = p / (p + r)
            x = x + k * (series[i] - x)
            p = (1 - k) * p
            filtered[i] = x
        return filtered

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
