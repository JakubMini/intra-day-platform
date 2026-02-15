from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd

from intraday_platform.application.use_cases.intraday_strategies.strategy_utils import (
    candles_to_frame,
    compute_vwap,
    generate_signal_id,
)
from intraday_platform.domain.entities.candle import Candle
from intraday_platform.domain.ports.strategy import IntradayStrategy
from intraday_platform.domain.value_objects.signal import Signal
from intraday_platform.domain.value_objects.side import SignalAction
from intraday_platform.infrastructure.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class MomentumVwapConfig:
    volume_window: int = 20
    vwap_threshold: float = 0.001
    momentum_window: int = 5
    exit_vwap_threshold: float = 0.0


class MomentumVwapStrategy(IntradayStrategy):
    """
    Momentum continuation strategy:
    - Enter long on a fresh cross above VWAP with volume confirmation and positive momentum.
    - Exit on a fresh cross below VWAP or momentum rolling over.
    """

    id = "momentum_vwap_v1"
    name = "Momentum VWAP Continuation"

    def __init__(self, config: MomentumVwapConfig | None = None) -> None:
        self._config = config or MomentumVwapConfig()

    def generate_signals(self, symbol: str, candles: Sequence[Candle]) -> Sequence[Signal]:
        if len(candles) < max(self._config.volume_window, self._config.momentum_window) + 2:
            logger.debug("Not enough candles for momentum VWAP", extra={"symbol": symbol})
            return []

        df = candles_to_frame(candles)
        df["vwap"] = compute_vwap(df)
        df["volume_ma"] = df["volume"].rolling(window=self._config.volume_window, min_periods=1).mean()
        df["momentum"] = df["close"].pct_change(self._config.momentum_window).fillna(0.0)

        latest = df.iloc[-1]
        previous = df.iloc[-2]

        vwap_value = latest["vwap"]
        prev_vwap = previous["vwap"]
        if pd.isna(vwap_value) or pd.isna(prev_vwap) or vwap_value <= 0 or prev_vwap <= 0:
            logger.debug("VWAP not available yet", extra={"symbol": symbol})
            return []

        price_above_now = latest["close"] > vwap_value * (1 + self._config.vwap_threshold)
        price_above_prev = previous["close"] > prev_vwap * (1 + self._config.vwap_threshold)
        price_below_now = latest["close"] <= vwap_value * (1 + self._config.exit_vwap_threshold)
        price_below_prev = previous["close"] <= prev_vwap * (1 + self._config.exit_vwap_threshold)

        volume_ma = latest["volume_ma"]
        if pd.isna(volume_ma) or volume_ma <= 0:
            volume_confirm = True
        else:
            volume_confirm = latest["volume"] > volume_ma

        momentum_now = latest["momentum"]
        momentum_prev = previous["momentum"]
        momentum_positive = momentum_now > 0
        momentum_cross_down = momentum_prev > 0 and momentum_now <= 0

        crossed_up = (not price_above_prev) and price_above_now
        crossed_down = (not price_below_prev) and price_below_now

        if crossed_up and volume_confirm and momentum_positive:
            signal = self._build_signal(symbol, candles[-1].timestamp, SignalAction.BUY)
            return [signal]

        if crossed_down or momentum_cross_down:
            signal = self._build_signal(symbol, candles[-1].timestamp, SignalAction.EXIT)
            return [signal]

        return []

    def _build_signal(self, symbol: str, timestamp, action: SignalAction) -> Signal:
        return Signal(
            signal_id=generate_signal_id(self.id, symbol, timestamp),
            symbol=symbol,
            action=action,
            confidence=0.6 if action == SignalAction.BUY else 0.5,
            timestamp=timestamp,
            strategy_id=self.id,
            metadata={"strategy": self.name},
        )
