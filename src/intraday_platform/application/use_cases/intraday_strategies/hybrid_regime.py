from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from intraday_platform.application.use_cases.intraday_strategies.strategy_utils import (
    candles_to_frame,
    compute_ema,
    compute_sma,
    compute_std,
    compute_vwap,
    generate_signal_id,
)
from intraday_platform.domain.entities.candle import Candle
from intraday_platform.domain.ports.strategy import IntradayStrategy
from intraday_platform.domain.value_objects.signal import Signal
from intraday_platform.domain.value_objects.side import SignalAction


@dataclass(frozen=True)
class HybridRegimeConfig:
    trend_window: int = 30
    trend_threshold: float = 0.5
    vwap_threshold: float = 0.001
    ema_fast: int = 12
    ema_slow: int = 26
    boll_window: int = 20
    boll_std: float = 2.0
    z_window: int = 50
    z_entry: float = 1.5
    z_exit: float = 0.2


class HybridRegimeStrategy(IntradayStrategy):
    """
    Hybrid regime strategy combining trend + mean reversion:
    - Detects trend regime using rolling return t-stat (mean/std).
    - Trend regime: VWAP + EMA confirmation for entries, VWAP/EMA for exits.
    - Mean-reversion regime: Z-score + Bollinger lower band entries, exit on reversion.
    """

    id = "hybrid_regime_v1"
    name = "Hybrid Regime (Trend + Reversion)"

    def __init__(self, config: HybridRegimeConfig | None = None) -> None:
        self._config = config or HybridRegimeConfig()

    def generate_signals(self, symbol: str, candles: Sequence[Candle]) -> Sequence[Signal]:
        min_len = max(self._config.z_window, self._config.boll_window, self._config.ema_slow, self._config.trend_window)
        if len(candles) < min_len + 2:
            return []

        df = candles_to_frame(candles)
        df["vwap"] = compute_vwap(df)
        df["ema_fast"] = compute_ema(df["close"], span=self._config.ema_fast)
        df["ema_slow"] = compute_ema(df["close"], span=self._config.ema_slow)
        df["sma"] = compute_sma(df["close"], window=self._config.boll_window)
        df["std"] = compute_std(df["close"], window=self._config.boll_window)
        df["z_mean"] = compute_sma(df["close"], window=self._config.z_window)
        df["z_std"] = compute_std(df["close"], window=self._config.z_window)
        df["z"] = (df["close"] - df["z_mean"]) / df["z_std"].replace(0, np.nan)

        df["ret"] = df["close"].pct_change()
        df["ret_mean"] = df["ret"].rolling(window=self._config.trend_window, min_periods=self._config.trend_window).mean()
        df["ret_std"] = df["ret"].rolling(window=self._config.trend_window, min_periods=self._config.trend_window).std()
        df["trend_score"] = (df["ret_mean"].abs() / df["ret_std"].replace(0, np.nan)).fillna(0.0)

        latest = df.iloc[-1]
        previous = df.iloc[-2]

        trend_regime = latest["trend_score"] >= self._config.trend_threshold
        if trend_regime:
            return self._trend_signals(symbol, latest, previous, candles[-1].timestamp)
        return self._reversion_signals(symbol, latest, previous, candles[-1].timestamp)

    def _trend_signals(self, symbol: str, latest: pd.Series, previous: pd.Series, timestamp) -> Sequence[Signal]:
        if pd.isna(latest["vwap"]) or pd.isna(previous["vwap"]):
            return []
        crossed_up = previous["close"] <= previous["vwap"] * (1 + self._config.vwap_threshold) and latest[
            "close"
        ] > latest["vwap"] * (1 + self._config.vwap_threshold)
        crossed_down = previous["close"] >= previous["vwap"] and latest["close"] < latest["vwap"]
        ema_bull = latest["ema_fast"] > latest["ema_slow"]
        ema_bear = latest["ema_fast"] < latest["ema_slow"]
        momentum_positive = latest["close"] > previous["close"]

        if crossed_up and ema_bull and momentum_positive:
            return [self._build_signal(symbol, timestamp, SignalAction.BUY, 0.65)]
        if crossed_down or ema_bear:
            return [self._build_signal(symbol, timestamp, SignalAction.EXIT, 0.5)]
        return []

    def _reversion_signals(self, symbol: str, latest: pd.Series, previous: pd.Series, timestamp) -> Sequence[Signal]:
        if pd.isna(latest["sma"]) or pd.isna(latest["std"]) or pd.isna(latest["z"]):
            return []
        lower = latest["sma"] - self._config.boll_std * latest["std"]
        oversold = latest["close"] < lower and latest["z"] <= -self._config.z_entry
        if oversold:
            confidence = min(1.0, abs(float(latest["z"])) / self._config.z_entry)
            return [self._build_signal(symbol, timestamp, SignalAction.BUY, confidence)]

        reversion = latest["close"] >= latest["sma"] or latest["z"] >= -self._config.z_exit
        if reversion:
            return [self._build_signal(symbol, timestamp, SignalAction.EXIT, 0.5)]
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
