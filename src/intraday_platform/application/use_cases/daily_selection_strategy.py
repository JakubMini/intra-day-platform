from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import pandas as pd

from intraday_platform.config import (
    DAILY_SELECTION_BREAKOUT_WINDOW,
    DAILY_SELECTION_LOOKBACK_DAYS,
    DAILY_SELECTION_PRIOR_WINDOW,
    DAILY_SELECTION_RECENT_WINDOW,
    MAX_STOCKS_PER_DAY,
)
from intraday_platform.domain.entities.candle import Candle
from intraday_platform.domain.ports.strategy import DailySelectionStrategy
from intraday_platform.infrastructure.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class SelectionWeights:
    volume_surge: float = 0.2
    volatility_expansion: float = 0.2
    momentum: float = 0.2
    breakout: float = 0.2
    relative_strength: float = 0.2


class MultiFactorDailySelectionStrategy(DailySelectionStrategy):
    """
    Selects stocks using multiple daily metrics:
    - Volume surge
    - Volatility expansion
    - Momentum
    - Breakout
    - Relative strength
    """

    id = "multi_factor_daily_v1"
    name = "Multi-Factor Daily Selection"

    def __init__(
        self,
        lookback_days: int = DAILY_SELECTION_LOOKBACK_DAYS,
        recent_window: int = DAILY_SELECTION_RECENT_WINDOW,
        prior_window: int = DAILY_SELECTION_PRIOR_WINDOW,
        breakout_window: int = DAILY_SELECTION_BREAKOUT_WINDOW,
        weights: SelectionWeights | None = None,
        max_symbols: int = MAX_STOCKS_PER_DAY,
    ) -> None:
        self._lookback_days = lookback_days
        self._recent_window = recent_window
        self._prior_window = prior_window
        self._breakout_window = breakout_window
        self._weights = weights or SelectionWeights()
        self._max_symbols = max_symbols

    def select(self, universe: Iterable[str], daily_candles: Sequence[Candle]) -> Sequence[str]:
        universe_list = list(universe)
        logger.debug("Daily selection universe size", extra={"count": len(universe_list)})
        grouped = self._group_by_symbol(daily_candles)
        metrics = self._compute_metrics(grouped)
        if not metrics:
            logger.warning("No symbols passed metric computation")
            return []

        normalized = self._normalize_metrics(metrics)
        ranked = sorted(normalized.items(), key=lambda item: item[1]["score"], reverse=True)
        selected = [symbol for symbol, _ in ranked[: self._max_symbols]]

        logger.debug(
            "Daily selection complete",
            extra={"selected": selected, "candidates": len(normalized)},
        )
        return selected

    def export_config(self) -> dict[str, float | int]:
        return {
            "lookback_days": self._lookback_days,
            "recent_window": self._recent_window,
            "prior_window": self._prior_window,
            "breakout_window": self._breakout_window,
            "max_symbols": self._max_symbols,
            "weight_volume_surge": self._weights.volume_surge,
            "weight_volatility_expansion": self._weights.volatility_expansion,
            "weight_momentum": self._weights.momentum,
            "weight_breakout": self._weights.breakout,
            "weight_relative_strength": self._weights.relative_strength,
        }

    def _group_by_symbol(self, candles: Sequence[Candle]) -> dict[str, list[Candle]]:
        grouped: dict[str, list[Candle]] = {}
        for candle in candles:
            grouped.setdefault(candle.symbol, []).append(candle)
        for symbol, series in grouped.items():
            series.sort(key=lambda c: c.timestamp)
            if len(series) < self._lookback_days:
                logger.debug(
                    "Insufficient candles for symbol",
                    extra={"symbol": symbol, "count": len(series)},
                )
        return grouped

    def _compute_metrics(self, grouped: dict[str, list[Candle]]) -> dict[str, dict[str, float]]:
        metrics: dict[str, dict[str, float]] = {}
        for symbol, candles in grouped.items():
            if len(candles) < max(self._lookback_days, self._prior_window + self._recent_window + 1):
                continue

            df = pd.DataFrame(
                {
                    "close": [c.close for c in candles],
                    "high": [c.high for c in candles],
                    "volume": [c.volume for c in candles],
                }
            )
            volume_surge = self._volume_surge(df)
            volatility_expansion = self._volatility_expansion(df)
            momentum = self._momentum(df)
            breakout = self._breakout(df)

            metrics[symbol] = {
                "volume_surge": volume_surge,
                "volatility_expansion": volatility_expansion,
                "momentum": momentum,
                "breakout": breakout,
            }

        return metrics

    def _normalize_metrics(self, metrics: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
        df = pd.DataFrame(metrics).T
        if df.empty:
            return {}

        df["relative_strength"] = df["momentum"].rank(pct=True)
        normalized = pd.DataFrame(index=df.index)
        for column in ["volume_surge", "volatility_expansion", "momentum", "breakout"]:
            series = df[column]
            min_val = series.min()
            max_val = series.max()
            if max_val == min_val:
                normalized[column] = 0.0
            else:
                normalized[column] = (series - min_val) / (max_val - min_val)
        normalized["relative_strength"] = df["relative_strength"].fillna(0.0)

        score = (
            normalized["volume_surge"] * self._weights.volume_surge
            + normalized["volatility_expansion"] * self._weights.volatility_expansion
            + normalized["momentum"] * self._weights.momentum
            + normalized["breakout"] * self._weights.breakout
            + normalized["relative_strength"] * self._weights.relative_strength
        )
        normalized["score"] = score

        return normalized.to_dict(orient="index")

    def _volume_surge(self, df: pd.DataFrame) -> float:
        recent = df["volume"].iloc[-self._recent_window :].mean()
        prior = df["volume"].iloc[-(self._recent_window + self._prior_window) : -self._recent_window].mean()
        if prior <= 0:
            return 0.0
        return float(recent / prior)

    def _volatility_expansion(self, df: pd.DataFrame) -> float:
        returns = df["close"].pct_change().dropna()
        recent = returns.iloc[-self._recent_window :].std()
        prior = returns.iloc[-(self._recent_window + self._prior_window) : -self._recent_window].std()
        if prior is None or prior == 0 or pd.isna(prior) or recent is None or pd.isna(recent):
            return 0.0
        return float(recent / prior) if recent is not None else 0.0

    def _momentum(self, df: pd.DataFrame) -> float:
        if len(df) < self._prior_window + 1:
            return 0.0
        recent_close = df["close"].iloc[-1]
        past_close = df["close"].iloc[-(self._prior_window + 1)]
        if past_close == 0:
            return 0.0
        return float((recent_close / past_close) - 1.0)

    def _breakout(self, df: pd.DataFrame) -> float:
        if len(df) < self._breakout_window + 1:
            return 0.0
        recent_close = df["close"].iloc[-1]
        prior_high = df["high"].iloc[-(self._breakout_window + 1) : -1].max()
        if prior_high == 0:
            return 0.0
        return float((recent_close / prior_high) - 1.0)
