from __future__ import annotations

from datetime import date, timedelta
from typing import Sequence

from intraday_platform.application.ports.market_data import MarketDataProvider
from intraday_platform.config import LIQUIDITY_LOOKBACK_DAYS, LIQUIDITY_MIN_AVG_DAILY_VOLUME
from intraday_platform.domain.entities.candle import Candle
from intraday_platform.infrastructure.logging import get_logger

logger = get_logger(__name__)


class LiquidityFilter:
    def __init__(
        self,
        provider: MarketDataProvider,
        lookback_days: int = LIQUIDITY_LOOKBACK_DAYS,
        min_avg_daily_volume: float = LIQUIDITY_MIN_AVG_DAILY_VOLUME,
    ) -> None:
        self._provider = provider
        self._lookback_days = lookback_days
        self._min_avg_daily_volume = min_avg_daily_volume

    def filter(self, symbols: Sequence[str], as_of: date) -> list[str]:
        filtered: list[str] = []
        lookback_span = max(self._lookback_days * 3, self._lookback_days + 7)
        start = as_of - timedelta(days=lookback_span)

        for symbol in symbols:
            candles = self._provider.fetch_daily_candles(symbol, start=start, end=as_of)
            avg_volume = self._average_volume(candles)
            if avg_volume >= self._min_avg_daily_volume:
                filtered.append(symbol)
                logger.debug(
                    "Liquidity pass",
                    extra={"symbol": symbol, "avg_volume": avg_volume},
                )
            else:
                logger.warning(
                    "Liquidity fail",
                    extra={"symbol": symbol, "avg_volume": avg_volume},
                )

        logger.debug(
            "Liquidity filtering complete",
            extra={"input": len(symbols), "output": len(filtered)},
        )
        return filtered

    def _average_volume(self, candles: Sequence[Candle]) -> float:
        if not candles:
            return 0.0
        recent = candles[-self._lookback_days :]
        volume_sum = sum(candle.volume for candle in recent)
        return volume_sum / max(len(recent), 1)
