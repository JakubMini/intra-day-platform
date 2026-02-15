from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from typing import Iterable, Sequence

from intraday_platform.application.ports.market_data import MarketDataProvider
from intraday_platform.application.ports.repositories import CandleRepository, UniverseRepository
from intraday_platform.application.ports.selection_backend import SelectionBackend
from intraday_platform.application.use_cases.liquidity_filter import LiquidityFilter
from intraday_platform.config import DAILY_SELECTION_LOOKBACK_DAYS, DAILY_SELECTION_MAX_WORKERS
from intraday_platform.domain.entities.candle import Candle
from intraday_platform.domain.ports.strategy import DailySelectionStrategy
from intraday_platform.infrastructure.logging import get_logger

logger = get_logger(__name__)


class DailySelectionEngine:
    """
    Orchestrates daily stock selection using a universe source, liquidity filter,
    and a pluggable selection strategy.
    """

    def __init__(
        self,
        provider: MarketDataProvider,
        universe_repository: UniverseRepository,
        strategy: DailySelectionStrategy,
        liquidity_filter: LiquidityFilter | None = None,
        selection_backend: SelectionBackend | None = None,
        candle_repository: CandleRepository | None = None,
        lookback_days: int = DAILY_SELECTION_LOOKBACK_DAYS,
        fetch_workers: int = DAILY_SELECTION_MAX_WORKERS,
    ) -> None:
        self._provider = provider
        self._universe_repository = universe_repository
        self._strategy = strategy
        self._liquidity_filter = liquidity_filter
        self._selection_backend = selection_backend
        self._candle_repository = candle_repository
        self._lookback_days = lookback_days
        self._fetch_workers = max(1, fetch_workers)

    def select(self, as_of: date) -> Sequence[str]:
        universe = self._universe_repository.list_symbols()
        if not universe:
            logger.warning("Universe is empty")
            return []

        if self._liquidity_filter is not None:
            universe = self._liquidity_filter.filter(universe, as_of=as_of)

        if not universe:
            logger.warning("Universe empty after liquidity filter")
            return []

        candles = self._collect_daily_candles(universe, as_of)
        if not candles:
            logger.warning("No daily candles collected for selection")
            return []

        if self._selection_backend and self._selection_backend.available():
            export = getattr(self._strategy, "export_config", None)
            if callable(export):
                try:
                    config = export()
                    config["lookback_days"] = self._lookback_days
                    selected = self._selection_backend.select(candles, config)
                    if selected:
                        return selected
                except Exception as exc:
                    logger.warning("Native selection backend failed; falling back to Python", extra={"error": str(exc)})

        return self._strategy.select(universe, candles)

    def _collect_daily_candles(self, symbols: Iterable[str], as_of: date) -> list[Candle]:
        start = as_of - timedelta(days=max(self._lookback_days * 3, self._lookback_days + 10))
        end = as_of + timedelta(days=1)
        collected: list[Candle] = []

        symbols_list = list(symbols)
        if self._fetch_workers <= 1 or len(symbols_list) <= 1:
            for symbol in symbols_list:
                candles = self._fetch_daily_candles(symbol, start, end)
                if not candles:
                    logger.debug("No daily candles", extra={"symbol": symbol})
                    continue
                collected.extend(candles)
        else:
            logger.debug(
                "Fetching daily candles in parallel",
                extra={"symbols": len(symbols_list), "workers": self._fetch_workers},
            )
            with ThreadPoolExecutor(max_workers=self._fetch_workers) as executor:
                future_map = {
                    executor.submit(self._fetch_daily_candles, symbol, start, end): symbol
                    for symbol in symbols_list
                }
                for future in as_completed(future_map):
                    symbol = future_map[future]
                    try:
                        candles = future.result()
                    except Exception as exc:  # pragma: no cover - network dependent
                        logger.warning("Daily candle fetch failed", extra={"symbol": symbol, "error": str(exc)})
                        continue
                    if not candles:
                        logger.debug("No daily candles", extra={"symbol": symbol})
                        continue
                    collected.extend(candles)

        logger.debug("Collected daily candles", extra={"count": len(collected), "symbols": len(symbols_list)})
        return collected

    def _fetch_daily_candles(self, symbol: str, start: date, end: date) -> Sequence[Candle]:
        if self._candle_repository is not None:
            cached = self._candle_repository.get_daily_candles(symbol, start, end)
            if cached:
                return cached

        candles = self._provider.fetch_daily_candles(symbol, start, end)
        if self._candle_repository is not None and candles:
            self._candle_repository.store_daily_candles(symbol, candles)
        return candles
