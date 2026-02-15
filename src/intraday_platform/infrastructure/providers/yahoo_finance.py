from __future__ import annotations

from datetime import date, datetime
from typing import Sequence
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

from intraday_platform.application.ports.market_data import MarketDataProvider
from intraday_platform.config import MARKET_TIMEZONE
from intraday_platform.domain.entities.candle import Candle
from intraday_platform.domain.value_objects.timeframe import Timeframe
from intraday_platform.infrastructure.logging import get_logger

logger = get_logger(__name__)


def _normalize_dataframe(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if df.empty:
        return df

    if isinstance(df.columns, pd.MultiIndex):
        levels = df.columns.names
        if symbol in df.columns.get_level_values(1):
            df = df.xs(symbol, level=1, axis=1)
        elif symbol in df.columns.get_level_values(0):
            df = df.xs(symbol, level=0, axis=1)
        else:
            # fallback to first symbol in multiindex
            first_symbol = df.columns.get_level_values(-1)[0]
            df = df.xs(first_symbol, level=-1, axis=1)

    df = df.rename(columns=lambda col: col.lower().replace(" ", "_"))
    return df


def _to_timestamp(ts: pd.Timestamp) -> datetime:
    if ts.tzinfo is None:
        return ts.to_pydatetime().replace(tzinfo=ZoneInfo(MARKET_TIMEZONE))
    return ts.tz_convert(ZoneInfo(MARKET_TIMEZONE)).to_pydatetime()


def _frame_to_candles(
    df: pd.DataFrame,
    symbol: str,
    timeframe: Timeframe,
    price_scale: float,
) -> list[Candle]:
    if df.empty:
        return []
    candles: list[Candle] = []
    for index, row in df.iterrows():
        if pd.isna(row.get("open")) or pd.isna(row.get("high")) or pd.isna(row.get("low")) or pd.isna(row.get("close")):
            continue
        timestamp = _to_timestamp(index)
        open_price = float(row["open"]) * price_scale
        high_price = float(row["high"]) * price_scale
        low_price = float(row["low"]) * price_scale
        close_price = float(row["close"]) * price_scale
        volume = float(row.get("volume", 0.0))
        if (
            open_price <= 0
            or high_price <= 0
            or low_price <= 0
            or close_price <= 0
            or pd.isna(volume)
            or volume < 0
        ):
            continue
        candles.append(
            Candle(
                symbol=symbol,
                timestamp=timestamp,
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=volume,
                timeframe=timeframe,
                vwap=float(row["vwap"]) * price_scale if "vwap" in row and pd.notna(row["vwap"]) else None,
            )
        )
    return candles


class YahooFinanceProvider(MarketDataProvider):
    def __init__(self) -> None:
        self._tz = ZoneInfo(MARKET_TIMEZONE)
        self._currency_cache: dict[str, str] = {}
        self._scale_cache: dict[str, float] = {}

    def fetch_intraday_candles(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: Timeframe,
    ) -> Sequence[Candle]:
        interval = timeframe.value
        logger.debug("Fetching intraday candles", extra={"symbol": symbol, "interval": interval})
        try:
            df = yf.download(
                tickers=symbol,
                start=start,
                end=end,
                interval=interval,
                progress=False,
                auto_adjust=False,
            )
        except Exception as exc:  # pragma: no cover - network dependent
            logger.error("Failed to fetch intraday candles", extra={"symbol": symbol, "error": str(exc)})
            return []

        df = _normalize_dataframe(df, symbol)
        price_scale = self._price_scale(symbol, df)
        return _frame_to_candles(df, symbol, timeframe, price_scale)

    def fetch_daily_candles(self, symbol: str, start: date, end: date) -> Sequence[Candle]:
        logger.debug("Fetching daily candles", extra={"symbol": symbol})
        try:
            df = yf.download(
                tickers=symbol,
                start=start,
                end=end,
                interval="1d",
                progress=False,
                auto_adjust=False,
            )
        except Exception as exc:  # pragma: no cover - network dependent
            logger.error("Failed to fetch daily candles", extra={"symbol": symbol, "error": str(exc)})
            return []

        df = _normalize_dataframe(df, symbol)
        price_scale = self._price_scale(symbol, df)
        return _frame_to_candles(df, symbol, Timeframe.ONE_DAY, price_scale)

    def fetch_latest_candle(self, symbol: str) -> Candle | None:
        logger.debug("Fetching latest candle", extra={"symbol": symbol})
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="1d", interval="1m", auto_adjust=False)
        except Exception as exc:  # pragma: no cover - network dependent
            logger.error("Failed to fetch latest candle", extra={"symbol": symbol, "error": str(exc)})
            return None

        df = _normalize_dataframe(df, symbol)
        if df.empty:
            logger.warning("No latest candle returned", extra={"symbol": symbol})
            return None

        latest = df.iloc[-1:]
        price_scale = self._price_scale(symbol, df)
        candles = _frame_to_candles(latest, symbol, Timeframe.ONE_MINUTE, price_scale)
        return candles[0] if candles else None

    def _price_scale(self, symbol: str, df: pd.DataFrame) -> float:
        cached = self._scale_cache.get(symbol)
        if cached is not None:
            return cached

        currency = self._currency(symbol)
        if currency in {"GBp", "GBX"}:
            scale = 0.01
        elif currency in {"GBP"}:
            scale = 1.0
        else:
            scale = self._heuristic_scale(symbol, df)
        self._scale_cache[symbol] = scale
        return scale

    def _currency(self, symbol: str) -> str:
        if symbol in self._currency_cache:
            return self._currency_cache[symbol]
        currency = ""
        try:
            ticker = yf.Ticker(symbol)
            fast_info = getattr(ticker, "fast_info", None) or {}
            currency = fast_info.get("currency") or ""
            if not currency:
                info = getattr(ticker, "info", None) or {}
                currency = info.get("currency") or ""
        except Exception as exc:  # pragma: no cover - network dependent
            logger.warning("Failed to read currency info", extra={"symbol": symbol, "error": str(exc)})
        self._currency_cache[symbol] = currency
        return currency

    def _heuristic_scale(self, symbol: str, df: pd.DataFrame) -> float:
        if df.empty or "close" not in df.columns:
            return 1.0
        close = df["close"]
        if isinstance(close, pd.DataFrame):
            close_series = close.iloc[:, 0]
        else:
            close_series = close
        median_price = float(close_series.median())
        if median_price >= 1000:
            logger.warning(
                "Currency unknown; assuming GBp pricing",
                extra={"symbol": symbol, "median_price": median_price},
            )
            return 0.01
        return 1.0
