from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Sequence

import pandas as pd

from intraday_platform.application.ports.repositories import CandleRepository
from intraday_platform.domain.entities.candle import Candle
from intraday_platform.domain.value_objects.timeframe import Timeframe
from intraday_platform.infrastructure.logging import get_logger

logger = get_logger(__name__)


class FileCandleRepository(CandleRepository):
    def __init__(self, root_dir: Path) -> None:
        self._root_dir = root_dir
        self._root_dir.mkdir(parents=True, exist_ok=True)

    def get_intraday_candles(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: Timeframe,
    ) -> Sequence[Candle]:
        path = self._path_for(symbol, timeframe)
        return self._load_candles(path, start, end)

    def get_daily_candles(self, symbol: str, start: date, end: date) -> Sequence[Candle]:
        path = self._path_for(symbol, Timeframe.ONE_DAY)
        return self._load_candles(path, start, end)

    def store_intraday_candles(
        self,
        symbol: str,
        timeframe: Timeframe,
        candles: Sequence[Candle],
    ) -> None:
        if not candles:
            return
        path = self._path_for(symbol, timeframe)
        df = self._candles_to_frame(candles)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        logger.debug("Stored candles", extra={"path": str(path), "count": len(candles)})

    def store_daily_candles(self, symbol: str, candles: Sequence[Candle]) -> None:
        if not candles:
            return
        path = self._path_for(symbol, Timeframe.ONE_DAY)
        df = self._candles_to_frame(candles)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        logger.debug("Stored daily candles", extra={"path": str(path), "count": len(candles)})

    def _path_for(self, symbol: str, timeframe: Timeframe) -> Path:
        safe_symbol = symbol.replace("/", "_").replace(" ", "_")
        filename = f"{safe_symbol}_{timeframe.value}.csv"
        return self._root_dir / filename

    def _load_candles(
        self,
        path: Path,
        start: datetime | date,
        end: datetime | date,
    ) -> Sequence[Candle]:
        if not path.exists():
            return []
        df = pd.read_csv(path)
        timestamps = pd.to_datetime(df["timestamp"], utc=False)
        if getattr(timestamps.dt, "tz", None) is not None:
            timestamps = timestamps.dt.tz_localize(None)
        df["timestamp"] = timestamps
        if isinstance(start, date) and not isinstance(start, datetime):
            start_dt = datetime.combine(start, datetime.min.time())
            end_dt = datetime.combine(end, datetime.max.time())
        else:
            start_dt = start  # type: ignore[assignment]
            end_dt = end  # type: ignore[assignment]
        if isinstance(start_dt, datetime) and start_dt.tzinfo is not None:
            start_dt = start_dt.replace(tzinfo=None)
        if isinstance(end_dt, datetime) and end_dt.tzinfo is not None:
            end_dt = end_dt.replace(tzinfo=None)
        df = df[(df["timestamp"] >= start_dt) & (df["timestamp"] <= end_dt)]
        return [
            Candle(
                symbol=row["symbol"],
                timestamp=row["timestamp"].to_pydatetime(),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
                timeframe=Timeframe(row["timeframe"]),
                vwap=float(row["vwap"]) if pd.notna(row["vwap"]) else None,
            )
            for _, row in df.iterrows()
        ]

    def _candles_to_frame(self, candles: Sequence[Candle]) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "symbol": candle.symbol,
                    "timestamp": candle.timestamp,
                    "open": candle.open,
                    "high": candle.high,
                    "low": candle.low,
                    "close": candle.close,
                    "volume": candle.volume,
                    "timeframe": candle.timeframe.value,
                    "vwap": candle.vwap,
                }
                for candle in candles
            ]
        )
