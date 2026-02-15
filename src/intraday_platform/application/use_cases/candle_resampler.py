from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from intraday_platform.domain.entities.candle import Candle
from intraday_platform.domain.value_objects.timeframe import Timeframe
from intraday_platform.infrastructure.logging import get_logger

logger = get_logger(__name__)

_TIMEFRAME_RULES = {
    Timeframe.ONE_MINUTE: "1min",
    Timeframe.TWO_MINUTE: "2min",
    Timeframe.FIVE_MINUTE: "5min",
    Timeframe.TEN_MINUTE: "10min",
    Timeframe.FIFTEEN_MINUTE: "15min",
    Timeframe.THIRTY_MINUTE: "30min",
    Timeframe.ONE_HOUR: "1h",
    Timeframe.TWO_HOUR: "2h",
}


def resample_candles(candles: Sequence[Candle], timeframe: Timeframe) -> list[Candle]:
    if not candles:
        return []

    if timeframe.seconds < 60:
        logger.warning("Sub-minute resampling is not supported with Yahoo data", extra={"timeframe": timeframe.value})
        return []

    rule = _TIMEFRAME_RULES.get(timeframe)
    if rule is None:
        logger.warning("Unsupported resample timeframe", extra={"timeframe": timeframe.value})
        return []

    df = pd.DataFrame(
        {
            "timestamp": [c.timestamp for c in candles],
            "open": [c.open for c in candles],
            "high": [c.high for c in candles],
            "low": [c.low for c in candles],
            "close": [c.close for c in candles],
            "volume": [c.volume for c in candles],
        }
    )
    df = df.set_index("timestamp").sort_index()

    ohlc = df["close"].resample(rule, label="right", closed="right").ohlc()
    ohlc["open"] = df["open"].resample(rule, label="right", closed="right").first()
    ohlc["high"] = df["high"].resample(rule, label="right", closed="right").max()
    ohlc["low"] = df["low"].resample(rule, label="right", closed="right").min()
    ohlc["close"] = df["close"].resample(rule, label="right", closed="right").last()

    volume = df["volume"].resample(rule, label="right", closed="right").sum()
    pv = (df["close"] * df["volume"]).resample(rule, label="right", closed="right").sum()
    vwap = (pv / volume.replace(0, np.nan)).ffill().fillna(0.0)

    ohlc["volume"] = volume.fillna(0.0)
    ohlc["vwap"] = vwap
    ohlc = ohlc.dropna(subset=["open", "high", "low", "close"]).reset_index()
    ohlc = ohlc[
        (ohlc["open"] > 0)
        & (ohlc["high"] > 0)
        & (ohlc["low"] > 0)
        & (ohlc["close"] > 0)
        & (ohlc["volume"] >= 0)
    ]

    return [
        Candle(
            symbol=candles[0].symbol,
            timestamp=row["timestamp"].to_pydatetime(),
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row["volume"]),
            timeframe=timeframe,
            vwap=float(row["vwap"]) if pd.notna(row["vwap"]) else None,
        )
        for _, row in ohlc.iterrows()
    ]
