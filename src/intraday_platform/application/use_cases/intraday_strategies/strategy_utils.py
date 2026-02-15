from __future__ import annotations

from datetime import datetime
from typing import Sequence
from uuid import uuid4

import pandas as pd
import numpy as np

from intraday_platform.domain.entities.candle import Candle


def generate_signal_id(strategy_id: str, symbol: str, timestamp: datetime) -> str:
    return f"{strategy_id}_{symbol}_{timestamp:%Y%m%d%H%M%S}_{uuid4().hex[:8]}"


def candles_to_frame(candles: Sequence[Candle]) -> pd.DataFrame:
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
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def compute_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def compute_sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def compute_std(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).std()


def compute_rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.rolling(window=period, min_periods=period).mean()
    avg_loss = losses.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(0.0)
    rsi = rsi.mask(avg_loss == 0, 100.0)
    rsi = rsi.mask(avg_gain == 0, 0.0)
    return rsi


def compute_vwap(df: pd.DataFrame) -> pd.Series:
    volume = df["volume"].astype("float64")
    price = df["close"].astype("float64")
    pv = (price * volume).cumsum()
    cum_vol = volume.cumsum().replace(0, np.nan)
    vwap = pv / cum_vol
    vwap = vwap.ffill()
    return vwap.fillna(0.0)
