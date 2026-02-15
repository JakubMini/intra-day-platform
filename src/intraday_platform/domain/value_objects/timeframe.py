from __future__ import annotations

from enum import Enum


class Timeframe(str, Enum):
    FIVE_SECOND = "5s"
    TEN_SECOND = "10s"
    FIFTEEN_SECOND = "15s"
    THIRTY_SECOND = "30s"
    ONE_MINUTE = "1m"
    TWO_MINUTE = "2m"
    FIVE_MINUTE = "5m"
    TEN_MINUTE = "10m"
    FIFTEEN_MINUTE = "15m"
    THIRTY_MINUTE = "30m"
    ONE_HOUR = "1h"
    TWO_HOUR = "2h"
    ONE_DAY = "1d"

    @property
    def seconds(self) -> int:
        mapping = {
            Timeframe.FIVE_SECOND: 5,
            Timeframe.TEN_SECOND: 10,
            Timeframe.FIFTEEN_SECOND: 15,
            Timeframe.THIRTY_SECOND: 30,
            Timeframe.ONE_MINUTE: 60,
            Timeframe.TWO_MINUTE: 120,
            Timeframe.FIVE_MINUTE: 300,
            Timeframe.TEN_MINUTE: 600,
            Timeframe.FIFTEEN_MINUTE: 900,
            Timeframe.THIRTY_MINUTE: 1800,
            Timeframe.ONE_HOUR: 3600,
            Timeframe.TWO_HOUR: 7200,
            Timeframe.ONE_DAY: 86400,
        }
        return mapping[self]
