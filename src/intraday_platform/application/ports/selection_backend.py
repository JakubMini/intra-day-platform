from __future__ import annotations

from typing import Mapping, Protocol, Sequence

from intraday_platform.domain.entities.candle import Candle


Numeric = float | int


class SelectionBackend(Protocol):
    def available(self) -> bool:
        ...

    def select(self, candles: Sequence[Candle], config: Mapping[str, Numeric]) -> Sequence[str]:
        ...
