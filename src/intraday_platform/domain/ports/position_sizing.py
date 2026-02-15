from __future__ import annotations

from abc import ABC, abstractmethod

from intraday_platform.domain.value_objects.position_sizing import (
    PositionSizeDecision,
    PositionSizingContext,
)


class PositionSizer(ABC):
    @abstractmethod
    def size_position(self, context: PositionSizingContext) -> PositionSizeDecision:
        raise NotImplementedError
