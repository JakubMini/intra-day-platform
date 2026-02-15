from __future__ import annotations

from datetime import datetime

from pydantic import Field

from intraday_platform.domain.base import DomainModel
from intraday_platform.domain.entities.position import Position
from intraday_platform.domain.entities.trade import Trade


class Portfolio(DomainModel):
    starting_cash: float = Field(..., ge=0.0)
    cash: float = Field(..., ge=0.0)
    equity: float = Field(..., ge=0.0)
    positions: dict[str, Position] = Field(default_factory=dict)
    trades: list[Trade] = Field(default_factory=list)
    last_update: datetime | None = None

    def open_positions(self) -> list[Position]:
        return [position for position in self.positions.values() if position.is_open]
