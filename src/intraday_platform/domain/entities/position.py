from __future__ import annotations

from datetime import datetime

from pydantic import Field

from intraday_platform.domain.base import DomainModel
from intraday_platform.domain.value_objects.side import PositionSide


class Position(DomainModel):
    position_id: str
    symbol: str
    side: PositionSide
    quantity: float = Field(..., gt=0.0)
    entry_price: float = Field(..., gt=0.0)
    entry_time: datetime
    current_price: float | None = Field(default=None, gt=0.0)
    exit_price: float | None = Field(default=None, gt=0.0)
    exit_time: datetime | None = None
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    commission_paid: float = 0.0
    strategy_id: str
    is_open: bool = True

    def market_value(self) -> float:
        price = self.current_price if self.current_price is not None else self.entry_price
        return self.quantity * price
