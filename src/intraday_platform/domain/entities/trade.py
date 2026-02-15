from __future__ import annotations

from datetime import datetime

from pydantic import Field

from intraday_platform.domain.base import DomainModel
from intraday_platform.domain.value_objects.side import OrderSide


class Trade(DomainModel):
    trade_id: str
    symbol: str
    side: OrderSide
    quantity: float = Field(..., gt=0.0)
    price: float = Field(..., gt=0.0)
    timestamp: datetime
    commission: float = Field(..., ge=0.0)
    strategy_id: str
    signal_id: str | None = None
    realized_pnl: float | None = None

    def notional(self) -> float:
        return self.quantity * self.price
