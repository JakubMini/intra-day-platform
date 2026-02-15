from __future__ import annotations

from pydantic import Field

from intraday_platform.domain.base import DomainModel


class PositionSizingContext(DomainModel):
    symbol: str
    price: float = Field(..., gt=0.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    available_cash: float = Field(..., ge=0.0)
    max_position_value: float = Field(..., gt=0.0)
    risk_fraction: float = Field(..., gt=0.0, le=1.0)


class PositionSizeDecision(DomainModel):
    symbol: str
    quantity: float = Field(..., ge=0.0)
    notional: float = Field(..., ge=0.0)
    reason: str | None = None
