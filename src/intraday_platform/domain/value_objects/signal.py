from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import Field

from intraday_platform.domain.base import DomainModel
from intraday_platform.domain.value_objects.side import SignalAction


class Signal(DomainModel):
    signal_id: str = Field(..., description="Unique signal id")
    symbol: str
    action: SignalAction
    confidence: float = Field(..., ge=0.0, le=1.0)
    timestamp: datetime
    strategy_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)
