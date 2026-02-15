from __future__ import annotations

from intraday_platform.config import MAX_POSITION_GBP, RISK_FRACTION_PER_TRADE
from intraday_platform.domain.ports.position_sizing import PositionSizer
from intraday_platform.domain.value_objects.position_sizing import (
    PositionSizeDecision,
    PositionSizingContext,
)
from intraday_platform.infrastructure.logging import get_logger

logger = get_logger(__name__)


class FixedRiskPositionSizer(PositionSizer):
    """
    Sizes positions based on a fixed fraction of available cash and confidence.
    """

    def __init__(
        self,
        risk_fraction: float = RISK_FRACTION_PER_TRADE,
        max_position_value: float = MAX_POSITION_GBP,
    ) -> None:
        self._risk_fraction = risk_fraction
        self._max_position_value = max_position_value

    def size_position(self, context: PositionSizingContext) -> PositionSizeDecision:
        if context.price <= 0:
            return PositionSizeDecision(symbol=context.symbol, quantity=0.0, notional=0.0, reason="invalid price")

        risk_fraction = context.risk_fraction if context.risk_fraction > 0 else self._risk_fraction
        risk_budget = context.available_cash * risk_fraction
        confidence_budget = risk_budget * max(context.confidence, 0.0)
        target_value = min(confidence_budget, self._max_position_value, context.max_position_value)

        quantity = target_value / context.price
        notional = quantity * context.price

        if quantity <= 0:
            return PositionSizeDecision(
                symbol=context.symbol,
                quantity=0.0,
                notional=0.0,
                reason="insufficient cash for sizing",
            )

        logger.debug(
            "Position sized",
            extra={"symbol": context.symbol, "quantity": quantity, "notional": notional},
        )
        return PositionSizeDecision(symbol=context.symbol, quantity=quantity, notional=notional)
