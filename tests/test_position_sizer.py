from __future__ import annotations

import pytest

from intraday_platform.application.use_cases.position_sizer import FixedRiskPositionSizer
from intraday_platform.domain.value_objects.position_sizing import PositionSizingContext


def test_position_sizer_respects_max_position() -> None:
    sizer = FixedRiskPositionSizer(risk_fraction=1.0, max_position_value=50.0)
    context = PositionSizingContext(
        symbol="TEST.L",
        price=10.0,
        confidence=1.0,
        available_cash=200.0,
        max_position_value=50.0,
        risk_fraction=1.0,
    )
    decision = sizer.size_position(context)
    assert decision.quantity == 5.0
    assert decision.notional == 50.0


def test_position_sizer_handles_invalid_price() -> None:
    sizer = FixedRiskPositionSizer(risk_fraction=1.0, max_position_value=50.0)
    with pytest.raises(ValueError):
        PositionSizingContext(
            symbol="TEST.L",
            price=0.0,
            confidence=1.0,
            available_cash=200.0,
            max_position_value=50.0,
            risk_fraction=1.0,
        )
