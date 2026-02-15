from __future__ import annotations

from datetime import datetime, timedelta

from intraday_platform.application.use_cases.performance_analyzer import PerformanceAnalyzer
from intraday_platform.application.use_cases.simulation_engine import EquityPoint, SimulationResult
from intraday_platform.domain.entities.portfolio import Portfolio
from intraday_platform.domain.entities.position import Position
from intraday_platform.domain.value_objects.side import PositionSide


def test_performance_analyzer_basic_metrics() -> None:
    start = datetime(2024, 1, 1, 9, 0, 0)
    equity_curve = [
        EquityPoint(timestamp=start, equity=100.0, cash=100.0),
        EquityPoint(timestamp=start + timedelta(minutes=1), equity=110.0, cash=110.0),
        EquityPoint(timestamp=start + timedelta(minutes=2), equity=105.0, cash=105.0),
    ]
    position = Position(
        position_id="pos1",
        symbol="TEST.L",
        side=PositionSide.LONG,
        quantity=1.0,
        entry_price=100.0,
        entry_time=start,
        current_price=105.0,
        strategy_id="test",
    )
    position.exit_price = 105.0
    position.exit_time = start + timedelta(minutes=2)
    position.realized_pnl = 5.0
    position.is_open = False
    portfolio = Portfolio(
        starting_cash=100.0,
        cash=105.0,
        equity=105.0,
        positions={},
        trades=[],
        last_update=start + timedelta(minutes=2),
    )
    result = SimulationResult(portfolio=portfolio, equity_curve=equity_curve, closed_positions=[position])

    report = PerformanceAnalyzer().analyze(result)
    assert report.metrics["total_pnl"] == 5.0
    assert report.metrics["win_rate"] == 1.0
    assert report.metrics["max_drawdown"] > 0.0
