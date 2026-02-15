from __future__ import annotations

from datetime import datetime, timedelta

from intraday_platform.application.use_cases.position_sizer import FixedRiskPositionSizer
from intraday_platform.application.use_cases.simulation_engine import SimulationEngine
from intraday_platform.domain.entities.candle import Candle
from intraday_platform.domain.ports.strategy import IntradayStrategy
from intraday_platform.domain.value_objects.signal import Signal
from intraday_platform.domain.value_objects.side import SignalAction


class BuyThenExitStrategy(IntradayStrategy):
    id = "buy_then_exit"
    name = "Buy then Exit"

    def generate_signals(self, symbol: str, candles: list[Candle]) -> list[Signal]:
        if len(candles) == 1:
            return [
                Signal(
                    signal_id="sig_buy",
                    symbol=symbol,
                    action=SignalAction.BUY,
                    confidence=1.0,
                    timestamp=candles[-1].timestamp,
                    strategy_id=self.id,
                )
            ]
        if len(candles) == 2:
            return [
                Signal(
                    signal_id="sig_exit",
                    symbol=symbol,
                    action=SignalAction.EXIT,
                    confidence=0.5,
                    timestamp=candles[-1].timestamp,
                    strategy_id=self.id,
                )
            ]
        return []


def test_simulation_engine_executes_trades() -> None:
    start = datetime(2024, 1, 2, 9, 0, 0)
    candles = [
        Candle(symbol="TEST.L", timestamp=start, open=10, high=10, low=10, close=10, volume=100),
        Candle(
            symbol="TEST.L",
            timestamp=start + timedelta(minutes=1),
            open=12,
            high=12,
            low=12,
            close=12,
            volume=100,
        ),
    ]
    sizer = FixedRiskPositionSizer(risk_fraction=1.0, max_position_value=50.0)
    engine = SimulationEngine(
        provider=None,  # type: ignore[arg-type] - not used in run_from_candles
        strategy=BuyThenExitStrategy(),
        position_sizer=sizer,
        commission_rate=0.0,
        max_position_value=50.0,
        risk_fraction=1.0,
        starting_cash=100.0,
    )
    result = engine.run_from_candles({"TEST.L": candles})

    assert len(result.portfolio.trades) == 2
    assert len(result.closed_positions) == 1
    assert result.portfolio.cash > 100.0
