from __future__ import annotations

from datetime import datetime
import heapq
from typing import Sequence
from uuid import uuid4

from intraday_platform.application.ports.market_data import MarketDataProvider
from intraday_platform.application.use_cases.simulation_engine import EquityPoint, SimulationResult
from intraday_platform.config import (
    COMMISSION_RATE,
    DEFAULT_TIMEFRAME,
    MAX_POSITION_GBP,
    RISK_FRACTION_PER_TRADE,
    STARTING_CAPITAL_GBP,
)
from intraday_platform.domain.entities.candle import Candle
from intraday_platform.domain.entities.portfolio import Portfolio
from intraday_platform.domain.entities.position import Position
from intraday_platform.domain.entities.trade import Trade
from intraday_platform.domain.ports.position_sizing import PositionSizer
from intraday_platform.domain.ports.strategy import IntradayStrategy
from intraday_platform.domain.value_objects.timeframe import Timeframe
from intraday_platform.domain.value_objects.position_sizing import PositionSizingContext
from intraday_platform.domain.value_objects.signal import Signal
from intraday_platform.domain.value_objects.side import OrderSide, PositionSide, SignalAction
from intraday_platform.infrastructure.logging import get_logger

logger = get_logger(__name__)


class MultiStrategySimulationEngine:
    def __init__(
        self,
        provider: MarketDataProvider,
        strategies: Sequence[IntradayStrategy],
        position_sizer: PositionSizer,
        commission_rate: float = COMMISSION_RATE,
        max_position_value: float = MAX_POSITION_GBP,
        risk_fraction: float = RISK_FRACTION_PER_TRADE,
        starting_cash: float = STARTING_CAPITAL_GBP,
    ) -> None:
        self._provider = provider
        self._strategies = list(strategies)
        self._position_sizer = position_sizer
        self._commission_rate = commission_rate
        self._max_position_value = max_position_value
        self._risk_fraction = risk_fraction
        self._starting_cash = starting_cash

    def run(
        self,
        symbols: Sequence[str],
        start: datetime,
        end: datetime,
        timeframe: Timeframe = DEFAULT_TIMEFRAME,
    ) -> dict[str, SimulationResult]:
        candles_by_symbol = self._fetch_candles(symbols, start, end, timeframe)
        return self.run_from_candles(candles_by_symbol)

    def run_from_candles(
        self,
        candles_by_symbol: dict[str, list[Candle]],
    ) -> dict[str, SimulationResult]:
        if not self._strategies:
            return {}

        symbols = list(candles_by_symbol.keys())
        portfolios: dict[str, Portfolio] = {
            strategy.id: Portfolio(
                starting_cash=self._starting_cash,
                cash=self._starting_cash,
                equity=self._starting_cash,
                positions={},
                trades=[],
                last_update=None,
            )
            for strategy in self._strategies
        }
        closed_positions: dict[str, list[Position]] = {strategy.id: [] for strategy in self._strategies}
        equity_curves: dict[str, list[EquityPoint]] = {strategy.id: [] for strategy in self._strategies}

        heap: list[tuple[datetime, str, int]] = []
        history: dict[str, list[Candle]] = {symbol: [] for symbol in symbols}

        for symbol, candles in candles_by_symbol.items():
            if candles:
                heapq.heappush(heap, (candles[0].timestamp, symbol, 0))

        if not heap:
            logger.warning("No candles available for multi-strategy simulation")
            return {
                strategy.id: SimulationResult(
                    portfolio=portfolios[strategy.id],
                    equity_curve=equity_curves[strategy.id],
                    closed_positions=closed_positions[strategy.id],
                )
                for strategy in self._strategies
            }

        while heap:
            current_ts, symbol, index = heapq.heappop(heap)
            batch = [(symbol, index)]
            while heap and heap[0][0] == current_ts:
                _, sym, idx = heapq.heappop(heap)
                batch.append((sym, idx))

            for sym, idx in batch:
                candle = candles_by_symbol[sym][idx]
                history[sym].append(candle)

                for strategy in self._strategies:
                    portfolio = portfolios[strategy.id]
                    self._mark_to_market(portfolio, sym, candle.close, candle.timestamp)
                    signals = strategy.generate_signals(sym, history[sym])
                    self._process_signals(portfolio, closed_positions[strategy.id], signals, candle)

                next_idx = idx + 1
                if next_idx < len(candles_by_symbol[sym]):
                    next_candle = candles_by_symbol[sym][next_idx]
                    heapq.heappush(heap, (next_candle.timestamp, sym, next_idx))

            for strategy in self._strategies:
                portfolio = portfolios[strategy.id]
                portfolio.equity = self._calculate_equity(portfolio)
                portfolio.last_update = current_ts
                equity_curves[strategy.id].append(
                    EquityPoint(timestamp=current_ts, equity=portfolio.equity, cash=portfolio.cash)
                )

        return {
            strategy.id: SimulationResult(
                portfolio=portfolios[strategy.id],
                equity_curve=equity_curves[strategy.id],
                closed_positions=closed_positions[strategy.id],
            )
            for strategy in self._strategies
        }

    def _fetch_candles(
        self,
        symbols: Sequence[str],
        start: datetime,
        end: datetime,
        timeframe: Timeframe,
    ) -> dict[str, list[Candle]]:
        candles_by_symbol: dict[str, list[Candle]] = {}
        for symbol in symbols:
            candles = self._provider.fetch_intraday_candles(symbol, start, end, timeframe)
            candles_by_symbol[symbol] = list(sorted(candles, key=lambda c: c.timestamp))
            logger.debug("Fetched intraday candles", extra={"symbol": symbol, "count": len(candles)})
        return candles_by_symbol

    def _process_signals(
        self,
        portfolio: Portfolio,
        closed_positions: list[Position],
        signals: Sequence[Signal],
        candle: Candle,
    ) -> None:
        for signal in signals:
            if signal.action == SignalAction.BUY:
                self._enter_position(portfolio, signal, candle)
            elif signal.action in {SignalAction.EXIT, SignalAction.SELL}:
                self._exit_position(portfolio, closed_positions, signal, candle)

    def _enter_position(self, portfolio: Portfolio, signal: Signal, candle: Candle) -> None:
        if candle.symbol in portfolio.positions and portfolio.positions[candle.symbol].is_open:
            return

        context = PositionSizingContext(
            symbol=candle.symbol,
            price=candle.close,
            confidence=signal.confidence,
            available_cash=portfolio.cash,
            max_position_value=self._max_position_value,
            risk_fraction=self._risk_fraction,
        )
        decision = self._position_sizer.size_position(context)
        if decision.quantity <= 0:
            return

        quantity = self._adjust_quantity_for_cash(decision.quantity, candle.close, portfolio.cash)
        if quantity <= 0:
            return

        notional = quantity * candle.close
        commission = notional * self._commission_rate
        total_cost = notional + commission
        if total_cost > portfolio.cash:
            return

        position = Position(
            position_id=self._position_id(candle.symbol),
            symbol=candle.symbol,
            side=PositionSide.LONG,
            quantity=quantity,
            entry_price=candle.close,
            entry_time=candle.timestamp,
            current_price=candle.close,
            commission_paid=commission,
            strategy_id=signal.strategy_id,
        )
        portfolio.positions[candle.symbol] = position
        portfolio.cash -= total_cost

        trade = Trade(
            trade_id=self._trade_id(candle.symbol),
            symbol=candle.symbol,
            side=OrderSide.BUY,
            quantity=quantity,
            price=candle.close,
            timestamp=candle.timestamp,
            commission=commission,
            strategy_id=signal.strategy_id,
            signal_id=signal.signal_id,
        )
        portfolio.trades.append(trade)

    def _exit_position(self, portfolio: Portfolio, closed_positions: list[Position], signal: Signal, candle: Candle) -> None:
        position = portfolio.positions.get(candle.symbol)
        if position is None or not position.is_open:
            return

        notional = position.quantity * candle.close
        commission = notional * self._commission_rate
        realized = (candle.close - position.entry_price) * position.quantity
        realized -= (commission + position.commission_paid)

        position.exit_price = candle.close
        position.exit_time = candle.timestamp
        position.current_price = candle.close
        position.realized_pnl = realized
        position.commission_paid += commission
        position.is_open = False

        portfolio.cash += notional - commission
        closed_positions.append(position)
        del portfolio.positions[candle.symbol]

        trade = Trade(
            trade_id=self._trade_id(candle.symbol),
            symbol=candle.symbol,
            side=OrderSide.SELL,
            quantity=position.quantity,
            price=candle.close,
            timestamp=candle.timestamp,
            commission=commission,
            strategy_id=signal.strategy_id,
            signal_id=signal.signal_id,
            realized_pnl=realized,
        )
        portfolio.trades.append(trade)

    def _mark_to_market(self, portfolio: Portfolio, symbol: str, price: float, timestamp: datetime) -> None:
        position = portfolio.positions.get(symbol)
        if position is None or not position.is_open:
            return
        position.current_price = price
        position.unrealized_pnl = (price - position.entry_price) * position.quantity
        portfolio.last_update = timestamp

    def _calculate_equity(self, portfolio: Portfolio) -> float:
        open_value = sum(position.market_value() for position in portfolio.open_positions())
        return portfolio.cash + open_value

    def _adjust_quantity_for_cash(self, quantity: float, price: float, cash: float) -> float:
        if quantity <= 0:
            return 0.0
        max_affordable = cash / (price * (1 + self._commission_rate))
        return min(quantity, max_affordable)

    def _position_id(self, symbol: str) -> str:
        return f"pos_{symbol}_{uuid4().hex[:8]}"

    def _trade_id(self, symbol: str) -> str:
        return f"trd_{symbol}_{uuid4().hex[:8]}"
