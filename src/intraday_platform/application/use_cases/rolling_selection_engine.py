from __future__ import annotations

from datetime import date, datetime, time, timedelta
import heapq
from typing import Sequence
from zoneinfo import ZoneInfo

from intraday_platform.application.ports.market_data import MarketDataProvider
from intraday_platform.application.use_cases.daily_selection_engine import DailySelectionEngine
from intraday_platform.application.use_cases.simulation_engine import EquityPoint, SimulationResult
from intraday_platform.config import (
    COMMISSION_RATE,
    DEFAULT_TIMEFRAME,
    MARKET_TIMEZONE,
    MAX_POSITION_GBP,
    MAX_STOCKS_PER_DAY,
    RISK_FRACTION_PER_TRADE,
    STARTING_CAPITAL_GBP,
)
from intraday_platform.domain.entities.candle import Candle
from intraday_platform.domain.entities.portfolio import Portfolio
from intraday_platform.domain.entities.position import Position
from intraday_platform.domain.entities.trade import Trade
from intraday_platform.domain.ports.position_sizing import PositionSizer
from intraday_platform.domain.ports.strategy import IntradayStrategy
from intraday_platform.domain.value_objects.position_sizing import PositionSizingContext
from intraday_platform.domain.value_objects.side import OrderSide, PositionSide, SignalAction
from intraday_platform.infrastructure.logging import get_logger

logger = get_logger(__name__)


class RollingSelectionSimulationEngine:
    def __init__(
        self,
        provider: MarketDataProvider,
        selection_engine: DailySelectionEngine,
        strategy: IntradayStrategy,
        position_sizer: PositionSizer,
        commission_rate: float = COMMISSION_RATE,
        max_position_value: float = MAX_POSITION_GBP,
        risk_fraction: float = RISK_FRACTION_PER_TRADE,
        starting_cash: float = STARTING_CAPITAL_GBP,
        max_positions: int = MAX_STOCKS_PER_DAY,
        market_open: time = time(8, 0),
        market_close: time = time(16, 30),
    ) -> None:
        self._provider = provider
        self._selection_engine = selection_engine
        self._strategy = strategy
        self._position_sizer = position_sizer
        self._commission_rate = commission_rate
        self._max_position_value = max_position_value
        self._risk_fraction = risk_fraction
        self._starting_cash = starting_cash
        self._max_positions = max_positions
        self._market_open = market_open
        self._market_close = market_close
        self._tz = ZoneInfo(MARKET_TIMEZONE)

    def run(self, start_date: date, end_date: date) -> SimulationResult:
        portfolio = Portfolio(
            starting_cash=self._starting_cash,
            cash=self._starting_cash,
            equity=self._starting_cash,
            positions={},
            trades=[],
            last_update=None,
        )
        closed_positions: list[Position] = []
        equity_curve: list[EquityPoint] = []
        history: dict[str, list[Candle]] = {}

        current_date = start_date
        while current_date <= end_date:
            selected = list(self._selection_engine.select(as_of=current_date))
            open_symbols = set(portfolio.positions.keys())
            active_symbols = set(selected) | open_symbols

            if not active_symbols:
                current_date += timedelta(days=1)
                continue

            day_start = datetime.combine(current_date, self._market_open, tzinfo=self._tz)
            self._rebalance_for_selection(selected, portfolio, history, day_start, closed_positions)

            open_symbols = set(portfolio.positions.keys())
            active_symbols = set(selected) | open_symbols
            if not active_symbols:
                current_date += timedelta(days=1)
                continue

            day_end = datetime.combine(current_date, self._market_close, tzinfo=self._tz)
            candles_by_symbol = self._fetch_day_candles(active_symbols, day_start, day_end)

            self._process_day(candles_by_symbol, portfolio, history, closed_positions, equity_curve)
            current_date += timedelta(days=1)

        return SimulationResult(portfolio=portfolio, equity_curve=equity_curve, closed_positions=closed_positions)

    def _fetch_day_candles(
        self,
        symbols: Sequence[str],
        start_dt: datetime,
        end_dt: datetime,
    ) -> dict[str, list[Candle]]:
        candles_by_symbol: dict[str, list[Candle]] = {}
        for symbol in symbols:
            candles = self._provider.fetch_intraday_candles(symbol, start_dt, end_dt, timeframe=DEFAULT_TIMEFRAME)
            candles_by_symbol[symbol] = list(sorted(candles, key=lambda c: c.timestamp))
        return candles_by_symbol

    def _process_day(
        self,
        candles_by_symbol: dict[str, list[Candle]],
        portfolio: Portfolio,
        history: dict[str, list[Candle]],
        closed_positions: list[Position],
        equity_curve: list[EquityPoint],
    ) -> None:
        heap: list[tuple[datetime, str, int]] = []
        for symbol, candles in candles_by_symbol.items():
            if candles:
                heapq.heappush(heap, (candles[0].timestamp, symbol, 0))

        while heap:
            current_ts, symbol, index = heapq.heappop(heap)
            batch = [(symbol, index)]
            while heap and heap[0][0] == current_ts:
                _, sym, idx = heapq.heappop(heap)
                batch.append((sym, idx))

            for sym, idx in batch:
                candle = candles_by_symbol[sym][idx]
                history.setdefault(sym, []).append(candle)
                self._mark_to_market(portfolio, sym, candle.close, candle.timestamp)
                signals = self._strategy.generate_signals(sym, history[sym])
                self._process_signals(portfolio, closed_positions, signals, candle)

                next_idx = idx + 1
                if next_idx < len(candles_by_symbol[sym]):
                    next_candle = candles_by_symbol[sym][next_idx]
                    heapq.heappush(heap, (next_candle.timestamp, sym, next_idx))

            portfolio.equity = self._calculate_equity(portfolio)
            portfolio.last_update = current_ts
            equity_curve.append(EquityPoint(timestamp=current_ts, equity=portfolio.equity, cash=portfolio.cash))

    def _process_signals(
        self,
        portfolio: Portfolio,
        closed_positions: list[Position],
        signals: Sequence,
        candle: Candle,
    ) -> None:
        for signal in signals:
            if signal.action == SignalAction.BUY:
                self._enter_position(portfolio, signal, candle)
            elif signal.action in {SignalAction.EXIT, SignalAction.SELL}:
                self._exit_position(portfolio, closed_positions, signal, candle)

    def _enter_position(self, portfolio: Portfolio, signal, candle: Candle) -> None:
        if candle.symbol in portfolio.positions and portfolio.positions[candle.symbol].is_open:
            return
        if len(portfolio.open_positions()) >= self._max_positions:
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

    def _exit_position(self, portfolio: Portfolio, closed_positions: list[Position], signal, candle: Candle) -> None:
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

    def _rebalance_for_selection(
        self,
        selected: list[str],
        portfolio: Portfolio,
        history: dict[str, list[Candle]],
        timestamp: datetime,
        closed_positions: list[Position],
    ) -> None:
        open_positions = portfolio.open_positions()
        open_symbols = {position.symbol for position in open_positions}
        new_symbols = [symbol for symbol in selected if symbol not in open_symbols]

        while new_symbols and len(open_positions) > self._max_positions - len(new_symbols):
            non_selected = [position for position in open_positions if position.symbol not in selected]
            candidates = non_selected if non_selected else open_positions
            worst = min(candidates, key=lambda p: p.unrealized_pnl)
            last_price = self._last_price(worst.symbol, history) or worst.entry_price
            self._force_close(portfolio, closed_positions, worst, last_price, timestamp)
            open_positions = portfolio.open_positions()
            open_symbols = {position.symbol for position in open_positions}
            new_symbols = [symbol for symbol in selected if symbol not in open_symbols]

    def _force_close(
        self,
        portfolio: Portfolio,
        closed_positions: list[Position],
        position: Position,
        price: float,
        timestamp: datetime,
    ) -> None:
        notional = position.quantity * price
        commission = notional * self._commission_rate
        realized = (price - position.entry_price) * position.quantity
        realized -= (commission + position.commission_paid)

        position.exit_price = price
        position.exit_time = timestamp
        position.current_price = price
        position.realized_pnl = realized
        position.commission_paid += commission
        position.is_open = False

        portfolio.cash += notional - commission
        closed_positions.append(position)
        if position.symbol in portfolio.positions:
            del portfolio.positions[position.symbol]

        trade = Trade(
            trade_id=self._trade_id(position.symbol),
            symbol=position.symbol,
            side=OrderSide.SELL,
            quantity=position.quantity,
            price=price,
            timestamp=timestamp,
            commission=commission,
            strategy_id=position.strategy_id,
            signal_id=None,
            realized_pnl=realized,
        )
        portfolio.trades.append(trade)

    def _last_price(self, symbol: str, history: dict[str, list[Candle]]) -> float | None:
        candles = history.get(symbol)
        if not candles:
            return None
        return candles[-1].close

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
        from uuid import uuid4

        return f"pos_{symbol}_{uuid4().hex[:8]}"

    def _trade_id(self, symbol: str) -> str:
        from uuid import uuid4

        return f"trd_{symbol}_{uuid4().hex[:8]}"
