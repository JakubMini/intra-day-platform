from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from intraday_platform.application.use_cases.simulation_engine import EquityPoint, SimulationResult
from intraday_platform.domain.entities.portfolio import Portfolio
from intraday_platform.domain.entities.position import Position
from intraday_platform.domain.entities.trade import Trade
from intraday_platform.infrastructure.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class PerformanceReport:
    metrics: dict[str, float]
    per_stock: dict[str, dict[str, float]]
    commission_impact: dict[str, float]
    equity_curve: list[EquityPoint]


class PerformanceAnalyzer:
    def analyze(self, result: SimulationResult) -> PerformanceReport:
        portfolio = result.portfolio
        equity_curve = result.equity_curve
        closed_positions = result.closed_positions

        total_commission = self._sum_commission(portfolio.trades)
        total_pnl = portfolio.equity - portfolio.starting_cash
        realized_pnl = sum(position.realized_pnl for position in closed_positions)
        unrealized_pnl = self._unrealized_pnl(portfolio)
        gross_pnl = total_pnl + total_commission

        trades_count = len(closed_positions)
        win_rate = self._win_rate(closed_positions)
        trade_expectancy = realized_pnl / trades_count if trades_count > 0 else 0.0
        max_drawdown = self._max_drawdown(equity_curve)
        sharpe_ratio = self._sharpe_ratio(equity_curve)

        metrics = {
            "total_pnl": total_pnl,
            "realized_pnl": realized_pnl,
            "unrealized_pnl": unrealized_pnl,
            "gross_pnl": gross_pnl,
            "total_commission": total_commission,
            "win_rate": win_rate,
            "trade_expectancy": trade_expectancy,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "trades_count": float(trades_count),
        }

        per_stock = self._per_stock_metrics(closed_positions)
        commission_impact = self._commission_impact(total_commission, trades_count, gross_pnl)

        logger.debug("Performance analysis complete", extra={"trades": trades_count})
        return PerformanceReport(
            metrics=metrics,
            per_stock=per_stock,
            commission_impact=commission_impact,
            equity_curve=list(equity_curve),
        )

    def _sum_commission(self, trades: Iterable[Trade]) -> float:
        return sum(trade.commission for trade in trades)

    def _unrealized_pnl(self, portfolio: Portfolio) -> float:
        return sum(position.unrealized_pnl for position in portfolio.open_positions())

    def _win_rate(self, positions: Iterable[Position]) -> float:
        positions_list = list(positions)
        if not positions_list:
            return 0.0
        wins = sum(1 for position in positions_list if position.realized_pnl > 0)
        return wins / len(positions_list)

    def _max_drawdown(self, equity_curve: list[EquityPoint]) -> float:
        if len(equity_curve) < 2:
            return 0.0
        peak = equity_curve[0].equity
        max_dd = 0.0
        for point in equity_curve:
            peak = max(peak, point.equity)
            if peak == 0:
                continue
            drawdown = (peak - point.equity) / peak
            max_dd = max(max_dd, drawdown)
        return max_dd

    def _sharpe_ratio(self, equity_curve: list[EquityPoint]) -> float:
        if len(equity_curve) < 3:
            return 0.0
        returns: list[float] = []
        for prev, current in zip(equity_curve[:-1], equity_curve[1:]):
            if prev.equity == 0:
                continue
            returns.append((current.equity / prev.equity) - 1.0)
        if len(returns) < 2:
            return 0.0
        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / (len(returns) - 1)
        std_dev = variance**0.5
        if std_dev == 0:
            return 0.0
        return mean_return / std_dev

    def _per_stock_metrics(self, positions: Iterable[Position]) -> dict[str, dict[str, float]]:
        metrics: dict[str, dict[str, float]] = {}
        for position in positions:
            symbol = position.symbol
            data = metrics.setdefault(
                symbol,
                {"trades": 0.0, "wins": 0.0, "realized_pnl": 0.0},
            )
            data["trades"] += 1.0
            data["realized_pnl"] += position.realized_pnl
            if position.realized_pnl > 0:
                data["wins"] += 1.0

        for symbol, data in metrics.items():
            trades = data["trades"]
            wins = data["wins"]
            data["win_rate"] = wins / trades if trades > 0 else 0.0
            data["avg_pnl"] = data["realized_pnl"] / trades if trades > 0 else 0.0

        return metrics

    def _commission_impact(self, total_commission: float, trades_count: int, gross_pnl: float) -> dict[str, float]:
        commission_per_trade = total_commission / trades_count if trades_count > 0 else 0.0
        commission_pct_of_gross = (total_commission / gross_pnl) if gross_pnl != 0 else 0.0
        return {
            "total_commission": total_commission,
            "commission_per_trade": commission_per_trade,
            "commission_pct_of_gross": commission_pct_of_gross,
        }
