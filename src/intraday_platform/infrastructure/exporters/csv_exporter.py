from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable

from intraday_platform.application.ports.exporters import MetricsExporter, TradeExporter
from intraday_platform.domain.entities.trade import Trade
from intraday_platform.infrastructure.logging import get_logger

logger = get_logger(__name__)


def _flatten_metrics(metrics: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in metrics.items():
        full_key = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
        if isinstance(value, dict):
            flattened.update(_flatten_metrics(value, prefix=full_key))
        else:
            flattened[full_key] = value
    return flattened


class CsvTradeExporter(TradeExporter):
    def export_trades(self, trades: Iterable[Trade], destination: str) -> None:
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)

        rows = []
        for trade in trades:
            rows.append(
                {
                    "trade_id": trade.trade_id,
                    "symbol": trade.symbol,
                    "side": trade.side.value,
                    "quantity": trade.quantity,
                    "price": trade.price,
                    "timestamp": trade.timestamp.isoformat(),
                    "commission": trade.commission,
                    "strategy_id": trade.strategy_id,
                    "signal_id": trade.signal_id or "",
                    "realized_pnl": trade.realized_pnl if trade.realized_pnl is not None else "",
                    "notional": trade.notional(),
                }
            )

        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
            if rows:
                writer.writeheader()
                writer.writerows(rows)

        logger.debug("Exported trades", extra={"path": str(path), "count": len(rows)})


class CsvMetricsExporter(MetricsExporter):
    def export_metrics(self, metrics: dict[str, Any], destination: str) -> None:
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)

        flattened = _flatten_metrics(metrics)
        rows = [{"metric": key, "value": value} for key, value in flattened.items()]

        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
            writer.writeheader()
            writer.writerows(rows)

        logger.debug("Exported metrics", extra={"path": str(path), "count": len(rows)})
