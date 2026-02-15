from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence

from intraday_platform.domain.entities.trade import Trade


class TradeExporter(ABC):
    @abstractmethod
    def export_trades(self, trades: Sequence[Trade], destination: str) -> None:
        raise NotImplementedError


class MetricsExporter(ABC):
    @abstractmethod
    def export_metrics(self, metrics: dict[str, Any], destination: str) -> None:
        raise NotImplementedError
