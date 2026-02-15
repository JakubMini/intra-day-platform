from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

from intraday_platform.application.ports.repositories import UniverseRepository
from intraday_platform.infrastructure.logging import get_logger

logger = get_logger(__name__)


class CsvUniverseRepository(UniverseRepository):
    def __init__(self, csv_path: Path, active_only: bool = True) -> None:
        self._csv_path = csv_path
        self._active_only = active_only

    def list_symbols(self) -> Sequence[str]:
        if not self._csv_path.exists():
            logger.error("Universe CSV not found", extra={"path": str(self._csv_path)})
            return []

        symbols: list[str] = []
        seen: set[str] = set()

        with self._csv_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                symbol = (row.get("symbol") or "").strip()
                if not symbol:
                    continue

                is_active = (row.get("is_active") or "true").strip().lower() in {"true", "1", "yes"}
                if self._active_only and not is_active:
                    continue

                if symbol in seen:
                    continue

                seen.add(symbol)
                symbols.append(symbol)

        logger.debug("Loaded universe symbols", extra={"count": len(symbols)})
        if len(symbols) < 50:
            logger.warning(
                "Universe appears small; replace CSV with full LSE universe",
                extra={"count": len(symbols)},
            )
        return symbols
