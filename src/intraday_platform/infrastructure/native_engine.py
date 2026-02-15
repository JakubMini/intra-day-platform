from __future__ import annotations

import csv
import os
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from intraday_platform.domain.entities.candle import Candle
from intraday_platform.domain.entities.portfolio import Portfolio
from intraday_platform.domain.entities.position import Position
from intraday_platform.domain.entities.trade import Trade
from intraday_platform.domain.value_objects.side import OrderSide, PositionSide
from intraday_platform.infrastructure.logging import get_logger
from intraday_platform.application.use_cases.simulation_engine import EquityPoint, SimulationResult
from intraday_platform.application.ports.selection_backend import SelectionBackend

logger = get_logger(__name__)


class NativeEngineRunner:
    def __init__(self, binary_path: Path | None = None) -> None:
        default_path = Path("cpp") / "build" / "native_engine"
        env_path = os.getenv("NATIVE_ENGINE_PATH")
        if env_path:
            self._binary_path = Path(env_path)
        else:
            self._binary_path = binary_path or default_path

    def available(self) -> bool:
        return self._binary_path.exists()

    def run(
        self,
        candles_by_symbol: dict[str, list[Candle]],
        strategy_id: str,
        commission_rate: float,
        max_position_value: float,
        risk_fraction: float,
        starting_cash: float,
        max_positions: int,
    ) -> SimulationResult:
        if not self.available():
            raise RuntimeError("Native engine binary not found")
        if not candles_by_symbol or not any(candles_by_symbol.values()):
            raise RuntimeError("No candle data available for native engine run")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            candles_path = tmp_path / "candles.csv"
            config_path = tmp_path / "config.txt"
            output_dir = tmp_path / "out"
            output_dir.mkdir(parents=True, exist_ok=True)

            rows_written = self._write_candles(candles_path, candles_by_symbol)
            if rows_written == 0:
                raise RuntimeError("No candle rows to send to native engine")
            self._write_config(
                config_path,
                strategy_id,
                commission_rate,
                max_position_value,
                risk_fraction,
                starting_cash,
                max_positions,
            )
            logger.debug(
                "Native engine config | strategy_id=%s commission_rate=%.6f max_position_value=%.4f risk_fraction=%.4f starting_cash=%.2f max_positions=%d",
                strategy_id,
                commission_rate,
                max_position_value,
                risk_fraction,
                starting_cash,
                max_positions,
            )

            cmd = [str(self._binary_path), "--config", str(config_path), "--candles", str(candles_path), "--output", str(output_dir)]
            logger.debug("Running native engine", extra={"cmd": " ".join(cmd)})
            try:
                result = subprocess.run(cmd, capture_output=True, text=True)
            except OSError as exc:
                logger.error("Failed to launch native engine", extra={"error": str(exc), "cmd": " ".join(cmd)})
                raise RuntimeError(f"Failed to launch native engine: {exc}") from exc
            if result.returncode != 0:
                stderr = result.stderr.strip()
                stdout = result.stdout.strip()
                preview = ""
                if candles_path.exists():
                    try:
                        with candles_path.open("r", encoding="utf-8") as handle:
                            preview_lines = []
                            for _ in range(6):
                                line = handle.readline()
                                if not line:
                                    break
                                preview_lines.append(line.rstrip())
                        preview = "\n".join(preview_lines)
                    except OSError:
                        preview = ""
                debug_dir = self._dump_debug_bundle(config_path, candles_path)
                debug_hint = f" debug_dump={debug_dir}" if debug_dir else ""
                error_message = (
                    "Native engine failed | code=%s rows_written=%s stderr=%s stdout=%s"
                    % (
                        result.returncode,
                        rows_written,
                        stderr or "<empty>",
                        stdout or "<empty>",
                    )
                )
                if debug_hint:
                    error_message = f"{error_message} |{debug_hint}"
                if preview:
                    error_message = f"{error_message}\\nCandles preview:\\n{preview}"
                logger.error(error_message)
                message = stderr or stdout or f"native_engine exited with code {result.returncode}"
                raise RuntimeError(message)

            trades = self._read_trades(output_dir / "trades.csv")
            equity_curve = self._read_equity(output_dir / "equity.csv")

        portfolio = self._build_portfolio(trades, equity_curve, starting_cash)
        closed_positions = self._build_closed_positions(trades)

        return SimulationResult(portfolio=portfolio, equity_curve=equity_curve, closed_positions=closed_positions)

    def _write_candles(self, path: Path, candles_by_symbol: dict[str, list[Candle]]) -> int:
        rows: list[list[str]] = []
        total_input = 0
        skipped_invalid = 0
        skipped_type = 0
        per_symbol_counts: dict[str, int] = {}
        for symbol, candles in candles_by_symbol.items():
            if not isinstance(candles, list):
                candles = list(candles)
            per_symbol_counts[symbol] = len(candles)
            for candle in candles:
                total_input += 1
                if not isinstance(candle, Candle):
                    skipped_type += 1
                    continue
                if (
                    candle.open <= 0
                    or candle.high <= 0
                    or candle.low <= 0
                    or candle.close <= 0
                    or candle.volume < 0
                ):
                    skipped_invalid += 1
                    continue
                ts_ms = int(candle.timestamp.timestamp() * 1000)
                rows.append(
                    [
                        symbol,
                        str(ts_ms),
                        f"{candle.open:.8f}",
                        f"{candle.high:.8f}",
                        f"{candle.low:.8f}",
                        f"{candle.close:.8f}",
                        f"{candle.volume:.8f}",
                        f"{candle.vwap:.8f}" if candle.vwap is not None else "",
                    ]
                )

        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["symbol", "timestamp", "open", "high", "low", "close", "volume", "vwap"])
            writer.writerows(rows)

        logger.debug(
            "Prepared candles for native engine | symbols=%d total_input=%d rows_written=%d skipped_invalid=%d skipped_type=%d",
            len(candles_by_symbol),
            total_input,
            len(rows),
            skipped_invalid,
            skipped_type,
        )
        if len(rows) == 0 or skipped_invalid > 0 or skipped_type > 0:
            zero_symbols = [symbol for symbol, count in per_symbol_counts.items() if count == 0]
            logger.warning(
                "Native engine candle preparation summary | symbols=%d total_input=%d rows_written=%d skipped_invalid=%d skipped_type=%d zero_symbol_count=%d zero_symbols=%s",
                len(candles_by_symbol),
                total_input,
                len(rows),
                skipped_invalid,
                skipped_type,
                len(zero_symbols),
                ",".join(zero_symbols[:10]),
            )
        return len(rows)

    def _write_config(
        self,
        path: Path,
        strategy_id: str,
        commission_rate: float,
        max_position_value: float,
        risk_fraction: float,
        starting_cash: float,
        max_positions: int,
    ) -> None:
        lines = [
            f"strategy_id={strategy_id}",
            f"commission_rate={commission_rate}",
            f"max_position_value={max_position_value}",
            f"risk_fraction={risk_fraction}",
            f"starting_cash={starting_cash}",
            f"max_positions={max_positions}",
        ]
        path.write_text("\n".join(lines), encoding="utf-8")

    def _dump_debug_bundle(self, config_path: Path, candles_path: Path) -> Path | None:
        debug_root = Path(os.getenv("NATIVE_ENGINE_DEBUG_DIR", "data/native_debug"))
        try:
            debug_root.mkdir(parents=True, exist_ok=True)
            stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            debug_dir = debug_root / f"native_fail_{stamp}"
            debug_dir.mkdir(parents=True, exist_ok=True)
            if config_path.exists():
                shutil.copy2(config_path, debug_dir / "config.txt")
            if candles_path.exists():
                shutil.copy2(candles_path, debug_dir / "candles.csv")
            if candles_path.exists():
                file_size = candles_path.stat().st_size
                logger.error("Native engine debug dump saved | path=%s size=%d", debug_dir, file_size)
            return debug_dir
        except OSError as exc:
            logger.error("Failed to persist native debug bundle | error=%s", exc)
            return None

    def _read_trades(self, path: Path) -> list[Trade]:
        trades: list[Trade] = []
        if not path.exists():
            return trades
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                trades.append(
                    Trade(
                        trade_id=row["trade_id"],
                        symbol=row["symbol"],
                        side=OrderSide(row["side"]),
                        quantity=float(row["quantity"]),
                        price=float(row["price"]),
                        timestamp=datetime.fromtimestamp(int(row["timestamp"]) / 1000),
                        commission=float(row["commission"]),
                        strategy_id=row["strategy_id"],
                        signal_id=row.get("signal_id") or None,
                        realized_pnl=float(row["realized_pnl"]) if row.get("realized_pnl") else None,
                    )
                )
        return trades

    def _read_equity(self, path: Path) -> list[EquityPoint]:
        points: list[EquityPoint] = []
        if not path.exists():
            return points
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                points.append(
                    EquityPoint(
                        timestamp=datetime.fromtimestamp(int(row["timestamp"]) / 1000),
                        equity=float(row["equity"]),
                        cash=float(row["cash"]),
                    )
                )
        return points

    def _build_portfolio(self, trades: list[Trade], equity_curve: list[EquityPoint], starting_cash: float) -> Portfolio:
        cash = equity_curve[-1].cash if equity_curve else starting_cash
        equity = equity_curve[-1].equity if equity_curve else starting_cash
        return Portfolio(
            starting_cash=starting_cash,
            cash=cash,
            equity=equity,
            positions={},
            trades=trades,
            last_update=equity_curve[-1].timestamp if equity_curve else None,
        )

    def _build_closed_positions(self, trades: Iterable[Trade]) -> list[Position]:
        positions: dict[str, Position] = {}
        closed: list[Position] = []
        for trade in trades:
            if trade.side == OrderSide.BUY:
                positions[trade.symbol] = Position(
                    position_id=f"pos_{trade.symbol}_{trade.trade_id}",
                    symbol=trade.symbol,
                    side=PositionSide.LONG,
                    quantity=trade.quantity,
                    entry_price=trade.price,
                    entry_time=trade.timestamp,
                    current_price=trade.price,
                    commission_paid=trade.commission,
                    strategy_id=trade.strategy_id,
                )
            else:
                position = positions.get(trade.symbol)
                if not position:
                    continue
                position.exit_price = trade.price
                position.exit_time = trade.timestamp
                position.current_price = trade.price
                position.realized_pnl = trade.realized_pnl or 0.0
                position.commission_paid += trade.commission
                position.is_open = False
                closed.append(position)
                positions.pop(trade.symbol, None)
        return closed


class NativeSelectionBackend(SelectionBackend):
    def __init__(self, binary_path: Path | None = None) -> None:
        default_path = Path("cpp") / "build" / "native_engine"
        env_path = os.getenv("NATIVE_ENGINE_PATH")
        if env_path:
            self._binary_path = Path(env_path)
        else:
            self._binary_path = binary_path or default_path

    def available(self) -> bool:
        return self._binary_path.exists()

    def select(self, candles: Sequence[Candle], config: Mapping[str, float | int]) -> Sequence[str]:
        if not self.available():
            raise RuntimeError("Native selection binary not found")
        if not candles:
            raise RuntimeError("No daily candles provided for native selection")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            candles_path = tmp_path / "daily_candles.csv"
            config_path = tmp_path / "config.txt"
            output_dir = tmp_path / "out"
            output_dir.mkdir(parents=True, exist_ok=True)

            rows_written = self._write_daily_candles(candles_path, candles)
            if rows_written == 0:
                raise RuntimeError("No candle rows to send to native selection")

            self._write_selection_config(config_path, config)

            cmd = [
                str(self._binary_path),
                "--config",
                str(config_path),
                "--candles",
                str(candles_path),
                "--output",
                str(output_dir),
            ]
            logger.debug("Running native selection", extra={"cmd": " ".join(cmd)})
            try:
                result = subprocess.run(cmd, capture_output=True, text=True)
            except OSError as exc:
                logger.error("Failed to launch native selection", extra={"error": str(exc), "cmd": " ".join(cmd)})
                raise RuntimeError(f"Failed to launch native selection: {exc}") from exc

            if result.returncode != 0:
                stderr = result.stderr.strip()
                stdout = result.stdout.strip()
                message = stderr or stdout or f"native_engine exited with code {result.returncode}"
                raise RuntimeError(message)

            selection_path = output_dir / "selection.csv"
            if not selection_path.exists():
                raise RuntimeError("Native selection did not produce selection.csv")

            return self._read_selection(selection_path)

    def _write_daily_candles(self, path: Path, candles: Sequence[Candle]) -> int:
        rows: list[list[str]] = []
        for candle in candles:
            if (
                candle.open <= 0
                or candle.high <= 0
                or candle.low <= 0
                or candle.close <= 0
                or candle.volume < 0
            ):
                continue
            ts_ms = int(candle.timestamp.timestamp() * 1000)
            rows.append(
                [
                    candle.symbol,
                    str(ts_ms),
                    f"{candle.open:.8f}",
                    f"{candle.high:.8f}",
                    f"{candle.low:.8f}",
                    f"{candle.close:.8f}",
                    f"{candle.volume:.8f}",
                    "",
                ]
            )
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["symbol", "timestamp", "open", "high", "low", "close", "volume", "vwap"])
            writer.writerows(rows)
        logger.debug(
            "Prepared daily candles for native selection | rows_written=%d symbols=%d",
            len(rows),
            len({c.symbol for c in candles}),
        )
        return len(rows)

    def _write_selection_config(self, path: Path, config: Mapping[str, float | int]) -> None:
        lines = [
            "mode=select",
            f"lookback_days={int(config.get('lookback_days', 20))}",
            f"recent_window={int(config.get('recent_window', 5))}",
            f"prior_window={int(config.get('prior_window', 10))}",
            f"breakout_window={int(config.get('breakout_window', 20))}",
            f"max_symbols={int(config.get('max_symbols', 5))}",
            f"weight_volume_surge={float(config.get('weight_volume_surge', 0.2))}",
            f"weight_volatility_expansion={float(config.get('weight_volatility_expansion', 0.2))}",
            f"weight_momentum={float(config.get('weight_momentum', 0.2))}",
            f"weight_breakout={float(config.get('weight_breakout', 0.2))}",
            f"weight_relative_strength={float(config.get('weight_relative_strength', 0.2))}",
        ]
        path.write_text("\n".join(lines), encoding="utf-8")

    def _read_selection(self, path: Path) -> list[str]:
        symbols: list[str] = []
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                symbol = row.get("symbol")
                if symbol:
                    symbols.append(symbol)
        return symbols
