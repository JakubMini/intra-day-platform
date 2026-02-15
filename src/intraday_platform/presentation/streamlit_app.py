from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, time, timedelta
from pathlib import Path
import time as time_module
from typing import Iterable
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from intraday_platform.application.use_cases import (
    resample_candles,
    DailySelectionEngine,
    FixedRiskPositionSizer,
    IntradayStrategyFactory,
    LiquidityFilter,
    MultiFactorDailySelectionStrategy,
    MultiStrategySimulationEngine,
    PerformanceAnalyzer,
    RollingSelectionSimulationEngine,
    SimulationEngine,
)
from intraday_platform.config import (
    COMMISSION_RATE,
    MARKET_TIMEZONE,
    MAX_POSITION_GBP,
    MAX_STOCKS_PER_DAY,
    OPTIMIZATION_MAX_WORKERS,
    RISK_FRACTION_PER_TRADE,
    STARTING_CAPITAL_GBP,
)
from intraday_platform.domain.entities.candle import Candle
from intraday_platform.domain.value_objects.side import SignalAction
from intraday_platform.domain.value_objects.timeframe import Timeframe
from intraday_platform.infrastructure.exporters.csv_exporter import CsvMetricsExporter, CsvTradeExporter
from intraday_platform.infrastructure.logging import get_logger
from intraday_platform.infrastructure.native_engine import NativeEngineRunner, NativeSelectionBackend
from intraday_platform.infrastructure.providers.yahoo_finance import YahooFinanceProvider
from intraday_platform.infrastructure.repositories.universe_csv import CsvUniverseRepository

logger = get_logger(__name__)


def main() -> None:
    st.set_page_config(page_title="LSE Intraday Lab", layout="wide")
    _apply_theme()

    st.title("LSE Intraday Lab")
    st.caption("Intraday simulation and research dashboard for LSE equities.")

    controls = _render_sidebar_controls()
    if controls.run_requested:
        try:
            result_bundle = _run_simulation(controls)
        except RuntimeError as exc:
            logger.error("Simulation failed", extra={"error": str(exc)})
            st.error(f"Simulation failed: {exc}")
            result_bundle = None
        except Exception as exc:  # pragma: no cover - safety net for UI
            logger.error("Unexpected simulation error", extra={"error": str(exc)})
            st.error(f"Unexpected error: {exc}")
            result_bundle = None
        if result_bundle:
            st.session_state["simulation"] = result_bundle
    else:
        result_bundle = st.session_state.get("simulation")

    if result_bundle is None:
        if not controls.run_requested:
            st.info("Configure inputs and run a simulation to see results.")
        return

    _render_results(result_bundle, controls)


def _apply_theme() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap');
        html, body, [class*="css"]  { font-family: 'Space Grotesk', sans-serif; }
        code, pre { font-family: 'IBM Plex Mono', monospace; }
        .stApp {
            background: radial-gradient(circle at top left, #0f172a 0%, #0b1120 45%, #020617 100%);
            color: #e2e8f0;
        }
        .block-container { padding-top: 2rem; }
        .metric-card {
            background: rgba(15, 23, 42, 0.85);
            border: 1px solid rgba(148, 163, 184, 0.15);
            border-radius: 12px;
            padding: 1rem;
            box-shadow: 0 10px 30px rgba(2, 6, 23, 0.6);
        }
        .section-title {
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            color: #e2e8f0;
            margin-bottom: 0.4rem;
        }
        .stDataFrame, .stTable {
            background: rgba(15, 23, 42, 0.8);
        }
        .stButton>button {
            background: linear-gradient(120deg, #22d3ee, #6366f1);
            color: #0f172a;
            border: none;
            font-weight: 600;
        }
        .stButton>button:hover { opacity: 0.9; }
        </style>
        """,
        unsafe_allow_html=True,
    )


class ControlState:
    def __init__(
        self,
        run_requested: bool,
        analysis_mode: str,
        mode: str,
        start_date: date,
        end_date: date,
        selection_mode: str,
        manual_symbols: list[str],
        strategy_id: str,
        compare_all: bool,
        timeframe: Timeframe,
        compare_timeframes: bool,
        compare_risks: bool,
        optimize_config: bool,
        optimize_objective: str,
        risk_fraction: float,
        commission_rate: float,
        max_position: float,
        starting_cash: float,
        animate: bool,
        liquidity_filter: bool,
        fast_mode: bool,
    ) -> None:
        self.run_requested = run_requested
        self.analysis_mode = analysis_mode
        self.mode = mode
        self.start_date = start_date
        self.end_date = end_date
        self.selection_mode = selection_mode
        self.manual_symbols = manual_symbols
        self.strategy_id = strategy_id
        self.compare_all = compare_all
        self.timeframe = timeframe
        self.compare_timeframes = compare_timeframes
        self.compare_risks = compare_risks
        self.optimize_config = optimize_config
        self.optimize_objective = optimize_objective
        self.risk_fraction = risk_fraction
        self.commission_rate = commission_rate
        self.max_position = max_position
        self.starting_cash = starting_cash
        self.animate = animate
        self.liquidity_filter = liquidity_filter
        self.fast_mode = fast_mode


def _render_sidebar_controls() -> ControlState:
    with st.sidebar:
        st.subheader("Simulation Controls")

        analysis_mode = st.selectbox(
            "Analysis mode",
            options=[
                "Single run",
                "Compare strategies",
                "Compare timeframes",
                "Compare risk levels",
                "Optimize config",
            ],
            help="Pick one analysis type to avoid conflicting settings.",
        )

        mode = st.selectbox(
            "Execution mode",
            options=["Backtest (fast)", "Simulated live playback"],
            help="Simulated live playback replays candles with a short delay.",
        )
        if analysis_mode != "Single run":
            mode = "Backtest (fast)"

        today = date.today()
        default_start = today - timedelta(days=3)
        start_date = st.date_input("Start date", value=default_start)
        end_date = st.date_input("End date", value=today)

        selection_mode = st.radio("Universe", ["Daily Selection Engine", "Manual symbols"], horizontal=True)
        manual_symbols: list[str] = []
        if selection_mode == "Manual symbols":
            manual_symbols = _parse_symbol_input(
                st.text_input("Symbols (comma or space separated)", value="AZN.L, SHEL.L, HSBA.L")
            )

        strategy_factory = IntradayStrategyFactory()
        strategy_options = strategy_factory.available()
        default_index = strategy_options.index("momentum_vwap_v1") if "momentum_vwap_v1" in strategy_options else 0
        strategy_id = strategy_options[default_index]
        if analysis_mode in {"Single run", "Compare timeframes", "Compare risk levels"}:
            strategy_id = st.selectbox("Intraday strategy", options=strategy_options, index=default_index)

        compare_all = analysis_mode == "Compare strategies"

        timeframe = Timeframe.ONE_MINUTE
        if analysis_mode in {"Single run", "Compare risk levels"}:
            timeframe = st.selectbox(
                "Candle timeframe",
                options=_timeframe_options(),
                index=_timeframe_options().index(Timeframe.ONE_MINUTE),
                format_func=lambda tf: tf.value,
            )

        compare_timeframes = analysis_mode == "Compare timeframes"
        compare_risks = analysis_mode == "Compare risk levels"
        optimize_config = analysis_mode == "Optimize config"
        optimize_objective = "total_pnl"
        if optimize_config:
            optimize_objective = st.selectbox(
                "Optimization objective",
                options=["total_pnl", "sharpe_ratio", "win_rate", "trade_expectancy", "max_drawdown"],
                index=0,
                help="Select the metric used to rank configurations.",
            )

        risk_fraction = RISK_FRACTION_PER_TRADE
        if analysis_mode in {"Single run", "Compare timeframes"}:
            risk_fraction = st.slider(
                "Risk fraction per trade",
                min_value=0.01,
                max_value=1.0,
                value=RISK_FRACTION_PER_TRADE,
                step=0.01,
                format="%.2f",
            )

        with st.expander("Advanced settings", expanded=False):
            commission_rate = st.number_input(
                "Commission rate",
                value=COMMISSION_RATE,
                step=0.0001,
                format="%.4f",
            )
            max_position = st.number_input("Max position size (£)", value=MAX_POSITION_GBP, step=1.0)
            starting_cash = st.number_input("Starting capital (£)", value=STARTING_CAPITAL_GBP, step=10.0)
            liquidity_filter = st.toggle("Apply liquidity filter", value=True)
            fast_mode = st.toggle("Fast mode (skip idle timestamps)", value=True)
            if analysis_mode == "Single run":
                animate = st.toggle("Animate playback", value=True)
            else:
                animate = False
                st.caption("Animation is only available for single-run mode.")

        run_requested = st.button("Run simulation", type="primary")

    return ControlState(
        run_requested=run_requested,
        analysis_mode=analysis_mode,
        mode=mode,
        start_date=start_date,
        end_date=end_date,
        selection_mode=selection_mode,
        manual_symbols=manual_symbols,
        strategy_id=strategy_id,
        compare_all=compare_all,
        timeframe=timeframe,
        compare_timeframes=compare_timeframes,
        compare_risks=compare_risks,
        optimize_config=optimize_config,
        optimize_objective=optimize_objective,
        risk_fraction=risk_fraction,
        commission_rate=commission_rate,
        max_position=max_position,
        starting_cash=starting_cash,
        animate=animate,
        liquidity_filter=liquidity_filter,
        fast_mode=fast_mode,
    )


def _run_simulation(controls: ControlState):
    if controls.end_date < controls.start_date:
        st.error("End date must be after start date.")
        return None

    span_days = (controls.end_date - controls.start_date).days
    if span_days > 7:
        st.info(
            "Yahoo Finance 1-minute data is often limited to ~7 days. "
            "We will attempt the full range, but data may be missing."
        )

    provider = YahooFinanceProvider()
    native_runner = NativeEngineRunner()
    use_native = native_runner.available()
    native_selection = NativeSelectionBackend() if use_native else None

    selection_engine = None
    if controls.selection_mode == "Daily Selection Engine":
        universe_repo = CsvUniverseRepository(Path("data/universe/lse_universe.csv"))
        liquidity = LiquidityFilter(provider) if controls.liquidity_filter else None
        selector = MultiFactorDailySelectionStrategy()
        selection_engine = DailySelectionEngine(
            provider=provider,
            universe_repository=universe_repo,
            strategy=selector,
            liquidity_filter=liquidity,
            selection_backend=native_selection,
        )
        with st.spinner("Selecting daily universe..."):
            symbols = list(selection_engine.select(as_of=controls.end_date))
    else:
        symbols = controls.manual_symbols

    if len(symbols) > MAX_STOCKS_PER_DAY:
        st.warning(f"Limiting to top {MAX_STOCKS_PER_DAY} symbols.")
    symbols = symbols[:MAX_STOCKS_PER_DAY]
    if not symbols:
        st.error("No symbols selected. Adjust universe settings or enter manual symbols.")
        return None
    st.caption(f"Selected symbols: {', '.join(symbols)}")

    sizer = FixedRiskPositionSizer(
        risk_fraction=controls.risk_fraction,
        max_position_value=controls.max_position,
    )

    start_dt, end_dt = _build_market_datetimes(controls.start_date, controls.end_date)

    compare_flags = [
        controls.compare_all,
        controls.compare_timeframes,
        controls.compare_risks,
        controls.optimize_config,
    ]
    if sum(1 for flag in compare_flags if flag) > 1:
        st.error("Choose only one comparison mode at a time.")
        return None

    if controls.compare_timeframes:
        strategy = IntradayStrategyFactory().create(controls.strategy_id)
        engine = SimulationEngine(
            provider=provider,
            strategy=strategy,
            position_sizer=sizer,
            commission_rate=controls.commission_rate,
            max_position_value=controls.max_position,
            risk_fraction=controls.risk_fraction,
            starting_cash=controls.starting_cash,
        )
        with st.spinner("Running simulations across timeframes..."):
            base_candles = _fetch_base_candles(provider, symbols, start_dt, end_dt)
            if not _has_candles(base_candles):
                st.error("No intraday candles returned for the selected range.")
                return None

            analyzer = PerformanceAnalyzer()
            timeframe_results = {}
            timeframe_reports = {}
            for tf in _timeframe_compare_list():
                if tf.seconds < 60:
                    timeframe_results[tf] = None
                    continue
                resampled = {sym: resample_candles(candles, tf) for sym, candles in base_candles.items()}
                if not any(resampled.values()):
                    timeframe_results[tf] = None
                    continue
                if use_native:
                    try:
                        result = native_runner.run(
                            resampled,
                            strategy_id=controls.strategy_id,
                            commission_rate=controls.commission_rate,
                            max_position_value=controls.max_position,
                            risk_fraction=controls.risk_fraction,
                            starting_cash=controls.starting_cash,
                            max_positions=MAX_STOCKS_PER_DAY,
                        )
                    except RuntimeError as exc:
                        st.warning(f"Native engine failed; falling back to Python. {exc}")
                        result = engine.run_from_candles(resampled, fast_mode=controls.fast_mode)
                else:
                    result = engine.run_from_candles(resampled, fast_mode=controls.fast_mode)
                timeframe_results[tf] = result
                timeframe_reports[tf] = analyzer.analyze(result)

        return {
            "symbols": symbols,
            "timeframe_results": timeframe_results,
            "timeframe_reports": timeframe_reports,
            "start_dt": start_dt,
            "end_dt": end_dt,
            "provider": provider,
            "timeframe": controls.timeframe,
            "strategy_id": controls.strategy_id,
            "compare_timeframes": True,
            "compare_all": False,
            "compare_risks": False,
            "compare_carry": False,
        }

    if controls.compare_risks:
        strategy = IntradayStrategyFactory().create(controls.strategy_id)
        candles_by_symbol = _fetch_candles_for_timeframe(provider, symbols, start_dt, end_dt, controls.timeframe)
        if not _has_candles(candles_by_symbol):
            st.error("No intraday candles returned for the selected range.")
            return None

        analyzer = PerformanceAnalyzer()
        risk_results = {}
        risk_reports = {}
        for risk in _risk_compare_list():
            if use_native:
                try:
                    result = native_runner.run(
                        candles_by_symbol,
                        strategy_id=controls.strategy_id,
                        commission_rate=controls.commission_rate,
                        max_position_value=controls.max_position,
                        risk_fraction=risk,
                        starting_cash=controls.starting_cash,
                        max_positions=MAX_STOCKS_PER_DAY,
                    )
                except RuntimeError as exc:
                    st.warning(f"Native engine failed; falling back to Python. {exc}")
                    engine = SimulationEngine(
                        provider=provider,
                        strategy=strategy,
                        position_sizer=sizer,
                        commission_rate=controls.commission_rate,
                        max_position_value=controls.max_position,
                        risk_fraction=risk,
                        starting_cash=controls.starting_cash,
                    )
                    result = engine.run_from_candles(candles_by_symbol, fast_mode=controls.fast_mode)
            else:
                engine = SimulationEngine(
                    provider=provider,
                    strategy=strategy,
                    position_sizer=sizer,
                    commission_rate=controls.commission_rate,
                    max_position_value=controls.max_position,
                    risk_fraction=risk,
                    starting_cash=controls.starting_cash,
                )
                result = engine.run_from_candles(candles_by_symbol, fast_mode=controls.fast_mode)
            risk_results[risk] = result
            risk_reports[risk] = analyzer.analyze(result)

        return {
            "symbols": symbols,
            "risk_results": risk_results,
            "risk_reports": risk_reports,
            "start_dt": start_dt,
            "end_dt": end_dt,
            "provider": provider,
            "timeframe": controls.timeframe,
            "strategy_id": controls.strategy_id,
            "compare_risks": True,
            "compare_timeframes": False,
            "compare_all": False,
            "compare_carry": False,
        }

    if controls.optimize_config:
        strategies = IntradayStrategyFactory().available()
        timeframes = [tf for tf in _timeframe_compare_list() if tf.seconds >= 60]
        risks = _risk_compare_list()
        base_candles = _fetch_base_candles(provider, symbols, start_dt, end_dt)
        if not _has_candles(base_candles):
            st.error("No intraday candles returned for the selected range.")
            return None

        analyzer = PerformanceAnalyzer()
        rows = []

        candles_by_tf: dict[Timeframe, dict[str, list[Candle]]] = {}
        for tf in timeframes:
            if tf == Timeframe.ONE_MINUTE:
                candles_tf = base_candles
            else:
                candles_tf = {sym: resample_candles(candles, tf) for sym, candles in base_candles.items()}
            if _has_candles(candles_tf):
                candles_by_tf[tf] = candles_tf

        if not candles_by_tf:
            st.error("No intraday candles returned for the selected range.")
            return None

        def _run_config(strategy_id: str, tf: Timeframe, risk: float) -> dict[str, float | str]:
            candles_tf = candles_by_tf[tf]
            if use_native:
                try:
                    result = native_runner.run(
                        candles_tf,
                        strategy_id=strategy_id,
                        commission_rate=controls.commission_rate,
                        max_position_value=controls.max_position,
                        risk_fraction=risk,
                        starting_cash=controls.starting_cash,
                        max_positions=MAX_STOCKS_PER_DAY,
                    )
                except RuntimeError as exc:
                    logger.warning("Native engine failed during optimization", extra={"error": str(exc)})
                    strategy_obj = IntradayStrategyFactory().create(strategy_id)
                    engine = SimulationEngine(
                        provider=provider,
                        strategy=strategy_obj,
                        position_sizer=sizer,
                        commission_rate=controls.commission_rate,
                        max_position_value=controls.max_position,
                        risk_fraction=risk,
                        starting_cash=controls.starting_cash,
                    )
                    result = engine.run_from_candles(candles_tf, fast_mode=controls.fast_mode)
            else:
                strategy_obj = IntradayStrategyFactory().create(strategy_id)
                engine = SimulationEngine(
                    provider=provider,
                    strategy=strategy_obj,
                    position_sizer=sizer,
                    commission_rate=controls.commission_rate,
                    max_position_value=controls.max_position,
                    risk_fraction=risk,
                    starting_cash=controls.starting_cash,
                )
                result = engine.run_from_candles(candles_tf, fast_mode=controls.fast_mode)

            report = analyzer.analyze(result)
            score = _optimization_score(report.metrics, controls.optimize_objective)
            return {
                "strategy_id": strategy_id,
                "timeframe": tf.value,
                "risk_fraction": risk,
                "score": score,
                "total_pnl": report.metrics.get("total_pnl", 0.0),
                "sharpe_ratio": report.metrics.get("sharpe_ratio", 0.0),
                "win_rate": report.metrics.get("win_rate", 0.0),
                "trade_expectancy": report.metrics.get("trade_expectancy", 0.0),
                "max_drawdown": report.metrics.get("max_drawdown", 0.0),
                "trades": report.metrics.get("trades_count", 0),
            }

        tasks: list[tuple[str, Timeframe, float]] = []
        for tf in candles_by_tf:
            for strategy_id in strategies:
                for risk in risks:
                    tasks.append((strategy_id, tf, risk))

        total = len(tasks)
        progress = st.progress(0)
        status = st.empty()
        completed = 0
        max_workers = max(1, min(OPTIMIZATION_MAX_WORKERS, total))

        logger.debug("Optimization parallel run", extra={"tasks": total, "workers": max_workers})
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_run_config, sid, tf, risk): (sid, tf, risk) for sid, tf, risk in tasks}
            for future in as_completed(futures):
                sid, tf, risk = futures[future]
                completed += 1
                progress.progress(min(completed / total, 1.0))
                status.text(f"Optimizing: {sid} | {tf.value} | risk {risk:.2f} ({completed}/{total})")
                try:
                    rows.append(future.result())
                except Exception as exc:  # pragma: no cover - defensive
                    logger.warning("Optimization task failed", extra={"error": str(exc)})

        progress.empty()
        status.empty()

        if not rows:
            st.error("Optimization produced no results. Adjust date range or symbols.")
            return None

        results_df = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
        best_row = results_df.iloc[0].to_dict()

        return {
            "symbols": symbols,
            "optimization_results": results_df,
            "best_config": best_row,
            "objective": controls.optimize_objective,
            "start_dt": start_dt,
            "end_dt": end_dt,
            "provider": provider,
            "timeframe": controls.timeframe,
            "strategy_id": controls.strategy_id,
            "compare_all": False,
            "compare_timeframes": False,
            "compare_risks": False,
            "compare_carry": False,
            "optimize_config": True,
        }

    if (
        selection_engine is not None
        and not controls.compare_all
        and not controls.compare_timeframes
        and not controls.compare_risks
        and not controls.optimize_config
    ):
        strategy = IntradayStrategyFactory().create(controls.strategy_id)
        rolling_engine = RollingSelectionSimulationEngine(
            provider=provider,
            selection_engine=selection_engine,
            strategy=strategy,
            position_sizer=sizer,
            commission_rate=controls.commission_rate,
            max_position_value=controls.max_position,
            risk_fraction=controls.risk_fraction,
            starting_cash=controls.starting_cash,
            max_positions=MAX_STOCKS_PER_DAY,
        )
        with st.spinner("Running rolling daily selection simulation..."):
            result = rolling_engine.run(controls.start_date, controls.end_date)

        if not result.equity_curve:
            st.error("No intraday candles returned for the selected range. Try different dates or symbols.")
            return None

        report = PerformanceAnalyzer().analyze(result)

        return {
            "symbols": symbols,
            "result": result,
            "report": report,
            "start_dt": start_dt,
            "end_dt": end_dt,
            "provider": provider,
            "timeframe": controls.timeframe,
            "strategy_id": controls.strategy_id,
            "compare_all": False,
            "compare_timeframes": False,
            "compare_risks": False,
            "compare_carry": True,
        }

    if controls.compare_all:
        strategies = [IntradayStrategyFactory().create(sid) for sid in IntradayStrategyFactory().available()]
        multi_engine = MultiStrategySimulationEngine(
            provider=provider,
            strategies=strategies,
            position_sizer=sizer,
            commission_rate=controls.commission_rate,
            max_position_value=controls.max_position,
            risk_fraction=controls.risk_fraction,
            starting_cash=controls.starting_cash,
        )
        with st.spinner("Running simulations for all strategies..."):
            if _supports_yahoo_timeframe(controls.timeframe):
                base_candles = {
                    symbol: list(provider.fetch_intraday_candles(symbol, start_dt, end_dt, timeframe=controls.timeframe))
                    for symbol in symbols
                }
            else:
                base_candles = _fetch_base_candles(provider, symbols, start_dt, end_dt)
                base_candles = {sym: resample_candles(candles, controls.timeframe) for sym, candles in base_candles.items()}

            if not _has_candles(base_candles):
                st.error("No intraday candles returned for the selected range.")
                return None

            if use_native:
                results = {}
                for strategy_id in IntradayStrategyFactory().available():
                    try:
                        results[strategy_id] = native_runner.run(
                            base_candles,
                            strategy_id=strategy_id,
                            commission_rate=controls.commission_rate,
                            max_position_value=controls.max_position,
                            risk_fraction=controls.risk_fraction,
                            starting_cash=controls.starting_cash,
                            max_positions=MAX_STOCKS_PER_DAY,
                        )
                    except RuntimeError as exc:
                        st.warning(f"Native engine failed for {strategy_id}; falling back to Python. {exc}")
                        results = multi_engine.run_from_candles(base_candles)
                        break
            else:
                results = multi_engine.run_from_candles(base_candles)
        if not results:
            st.error("No intraday candles returned for the selected range. Try different dates or symbols.")
            return None

        analyzer = PerformanceAnalyzer()
        reports = {sid: analyzer.analyze(res) for sid, res in results.items()}
        return {
            "symbols": symbols,
            "results": results,
            "reports": reports,
            "start_dt": start_dt,
            "end_dt": end_dt,
            "provider": provider,
            "timeframe": controls.timeframe,
            "strategy_id": controls.strategy_id,
            "compare_all": True,
            "compare_risks": False,
            "compare_carry": False,
        }

    strategy = IntradayStrategyFactory().create(controls.strategy_id)
    engine = SimulationEngine(
        provider=provider,
        strategy=strategy,
        position_sizer=sizer,
        commission_rate=controls.commission_rate,
        max_position_value=controls.max_position,
        risk_fraction=controls.risk_fraction,
        starting_cash=controls.starting_cash,
    )

    with st.spinner("Running simulation..."):
        if _supports_yahoo_timeframe(controls.timeframe):
            candles_by_symbol = {
                symbol: list(provider.fetch_intraday_candles(symbol, start_dt, end_dt, timeframe=controls.timeframe))
                for symbol in symbols
            }
        else:
            base_candles = _fetch_base_candles(provider, symbols, start_dt, end_dt)
            candles_by_symbol = {
                sym: resample_candles(candles, controls.timeframe) for sym, candles in base_candles.items()
            }

        if not _has_candles(candles_by_symbol):
            st.error("No intraday candles returned for the selected range.")
            return None

        if use_native:
            try:
                result = native_runner.run(
                    candles_by_symbol,
                    strategy_id=controls.strategy_id,
                    commission_rate=controls.commission_rate,
                    max_position_value=controls.max_position,
                    risk_fraction=controls.risk_fraction,
                    starting_cash=controls.starting_cash,
                    max_positions=MAX_STOCKS_PER_DAY,
                )
            except RuntimeError as exc:
                st.warning(f"Native engine failed; falling back to Python. {exc}")
                result = engine.run_from_candles(candles_by_symbol, fast_mode=controls.fast_mode)
        else:
            result = engine.run_from_candles(candles_by_symbol, fast_mode=controls.fast_mode)

    if not result.equity_curve:
        st.error("No intraday candles returned for the selected range. Try different dates or symbols.")
        return None

    report = PerformanceAnalyzer().analyze(result)

    logger.debug("Simulation complete", extra={"symbols": symbols, "trades": len(result.portfolio.trades)})
    return {
        "symbols": symbols,
        "result": result,
        "report": report,
        "start_dt": start_dt,
        "end_dt": end_dt,
        "provider": provider,
        "timeframe": controls.timeframe,
        "strategy_id": controls.strategy_id,
        "compare_all": False,
        "compare_timeframes": False,
        "compare_risks": False,
        "compare_carry": False,
    }


def _build_market_datetimes(start_date: date, end_date: date) -> tuple[datetime, datetime]:
    tz = ZoneInfo(MARKET_TIMEZONE)
    start_dt = datetime.combine(start_date, time(8, 0), tzinfo=tz)
    end_dt = datetime.combine(end_date, time(16, 30), tzinfo=tz)
    return start_dt, end_dt


def _render_results(bundle: dict, controls: ControlState) -> None:
    if bundle.get("compare_all"):
        _render_strategy_comparison(bundle, controls)
        return
    if bundle.get("compare_timeframes"):
        _render_timeframe_comparison(bundle, controls)
        return
    if bundle.get("compare_risks"):
        _render_risk_comparison(bundle, controls)
        return
    if bundle.get("optimize_config"):
        _render_optimization(bundle)
        return

    report = bundle["report"]
    result = bundle["result"]
    symbols = bundle["symbols"]

    if not result.portfolio.trades:
        st.warning(
            "No trades executed. Try increasing risk fraction, lowering max position cap, "
            "or selecting lower-priced symbols."
        )
        diagnostics = _build_signal_diagnostics(
            bundle["provider"],
            bundle["strategy_id"],
            symbols,
            bundle["start_dt"],
            bundle["end_dt"],
        )
        if diagnostics is not None:
            st.markdown("<div class='section-title'>Signal Diagnostics</div>", unsafe_allow_html=True)
            st.dataframe(diagnostics, width="stretch", height=220)

    st.markdown("<div class='section-title'>Performance Overview</div>", unsafe_allow_html=True)
    _render_metrics(report)

    st.markdown("<div class='section-title'>Equity Curve</div>", unsafe_allow_html=True)
    equity_df = _equity_curve_df(report.equity_curve)
    if equity_df.empty:
        st.warning("No equity curve data available.")
        return
    if controls.mode == "Simulated live playback" and controls.animate:
        _animate_equity_curve(equity_df)
    else:
        st.plotly_chart(_equity_curve_chart(equity_df), width="stretch")

    st.markdown("<div class='section-title'>Trade Log</div>", unsafe_allow_html=True)
    trades_df = _trades_df(result.portfolio.trades)
    st.dataframe(trades_df, width="stretch", height=300)

    st.markdown("<div class='section-title'>Per-Stock Breakdown</div>", unsafe_allow_html=True)
    per_stock_df = pd.DataFrame(report.per_stock).T.reset_index().rename(columns={"index": "symbol"})
    st.dataframe(per_stock_df, width="stretch", height=240)

    st.markdown("<div class='section-title'>Commission Impact</div>", unsafe_allow_html=True)
    commission_df = pd.DataFrame([report.commission_impact])
    st.dataframe(commission_df, width="stretch", height=120)

    st.markdown("<div class='section-title'>Price Chart</div>", unsafe_allow_html=True)
    _render_price_chart(bundle)

    st.markdown("<div class='section-title'>Exports</div>", unsafe_allow_html=True)
    _render_exports(report, result)


def _render_metrics(report) -> None:
    metrics = report.metrics
    capital_usage = 0.0
    if report.equity_curve:
        latest = report.equity_curve[-1]
        if latest.equity > 0:
            capital_usage = 1 - (latest.cash / latest.equity)
    columns = st.columns(4)
    metrics_map = [
        ("Total PnL", metrics["total_pnl"], "£"),
        ("Win Rate", metrics["win_rate"] * 100, "%"),
        ("Max Drawdown", metrics["max_drawdown"] * 100, "%"),
        ("Sharpe", metrics["sharpe_ratio"], ""),
        ("Trades", metrics["trades_count"], ""),
        ("Commission", metrics["total_commission"], "£"),
        ("Gross PnL", metrics["gross_pnl"], "£"),
        ("Expectancy", metrics["trade_expectancy"], "£"),
        ("Capital Usage", capital_usage * 100, "%"),
    ]

    for index, (label, value, suffix) in enumerate(metrics_map):
        column = columns[index % 4]
        with column:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric(label, f"{value:,.2f}{suffix}")
            st.markdown("</div>", unsafe_allow_html=True)


def _equity_curve_df(points: Iterable) -> pd.DataFrame:
    return pd.DataFrame(
        [{"timestamp": p.timestamp, "equity": p.equity, "cash": p.cash} for p in points]
    )


def _equity_curve_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["equity"], mode="lines", name="Equity"))
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["cash"],
            mode="lines",
            name="Cash",
            line=dict(dash="dot"),
        )
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        height=360,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h"),
    )
    return fig


def _animate_equity_curve(df: pd.DataFrame) -> None:
    placeholder = st.empty()
    progress = st.progress(0)
    step = max(len(df) // 200, 1)
    for index in range(2, len(df) + 1, step):
        slice_df = df.iloc[:index]
        placeholder.plotly_chart(_equity_curve_chart(slice_df), width="stretch")
        progress.progress(min(index / len(df), 1.0))
        time_module.sleep(0.02)
    progress.progress(1.0)


def _trades_df(trades: Iterable) -> pd.DataFrame:
    rows = [
        {
            "trade_id": trade.trade_id,
            "symbol": trade.symbol,
            "side": trade.side.value,
            "quantity": trade.quantity,
            "price": trade.price,
            "timestamp": trade.timestamp,
            "commission": trade.commission,
            "strategy_id": trade.strategy_id,
            "realized_pnl": trade.realized_pnl,
        }
        for trade in trades
    ]
    columns = [
        "trade_id",
        "symbol",
        "side",
        "quantity",
        "price",
        "timestamp",
        "commission",
        "strategy_id",
        "realized_pnl",
    ]
    return pd.DataFrame(rows, columns=columns)


def _render_price_chart(bundle: dict) -> None:
    provider = bundle["provider"]
    symbols = bundle["symbols"]
    result = bundle["result"]
    start_dt = bundle["start_dt"]
    end_dt = bundle["end_dt"]
    timeframe = bundle.get("timeframe", Timeframe.ONE_MINUTE)

    symbol = st.selectbox("Symbol", options=symbols)
    candles = provider.fetch_intraday_candles(symbol, start_dt, end_dt, timeframe=Timeframe.ONE_MINUTE)
    if timeframe != Timeframe.ONE_MINUTE and timeframe.seconds >= 60:
        candles = resample_candles(candles, timeframe)
    if not candles:
        st.warning("No intraday candles available for this symbol.")
        return

    df = _candles_df(candles)
    trades_df = _trades_df(result.portfolio.trades)
    if trades_df.empty or "symbol" not in trades_df.columns:
        symbol_trades = trades_df
    else:
        symbol_trades = trades_df[trades_df["symbol"] == symbol]

    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df["timestamp"],
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                name="Price",
            )
        ]
    )

    if not symbol_trades.empty:
        buys = symbol_trades[symbol_trades["side"] == "buy"]
        sells = symbol_trades[symbol_trades["side"] == "sell"]

        fig.add_trace(
            go.Scatter(
                x=buys["timestamp"],
                y=buys["price"],
                mode="markers",
                marker=dict(symbol="triangle-up", size=10, color="#16a34a"),
                name="Buy",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=sells["timestamp"],
                y=sells["price"],
                mode="markers",
                marker=dict(symbol="triangle-down", size=10, color="#dc2626"),
                name="Sell",
            )
        )

    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        height=420,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig, width="stretch")


def _candles_df(candles: Iterable[Candle]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "timestamp": candle.timestamp,
                "open": candle.open,
                "high": candle.high,
                "low": candle.low,
                "close": candle.close,
                "volume": candle.volume,
            }
            for candle in candles
        ]
    )


def _render_exports(report, result) -> None:
    export_dir = st.text_input("Export directory", value="exports")
    export_path = Path(export_dir)

    if st.button("Export CSVs"):
        trade_exporter = CsvTradeExporter()
        metrics_exporter = CsvMetricsExporter()

        trades_path = export_path / "trades.csv"
        metrics_path = export_path / "metrics.csv"

        trade_exporter.export_trades(result.portfolio.trades, str(trades_path))
        metrics_payload = {
            "metrics": report.metrics,
            "commission": report.commission_impact,
            "per_stock": report.per_stock,
        }
        metrics_exporter.export_metrics(metrics_payload, str(metrics_path))
        st.success(f"Exported to {export_dir}")


def _render_strategy_comparison(bundle: dict, controls: ControlState) -> None:
    results = bundle["results"]
    reports = bundle["reports"]

    summary_rows = []
    for strategy_id, report in reports.items():
        metrics = report.metrics
        summary_rows.append(
            {
                "strategy_id": strategy_id,
                "total_pnl": metrics.get("total_pnl", 0.0),
                "win_rate": metrics.get("win_rate", 0.0),
                "max_drawdown": metrics.get("max_drawdown", 0.0),
                "sharpe_ratio": metrics.get("sharpe_ratio", 0.0),
                "trades": metrics.get("trades_count", 0.0),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("total_pnl", ascending=False)
    st.markdown("<div class='section-title'>Strategy Comparison</div>", unsafe_allow_html=True)
    st.dataframe(summary_df, width="stretch", height=240)

    strategy_options = summary_df["strategy_id"].tolist()
    selected = st.selectbox("Inspect strategy", options=strategy_options, index=0 if strategy_options else 0)
    if not selected:
        return

    report = reports[selected]
    result = results[selected]

    if not result.portfolio.trades:
        st.warning("No trades executed for this strategy.")

    st.markdown("<div class='section-title'>Performance Overview</div>", unsafe_allow_html=True)
    _render_metrics(report)

    st.markdown("<div class='section-title'>Equity Curve</div>", unsafe_allow_html=True)
    equity_df = _equity_curve_df(report.equity_curve)
    if equity_df.empty:
        st.warning("No equity curve data available.")
        return
    if controls.mode == "Simulated live playback" and controls.animate:
        _animate_equity_curve(equity_df)
    else:
        st.plotly_chart(_equity_curve_chart(equity_df), width="stretch")

    st.markdown("<div class='section-title'>Trade Log</div>", unsafe_allow_html=True)
    trades_df = _trades_df(result.portfolio.trades)
    st.dataframe(trades_df, width="stretch", height=300)

    st.markdown("<div class='section-title'>Per-Stock Breakdown</div>", unsafe_allow_html=True)
    per_stock_df = pd.DataFrame(report.per_stock).T.reset_index().rename(columns={"index": "symbol"})
    st.dataframe(per_stock_df, width="stretch", height=240)

    st.markdown("<div class='section-title'>Commission Impact</div>", unsafe_allow_html=True)
    commission_df = pd.DataFrame([report.commission_impact])
    st.dataframe(commission_df, width="stretch", height=120)

    st.markdown("<div class='section-title'>Price Chart</div>", unsafe_allow_html=True)
    bundle_selected = dict(bundle)
    bundle_selected["result"] = result
    bundle_selected["timeframe"] = bundle.get("timeframe")
    _render_price_chart(bundle_selected)

    st.markdown("<div class='section-title'>Exports</div>", unsafe_allow_html=True)
    _render_exports(report, result)


def _parse_symbol_input(raw: str) -> list[str]:
    tokens = [token.strip().upper() for token in raw.replace(",", " ").split() if token.strip()]
    unique: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if "." not in token:
            token = f"{token}.L"
        if token not in seen:
            seen.add(token)
            unique.append(token)
    return unique


def _timeframe_options() -> list[Timeframe]:
    return [
        Timeframe.ONE_MINUTE,
        Timeframe.TWO_MINUTE,
        Timeframe.FIVE_MINUTE,
        Timeframe.TEN_MINUTE,
        Timeframe.FIFTEEN_MINUTE,
        Timeframe.THIRTY_MINUTE,
        Timeframe.ONE_HOUR,
        Timeframe.TWO_HOUR,
    ]


def _timeframe_compare_list() -> list[Timeframe]:
    return [
        Timeframe.FIVE_SECOND,
        Timeframe.TEN_SECOND,
        Timeframe.FIFTEEN_SECOND,
        Timeframe.THIRTY_SECOND,
        Timeframe.ONE_MINUTE,
        Timeframe.TWO_MINUTE,
        Timeframe.FIVE_MINUTE,
        Timeframe.TEN_MINUTE,
        Timeframe.FIFTEEN_MINUTE,
        Timeframe.THIRTY_MINUTE,
        Timeframe.ONE_HOUR,
        Timeframe.TWO_HOUR,
    ]


def _supports_yahoo_timeframe(timeframe: Timeframe) -> bool:
    return timeframe in {
        Timeframe.ONE_MINUTE,
        Timeframe.TWO_MINUTE,
        Timeframe.FIVE_MINUTE,
        Timeframe.FIFTEEN_MINUTE,
        Timeframe.THIRTY_MINUTE,
        Timeframe.ONE_HOUR,
    }


def _risk_compare_list() -> list[float]:
    return [round(step / 10, 2) for step in range(1, 11)]


def _optimization_score(metrics: dict[str, float], objective: str) -> float:
    if objective == "max_drawdown":
        return -abs(metrics.get("max_drawdown", 0.0))
    value = metrics.get(objective)
    if value is None:
        return float("-inf")
    return float(value)


def _fetch_candles_for_timeframe(
    provider: YahooFinanceProvider,
    symbols: list[str],
    start_dt: datetime,
    end_dt: datetime,
    timeframe: Timeframe,
) -> dict[str, list[Candle]]:
    if _supports_yahoo_timeframe(timeframe):
        candles_by_symbol = {
            symbol: list(provider.fetch_intraday_candles(symbol, start_dt, end_dt, timeframe=timeframe))
            for symbol in symbols
        }
        counts = {sym: len(candles) for sym, candles in candles_by_symbol.items()}
        logger.debug(
            "Fetched candles for supported timeframe | timeframe=%s counts=%s",
            timeframe.value,
            counts,
        )
        return candles_by_symbol
    base_candles = _fetch_base_candles(provider, symbols, start_dt, end_dt)
    resampled = {sym: resample_candles(candles, timeframe) for sym, candles in base_candles.items()}
    counts = {sym: len(candles) for sym, candles in resampled.items()}
    logger.debug(
        "Resampled candles for timeframe | timeframe=%s counts=%s",
        timeframe.value,
        counts,
    )
    return resampled


def _fetch_base_candles(
    provider: YahooFinanceProvider,
    symbols: list[str],
    start_dt: datetime,
    end_dt: datetime,
) -> dict[str, list[Candle]]:
    candles_by_symbol = {}
    for symbol in symbols:
        candles = provider.fetch_intraday_candles(symbol, start_dt, end_dt, timeframe=Timeframe.ONE_MINUTE)
        candles_by_symbol[symbol] = list(candles)
        logger.debug(
            "Fetched base candles | symbol=%s count=%d start=%s end=%s",
            symbol,
            len(candles_by_symbol[symbol]),
            start_dt.isoformat(),
            end_dt.isoformat(),
        )
    return candles_by_symbol


def _has_candles(candles_by_symbol: dict[str, list[Candle]]) -> bool:
    return bool(candles_by_symbol) and any(len(candles) > 0 for candles in candles_by_symbol.values())


def _render_timeframe_comparison(bundle: dict, controls: ControlState) -> None:
    timeframe_reports = bundle["timeframe_reports"]
    timeframe_results = bundle["timeframe_results"]

    summary_rows = []
    for timeframe in _timeframe_compare_list():
        report = timeframe_reports.get(timeframe)
        if report is None:
            summary_rows.append(
                {
                    "timeframe": timeframe.value,
                    "status": "unsupported/empty",
                    "total_pnl": None,
                    "win_rate": None,
                    "max_drawdown": None,
                    "sharpe_ratio": None,
                    "trades": None,
                }
            )
            continue
        metrics = report.metrics
        summary_rows.append(
            {
                "timeframe": timeframe.value,
                "status": "ok",
                "total_pnl": metrics.get("total_pnl", 0.0),
                "win_rate": metrics.get("win_rate", 0.0),
                "max_drawdown": metrics.get("max_drawdown", 0.0),
                "sharpe_ratio": metrics.get("sharpe_ratio", 0.0),
                "trades": metrics.get("trades_count", 0.0),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    st.markdown("<div class='section-title'>Timeframe Comparison</div>", unsafe_allow_html=True)
    st.dataframe(summary_df, width="stretch", height=260)

    valid_timeframes = [tf for tf in _timeframe_compare_list() if timeframe_reports.get(tf) is not None]
    if not valid_timeframes:
        st.warning("No supported timeframes returned data.")
        return

    selected_tf = st.selectbox(
        "Inspect timeframe",
        options=valid_timeframes,
        format_func=lambda tf: tf.value,
    )
    report = timeframe_reports[selected_tf]
    result = timeframe_results[selected_tf]

    st.markdown("<div class='section-title'>Performance Overview</div>", unsafe_allow_html=True)
    _render_metrics(report)

    st.markdown("<div class='section-title'>Equity Curve</div>", unsafe_allow_html=True)
    equity_df = _equity_curve_df(report.equity_curve)
    if equity_df.empty:
        st.warning("No equity curve data available.")
        return
    if controls.mode == "Simulated live playback" and controls.animate:
        _animate_equity_curve(equity_df)
    else:
        st.plotly_chart(_equity_curve_chart(equity_df), width="stretch")

    st.markdown("<div class='section-title'>Trade Log</div>", unsafe_allow_html=True)
    trades_df = _trades_df(result.portfolio.trades)
    st.dataframe(trades_df, width="stretch", height=300)

    st.markdown("<div class='section-title'>Per-Stock Breakdown</div>", unsafe_allow_html=True)
    per_stock_df = pd.DataFrame(report.per_stock).T.reset_index().rename(columns={"index": "symbol"})
    st.dataframe(per_stock_df, width="stretch", height=240)

    st.markdown("<div class='section-title'>Commission Impact</div>", unsafe_allow_html=True)
    commission_df = pd.DataFrame([report.commission_impact])
    st.dataframe(commission_df, width="stretch", height=120)


def _render_risk_comparison(bundle: dict, controls: ControlState) -> None:
    risk_reports = bundle["risk_reports"]
    risk_results = bundle["risk_results"]

    summary_rows = []
    for risk in _risk_compare_list():
        report = risk_reports.get(risk)
        if report is None:
            summary_rows.append(
                {
                    "risk": f"{risk:.0%}",
                    "status": "missing",
                    "total_pnl": None,
                    "win_rate": None,
                    "max_drawdown": None,
                    "sharpe_ratio": None,
                    "trades": None,
                }
            )
            continue
        metrics = report.metrics
        summary_rows.append(
            {
                "risk": f"{risk:.0%}",
                "status": "ok",
                "total_pnl": metrics.get("total_pnl", 0.0),
                "win_rate": metrics.get("win_rate", 0.0),
                "max_drawdown": metrics.get("max_drawdown", 0.0),
                "sharpe_ratio": metrics.get("sharpe_ratio", 0.0),
                "trades": metrics.get("trades_count", 0.0),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    st.markdown("<div class='section-title'>Risk Comparison</div>", unsafe_allow_html=True)
    st.dataframe(summary_df, width="stretch", height=260)

    valid_risks = [risk for risk in _risk_compare_list() if risk_reports.get(risk) is not None]
    if not valid_risks:
        st.warning("No risk levels produced results.")
        return

    selected_risk = st.selectbox(
        "Inspect risk level",
        options=valid_risks,
        format_func=lambda r: f"{r:.0%}",
    )
    report = risk_reports[selected_risk]
    result = risk_results[selected_risk]

    st.markdown("<div class='section-title'>Performance Overview</div>", unsafe_allow_html=True)
    _render_metrics(report)

    st.markdown("<div class='section-title'>Equity Curve</div>", unsafe_allow_html=True)
    equity_df = _equity_curve_df(report.equity_curve)
    if equity_df.empty:
        st.warning("No equity curve data available.")
        return
    if controls.mode == "Simulated live playback" and controls.animate:
        _animate_equity_curve(equity_df)
    else:
        st.plotly_chart(_equity_curve_chart(equity_df), width="stretch")

    st.markdown("<div class='section-title'>Trade Log</div>", unsafe_allow_html=True)
    trades_df = _trades_df(result.portfolio.trades)
    st.dataframe(trades_df, width="stretch", height=300)

    st.markdown("<div class='section-title'>Per-Stock Breakdown</div>", unsafe_allow_html=True)
    per_stock_df = pd.DataFrame(report.per_stock).T.reset_index().rename(columns={"index": "symbol"})
    st.dataframe(per_stock_df, width="stretch", height=240)

    st.markdown("<div class='section-title'>Commission Impact</div>", unsafe_allow_html=True)
    commission_df = pd.DataFrame([report.commission_impact])
    st.dataframe(commission_df, width="stretch", height=120)


def _render_optimization(bundle: dict) -> None:
    results_df: pd.DataFrame = bundle["optimization_results"]
    best_config = bundle["best_config"]
    objective = bundle["objective"]

    st.markdown("<div class='section-title'>Optimization Summary</div>", unsafe_allow_html=True)
    st.write(f"Objective: `{objective}`")
    best_df = pd.DataFrame([best_config])
    st.dataframe(best_df, width="stretch", height=120)

    st.markdown("<div class='section-title'>Optimization Explorer</div>", unsafe_allow_html=True)
    df = results_df.copy()
    timeframes = sorted(df["timeframe"].unique().tolist())
    strategies = sorted(df["strategy_id"].unique().tolist())

    selected_timeframes = st.multiselect("Timeframes", options=timeframes, default=timeframes)
    selected_strategies = st.multiselect("Strategies", options=strategies, default=strategies)

    filtered = df[
        df["timeframe"].isin(selected_timeframes)
        & df["strategy_id"].isin(selected_strategies)
    ]
    if filtered.empty:
        st.warning("No configurations match the current filters.")
        return

    scatter = px.scatter(
        filtered,
        x="risk_fraction",
        y="score",
        color="strategy_id",
        symbol="timeframe",
        size="trades",
        hover_data=["total_pnl", "sharpe_ratio", "win_rate", "trade_expectancy", "max_drawdown"],
        labels={
            "risk_fraction": "Risk Fraction",
            "score": f"Score ({objective})",
            "strategy_id": "Strategy",
            "timeframe": "Timeframe",
        },
    )
    scatter.update_layout(
        template="plotly_dark",
        legend_title_text="Strategy",
        height=420,
    )
    st.plotly_chart(scatter, width="stretch")

    heat = filtered.groupby(["strategy_id", "timeframe"])["score"].max().reset_index()
    heat_pivot = heat.pivot(index="strategy_id", columns="timeframe", values="score").fillna(0.0)
    heatmap = px.imshow(
        heat_pivot,
        aspect="auto",
        color_continuous_scale="Viridis",
        labels={"color": f"Score ({objective})"},
    )
    heatmap.update_layout(template="plotly_dark", height=360)
    st.plotly_chart(heatmap, width="stretch")

    st.markdown("<div class='section-title'>All Configurations</div>", unsafe_allow_html=True)
    st.dataframe(filtered.sort_values("score", ascending=False), width="stretch", height=420)

    st.info("Run the best configuration separately to view charts and exports.")


def _build_signal_diagnostics(
    provider: YahooFinanceProvider,
    strategy_id: str,
    symbols: list[str],
    start_dt: datetime,
    end_dt: datetime,
) -> pd.DataFrame | None:
    try:
        strategy = IntradayStrategyFactory().create(strategy_id)
    except Exception as exc:
        logger.error("Failed to build strategy for diagnostics", extra={"error": str(exc)})
        return None

    rows = []
    for symbol in symbols:
        candles = provider.fetch_intraday_candles(symbol, start_dt, end_dt, timeframe=Timeframe.ONE_MINUTE)
        history: list[Candle] = []
        buy_signals = 0
        exit_signals = 0
        total_signals = 0
        for candle in candles:
            history.append(candle)
            signals = strategy.generate_signals(symbol, history)
            for signal in signals:
                total_signals += 1
                if signal.action == SignalAction.BUY:
                    buy_signals += 1
                elif signal.action in {SignalAction.EXIT, SignalAction.SELL}:
                    exit_signals += 1

        rows.append(
            {
                "symbol": symbol,
                "candles": len(candles),
                "signals": total_signals,
                "buys": buy_signals,
                "exits": exit_signals,
                "last_price": candles[-1].close if candles else None,
            }
        )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
