# Intraday Platform Architecture

## Principles
- Clean architecture with strict dependency direction: presentation -> application -> domain
- Domain is framework-agnostic and contains core entities, value objects, and strategy contracts
- Infrastructure provides implementations for data providers, repositories, and exporters
- All IO is abstracted behind ports (interfaces) for easy replacement

## Python vs C++ Execution
- **Python engine**: reference implementation for all strategies and orchestration.
- **C++ native engine**: optional high‑performance path for simulation and daily selection.
- The Streamlit app detects the native binary and uses it automatically when available.
- If the native engine fails, the system **falls back to Python**.

## Package Structure
- src/intraday_platform
- src/intraday_platform/domain
- src/intraday_platform/domain/entities
- src/intraday_platform/domain/value_objects
- src/intraday_platform/domain/ports
- src/intraday_platform/application
- src/intraday_platform/application/ports
- src/intraday_platform/application/use_cases
- src/intraday_platform/infrastructure
- src/intraday_platform/infrastructure/providers
- src/intraday_platform/infrastructure/repositories
- src/intraday_platform/infrastructure/exporters
- src/intraday_platform/presentation
- cpp (native engine)

## Layer Responsibilities
- Domain: core business rules, data models, and strategy contracts
- Application: use cases such as simulation, selection, and analytics orchestration
- Infrastructure: adapters for market data, caching, and CSV export
- Presentation: Streamlit dashboard and CLI entrypoints

## Data Flow (High Level)
- Presentation triggers a simulation or optimization run
- Application orchestrates data fetch, stock selection, strategy signals, and execution
- Domain models represent the state and trades
- Infrastructure persists outputs and provides data adapters

## Native Engine Integration
- `NativeEngineRunner` executes the C++ binary for simulation.
- `NativeSelectionBackend` executes the C++ binary for daily universe selection.
- CSV is used as the IPC boundary (candles/config -> trades/equity/selection).
- Debug bundles are emitted on native failures to `data/native_debug/`.

## Optimization & Parallelism
- Daily universe selection uses threaded Python fetches.
- Optimization runs are parallelized using `ThreadPoolExecutor`.
- Native engine accelerates per‑config simulation when available.
