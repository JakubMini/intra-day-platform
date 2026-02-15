# Intraday LSE Simulation Platform

Research-grade intraday trading simulator for London Stock Exchange equities. The platform focuses on **repeatable experimentation** with modular strategy design, realistic transaction costs, and rich performance analytics. It is a simulation and research environment only (no live brokerage integration).

## What You Can Do
- Simulate intraday trading with **1‑minute candles** (resampled to other intervals).
- Select a daily universe of 5 stocks using a multi‑factor engine with liquidity filtering.
- Run multiple strategies, compare risk/timeframes, and **optimize** configurations.
- Track performance, commission impact, equity curves, and per‑stock statistics.
- Export trades and performance metrics to CSV.
- Use a Streamlit dashboard for interactive analysis.

## Highlights
- Clean Architecture (domain → application → infrastructure → presentation).
- Strategy, Adapter, Factory, and Repository patterns.
- Full Pydantic validation for domain entities.
- **Native C++ engine** for faster simulation and daily selection (optional).
- Parallelized universe selection and optimization runs.

---

## Quick Start

### 1. Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2. (Optional) Build Native Engine
```bash
bash scripts/build_native_engine.sh
```

### 3. Run the App
```bash
intraday_platform
```

---

## Usage Guide

### Analysis Modes (UI)
- **Single run**: One strategy + one timeframe + one risk level.
- **Compare strategies**: Run all strategies on the same dataset.
- **Compare timeframes**: Run one strategy across timeframes.
- **Compare risk levels**: Run one strategy across risk levels.
- **Optimize config**: Grid search across strategy + timeframe + risk.

### Notes on Data Availability
Yahoo Finance often limits **1‑minute data to ~7 days**. The app will attempt longer windows but data may be missing.

---

## Python vs C++ Execution

### Python Engine
- Always available.
- Uses the same strategy logic as the UI.
  
### C++ Native Engine (Optional)
- Faster simulation and daily universe selection.
- Invoked automatically when `cpp/build/native_engine` exists.
  
#### C++ Improvements
- Faster event loop for candle simulation.
- Native daily universe selection (multi‑factor ranking).
- Reduced Python overhead in optimization loops.

You can override the binary location:
```bash
export NATIVE_ENGINE_PATH=/path/to/native_engine
```

---

## Configuration
Project defaults live in `src/intraday_platform/config.py`.  
Key controls include:
- `STARTING_CAPITAL_GBP`
- `COMMISSION_RATE`
- `MAX_POSITION_GBP`
- `RISK_FRACTION_PER_TRADE`
- `DAILY_SELECTION_MAX_WORKERS`
- `OPTIMIZATION_MAX_WORKERS`

---

## Troubleshooting

### Enable Debug Logs
```bash
LOG_LEVEL=DEBUG intraday_platform
```

### Native Debug Dumps
When the native engine fails, candle/config dumps are saved to:
```
data/native_debug/
```

---

## Project Structure
```
src/intraday_platform/
  domain/        # Entities, value objects, strategy contracts
  application/   # Use cases (simulation, selection, analytics)
  infrastructure/# Adapters (Yahoo data, native engine, CSV exporters)
  presentation/ # Streamlit UI
cpp/             # Native C++ engine
```

See `ARCHITECTURE.md` for full design details.

---

## License
This repository is a research platform and is not a financial product. Use at your own risk.
