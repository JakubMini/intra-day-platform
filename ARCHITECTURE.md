# Intraday Platform Architecture

## Principles
- Clean architecture with strict dependency direction: presentation -> application -> domain
- Domain is framework-agnostic and contains core entities, value objects, and strategy contracts
- Infrastructure provides implementations for data providers, repositories, and exporters
- All IO is abstracted behind ports (interfaces) for easy replacement

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

## Layer Responsibilities
- Domain: core business rules, data models, and strategy contracts
- Application: use cases such as simulation, selection, and analytics orchestration
- Infrastructure: adapters for market data, caching, and CSV export
- Presentation: Streamlit dashboard and CLI entrypoints

## Planned Data Flow (High Level)
- Presentation triggers a simulation run
- Application orchestrates data fetch, stock selection, strategy signals, and execution
- Domain models represent the state and trades
- Infrastructure persists outputs and provides data adapters
