from __future__ import annotations

from typing import Callable

from intraday_platform.application.use_cases.intraday_strategies.bollinger_reversion import (
    BollingerReversionConfig,
    BollingerReversionStrategy,
)
from intraday_platform.application.use_cases.intraday_strategies.donchian_breakout import (
    DonchianBreakoutConfig,
    DonchianBreakoutStrategy,
)
from intraday_platform.application.use_cases.intraday_strategies.ema_crossover import (
    EmaCrossoverConfig,
    EmaCrossoverStrategy,
)
from intraday_platform.application.use_cases.intraday_strategies.kalman_trend import (
    KalmanTrendConfig,
    KalmanTrendStrategy,
)
from intraday_platform.application.use_cases.intraday_strategies.momentum_vwap import (
    MomentumVwapConfig,
    MomentumVwapStrategy,
)
from intraday_platform.application.use_cases.intraday_strategies.rsi_mean_reversion import (
    RsiMeanReversionConfig,
    RsiMeanReversionStrategy,
)
from intraday_platform.application.use_cases.intraday_strategies.zscore_reversion import (
    ZScoreReversionConfig,
    ZScoreReversionStrategy,
)
from intraday_platform.domain.ports.strategy import IntradayStrategy
from intraday_platform.infrastructure.logging import get_logger

logger = get_logger(__name__)


StrategyBuilder = Callable[[], IntradayStrategy]


class IntradayStrategyFactory:
    def __init__(self) -> None:
        self._registry: dict[str, StrategyBuilder] = {
            MomentumVwapStrategy.id: lambda: MomentumVwapStrategy(MomentumVwapConfig()),
            RsiMeanReversionStrategy.id: lambda: RsiMeanReversionStrategy(RsiMeanReversionConfig()),
            EmaCrossoverStrategy.id: lambda: EmaCrossoverStrategy(EmaCrossoverConfig()),
            BollingerReversionStrategy.id: lambda: BollingerReversionStrategy(BollingerReversionConfig()),
            DonchianBreakoutStrategy.id: lambda: DonchianBreakoutStrategy(DonchianBreakoutConfig()),
            ZScoreReversionStrategy.id: lambda: ZScoreReversionStrategy(ZScoreReversionConfig()),
            KalmanTrendStrategy.id: lambda: KalmanTrendStrategy(KalmanTrendConfig()),
        }

    def available(self) -> list[str]:
        return sorted(self._registry.keys())

    def create(self, strategy_id: str) -> IntradayStrategy:
        if strategy_id not in self._registry:
            logger.error("Unknown strategy id", extra={"strategy_id": strategy_id})
            raise ValueError(f"Unknown strategy id: {strategy_id}")
        return self._registry[strategy_id]()
