from __future__ import annotations

import pandas as pd

from intraday_platform.application.use_cases.intraday_strategies.strategy_utils import compute_vwap


def test_compute_vwap_positive() -> None:
    df = pd.DataFrame(
        {
            "close": [100.0, 101.0, 102.0, 103.0],
            "volume": [10.0, 10.0, 10.0, 10.0],
        }
    )
    vwap = compute_vwap(df)
    assert len(vwap) == 4
    assert (vwap >= 0).all()
