"""Classical signal models on residual time-series (paper benchmarks).

The signal function maps the recent cumulative-residual path of a stock to a
desired position in that residual. These are the non-deep-learning benchmarks
against which the CNN+Transformer is compared.

Convention: given a lookback window of residual returns, we form the cumulative
residual ("residual price") path and emit a position with sign opposite to the
recent deviation — i.e. mean-reversion: buy when the residual price is low,
sell when it is high.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _cumulative(residuals: pd.DataFrame) -> pd.DataFrame:
    """Cumulative residual 'price' process (integrated residual returns)."""
    return residuals.fillna(0).cumsum()


def mean_reversion_signal(
    residuals: pd.DataFrame,
    lookback: int = 30,
) -> pd.DataFrame:
    """Z-score mean-reversion position in each residual.

    For each day t and stock, compute the cumulative residual over the trailing
    ``lookback`` window, z-score it, and take position = -zscore (clipped).
    Positions are NaN where the residual itself is missing (stock not traded).
    """
    cum = _cumulative(residuals)
    window = cum.rolling(lookback)
    z = (cum - window.mean()) / window.std(ddof=0)
    signal = -z.clip(-3, 3)
    signal = signal.where(residuals.notna())
    return signal
