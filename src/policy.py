"""Allocation policy: turn per-stock signals into portfolio weights.

The paper jointly learns the signal and allocation to maximize risk-adjusted
return. For the classical benchmarks we use a simple, deterministic allocation:
cross-sectionally normalize the signals so the book is dollar-neutral and scaled
to a target gross leverage. The deep-learning model (models.py) replaces this
with a learned allocation head trained on a Sharpe objective (see losses there).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def signals_to_weights(
    signals: pd.DataFrame,
    max_gross_leverage: float = 1.0,
) -> pd.DataFrame:
    """Dollar-neutral, gross-leverage-constrained weights from raw signals.

    For each day: demean across stocks (market-neutral), then scale so the sum
    of absolute weights equals ``max_gross_leverage``.
    """
    s = signals.subtract(signals.mean(axis=1), axis=0)  # dollar-neutral
    gross = s.abs().sum(axis=1)
    gross = gross.replace(0, np.nan)
    w = s.div(gross, axis=0) * max_gross_leverage
    return w.fillna(0.0)
