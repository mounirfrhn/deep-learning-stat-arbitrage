"""Vectorized daily backtest of a residual-trading strategy.

Trading convention (no look-ahead): the weight formed at the close of day *t*
(using information up to *t*) is held over day *t+1* and earns the day-*t+1*
residual return. Transaction costs are charged on daily turnover.

We trade the residual returns directly (the conceptual version of the strategy).
Projecting residual positions back into tradeable stock positions is a known
extension; for a free-data methods reproduction, residual-space PnL is the
cleanest apples-to-apples comparison across signal models.
"""

from __future__ import annotations

import pandas as pd

from src import metrics


def run_backtest(
    weights: pd.DataFrame,
    residual_returns: pd.DataFrame,
    transaction_cost_bps: float = 5.0,
) -> dict:
    """Backtest a weight matrix against realized residual returns.

    Parameters
    ----------
    weights : (dates x stocks) target weights formed at each day's close.
    residual_returns : (dates x stocks) residual returns to be earned.
    transaction_cost_bps : round-trip cost per unit turnover, in basis points.

    Returns a dict with the daily strategy return series, turnover series, and
    a performance summary.
    """
    w = weights.reindex_like(residual_returns).fillna(0.0)

    # Position held over t+1 is the weight decided at t.
    w_held = w.shift(1).fillna(0.0)
    gross_ret = (w_held * residual_returns.fillna(0.0)).sum(axis=1)

    # Turnover = sum of absolute weight changes between consecutive days.
    turnover = (w - w.shift(1)).abs().sum(axis=1).fillna(0.0)
    cost = turnover * (transaction_cost_bps / 1e4)

    net_ret = (gross_ret - cost).rename("strategy_return")

    # Drop the warm-up region where weights are all zero.
    active = net_ret.loc[w_held.abs().sum(axis=1) > 0]

    return {
        "returns": net_ret,
        "gross_returns": gross_ret,
        "turnover": turnover,
        "summary": metrics.summary(active, turnover.loc[active.index]),
    }
