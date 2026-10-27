"""Performance metrics for a daily strategy return series."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def annualized_return(returns: pd.Series) -> float:
    return float(returns.mean() * TRADING_DAYS)


def annualized_vol(returns: pd.Series) -> float:
    return float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS))


def sharpe_ratio(returns: pd.Series) -> float:
    """Annualized Sharpe. Returns are assumed already in excess-of-RF terms."""
    sd = returns.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return 0.0
    return float(returns.mean() / sd * np.sqrt(TRADING_DAYS))


def max_drawdown(returns: pd.Series) -> float:
    equity = (1 + returns).cumprod()
    peak = equity.cummax()
    return float((equity / peak - 1).min())


def summary(returns: pd.Series, turnover: pd.Series | None = None) -> dict:
    out = {
        "ann_return": annualized_return(returns),
        "ann_vol": annualized_vol(returns),
        "sharpe": sharpe_ratio(returns),
        "max_drawdown": max_drawdown(returns),
        "n_days": int(returns.notna().sum()),
    }
    if turnover is not None:
        out["avg_daily_turnover"] = float(turnover.mean())
    return out
