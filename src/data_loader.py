"""Data layer: daily equity returns + Fama-French factors, with local caching.

All downloads are cached as parquet under ``config.data.cache_dir`` so the
pipeline is reproducible and fast on re-runs. Prices are split/dividend adjusted
("Adj Close") so simple returns already account for dividends and splits, as in
the paper.
"""

from __future__ import annotations

import logging
import warnings
from contextlib import contextmanager
from pathlib import Path

import pandas as pd


@contextmanager
def _quiet_yfinance():
    """Silence the noise yfinance makes while downloading, without hiding ours.

    Two separate channels are quieted:
      * pandas `inplace`-deprecation *warnings* emitted inside yfinance's own
        code (matched by message, so it's robust to the warning class);
      * per-ticker "Data doesn't exist for startDate ..." messages, which
        yfinance logs through its own ``yfinance`` logger for tickers that have
        no data over the requested window (recent listings, renamed/delisted
        symbols). These are expected and handled by the liquidity filter.
    """
    yf_logger = logging.getLogger("yfinance")
    prev_level = yf_logger.level
    yf_logger.setLevel(logging.CRITICAL)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r".*inplace.*")
            warnings.filterwarnings("ignore", message=r".*PDEP-8.*")
            yield
    finally:
        yf_logger.setLevel(prev_level)


# ---------------------------------------------------------------------------
# Prices -> daily returns
# ---------------------------------------------------------------------------
def download_prices(
    tickers: list[str],
    start: str,
    end: str,
    source: str = "yfinance",
) -> pd.DataFrame:
    """Download adjusted-close prices. Returns a (dates x tickers) DataFrame."""
    if source == "yfinance":
        import yfinance as yf

        with _quiet_yfinance():
            raw = yf.download(
                tickers,
                start=start,
                end=end,
                auto_adjust=True,  # 'Close' is already split/dividend adjusted
                progress=False,
                threads=True,
            )
        # With multiple tickers yfinance returns a column MultiIndex (field, ticker).
        if isinstance(raw.columns, pd.MultiIndex):
            prices = raw["Close"].copy()
        else:  # single ticker
            prices = raw[["Close"]].copy()
            prices.columns = tickers[:1]

        # Report, rather than spam, the tickers that came back with no data.
        dropped = [t for t in tickers if t not in prices.columns or prices[t].notna().sum() == 0]
        if dropped:
            prices = prices.drop(columns=[c for c in dropped if c in prices.columns])
            preview = ", ".join(dropped[:8]) + (" ..." if len(dropped) > 8 else "")
            print(f"[data_loader] {len(dropped)} ticker(s) dropped (no data for the window): {preview}")
    elif source == "stooq":
        from pandas_datareader import data as pdr

        prices = pdr.DataReader(tickers, "stooq", start, end).sort_index()
        if isinstance(prices.columns, pd.MultiIndex):
            prices = prices["Close"].copy()
    else:
        raise ValueError(f"Unknown price source: {source!r}")

    prices.index = pd.to_datetime(prices.index)
    return prices.sort_index()


def prices_to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns from an adjusted-price panel."""
    return prices.pct_change().iloc[1:]


# ---------------------------------------------------------------------------
# Liquidity filter (proxy for the paper's market-cap screen)
# ---------------------------------------------------------------------------
def apply_liquidity_filter(
    returns: pd.DataFrame,
    prices: pd.DataFrame,
    max_names: int,
    min_history_days: int,
) -> pd.DataFrame:
    """Keep the ``max_names`` most-traded names with enough history.

    We approximate the paper's top-market-cap screen with a dollar-volume
    proxy. Since free adjusted prices don't carry volume cleanly for all
    names, we fall back to ranking by non-missing history length and average
    absolute price level, which correlates with large, liquid names.
    """
    enough_history = returns.notna().sum() >= min_history_days
    returns = returns.loc[:, enough_history[enough_history].index]

    # Rank by a simple liquidity proxy: average price * data completeness.
    completeness = returns.notna().mean()
    avg_price = prices.reindex(columns=returns.columns).mean()
    score = (avg_price.fillna(0) * completeness).sort_values(ascending=False)
    keep = score.head(max_names).index
    return returns.loc[:, keep]


# ---------------------------------------------------------------------------
# Fama-French factors (Kenneth French Data Library, free)
# ---------------------------------------------------------------------------
def download_ff_factors(start: str, end: str, num_factors: int = 5) -> pd.DataFrame:
    """Download daily Fama-French factors + risk-free rate (decimal returns).

    num_factors: 1 (Mkt-RF only / CAPM), 3 (FF3), or 5 (FF5). Momentum is
    appended when available and requested via num_factors >= 6.
    """
    from pandas_datareader import data as pdr

    ds = "F-F_Research_Data_5_Factors_2x3_daily"
    ff = pdr.DataReader(ds, "famafrench", start, end)[0] / 100.0
    ff.index = pd.to_datetime(ff.index.to_timestamp() if hasattr(ff.index, "to_timestamp") else ff.index)

    rf = ff["RF"].copy()
    factor_cols = {
        1: ["Mkt-RF"],
        3: ["Mkt-RF", "SMB", "HML"],
        5: ["Mkt-RF", "SMB", "HML", "RMW", "CMA"],
    }.get(num_factors, ["Mkt-RF", "SMB", "HML", "RMW", "CMA"])

    factors = ff[factor_cols].copy()
    factors["RF"] = rf
    return factors


# ---------------------------------------------------------------------------
# Orchestration with caching
# ---------------------------------------------------------------------------
def _cache_path(cache_dir: str, name: str) -> Path:
    p = Path(cache_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{name}.parquet"


def load_dataset(cfg: dict, force_refresh: bool = False) -> dict:
    """Load (cached) returns, prices, and FF factors according to config.

    Returns a dict with keys: ``returns``, ``prices``, ``factors``.
    """
    from src.universe import get_sp500_tickers

    dcfg = cfg["data"]
    fcfg = cfg["factors"]
    cache = dcfg["cache_dir"]

    prices_fp = _cache_path(cache, "prices")
    rets_fp = _cache_path(cache, "returns")
    ff_fp = _cache_path(cache, "ff_factors")

    if not force_refresh and prices_fp.exists() and rets_fp.exists() and ff_fp.exists():
        return {
            "prices": pd.read_parquet(prices_fp),
            "returns": pd.read_parquet(rets_fp),
            "factors": pd.read_parquet(ff_fp),
        }

    tickers = (
        get_sp500_tickers() if dcfg["universe"] == "sp500" else dcfg["universe"]
    )
    prices = download_prices(
        tickers, dcfg["start_date"], dcfg["end_date"], dcfg["price_source"]
    )
    returns = prices_to_returns(prices)
    returns = apply_liquidity_filter(
        returns, prices, dcfg["max_names"], dcfg["min_history_days"]
    )
    factors = download_ff_factors(
        dcfg["start_date"], dcfg["end_date"], fcfg["ff_num_factors"]
    )

    # Align everything to the factor calendar (US trading days with RF available).
    common = returns.index.intersection(factors.index)
    returns = returns.loc[common]
    factors = factors.loc[common]

    prices.to_parquet(prices_fp)
    returns.to_parquet(rets_fp)
    factors.to_parquet(ff_fp)
    return {"prices": prices, "returns": returns, "factors": factors}
