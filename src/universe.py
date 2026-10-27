"""Investable universe: current S&P 500 constituents.

The original paper uses the ~550 largest US stocks by market cap each month,
reconstituted monthly from CRSP. We don't have point-in-time constituents for
free, so we use the *current* S&P 500 membership. This introduces survivorship
bias (today's index excludes firms that were delisted/dropped), which is
documented as a known limitation in the README. For a methods-reproduction
project this is an acceptable, clearly-stated trade-off.
"""

from __future__ import annotations

import io

import pandas as pd

WIKI_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

# Small fallback used only if the Wikipedia fetch fails (e.g. offline tests).
_FALLBACK_TICKERS = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "BRK-B", "JPM", "JNJ",
    "V", "PG", "XOM", "HD", "MA", "KO", "PEP", "BAC", "WMT", "DIS", "CSCO",
]


def get_sp500_tickers(use_fallback_on_error: bool = True) -> list[str]:
    """Return current S&P 500 tickers in a yfinance-compatible format.

    Tickers with a dot (e.g. ``BRK.B``) are converted to the dash form
    (``BRK-B``) that Yahoo Finance expects.
    """
    try:
        # Fetch via requests (bundles certifi CA roots) rather than letting
        # pandas.read_html use urllib, which fails on macOS framework Python
        # with SSL CERTIFICATE_VERIFY_FAILED.
        import requests

        headers = {"User-Agent": "Mozilla/5.0 (research script)"}
        html = requests.get(WIKI_SP500_URL, headers=headers, timeout=30).text
        tables = pd.read_html(io.StringIO(html))
        symbols = tables[0]["Symbol"].astype(str).str.strip().tolist()
    except Exception:
        if not use_fallback_on_error:
            raise
        return list(_FALLBACK_TICKERS)
    return [s.replace(".", "-") for s in symbols]
