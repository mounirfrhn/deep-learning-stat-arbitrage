"""Quick manual check that the data layer downloads and caches correctly.

Usage:  python scripts/smoke_data.py
Downloads a tiny universe over a short window so it runs in seconds.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import data_loader as dl  # noqa: E402

cfg = {
    "data": {
        "universe": ["AAPL", "MSFT", "KO", "JPM", "XOM"],
        "start_date": "2020-01-01",
        "end_date": "2021-01-01",
        "price_source": "yfinance",
        "cache_dir": "data/_smoke",
        "max_names": 5,
        "min_history_days": 100,
    },
    "factors": {"ff_num_factors": 5},
}

ds = dl.load_dataset(cfg, force_refresh=True)
r, p, f = ds["returns"], ds["prices"], ds["factors"]
print("prices :", p.shape, "cols:", list(p.columns))
print("returns:", r.shape, "range:", r.index.min().date(), "->", r.index.max().date())
print("factors:", f.shape, "cols:", list(f.columns))
print("\nreturns head:\n", r.head(3))
print("\nfactors head:\n", f.head(3))
assert r.shape[0] > 100, "too few return rows"
assert f.shape[0] > 100, "too few factor rows"
assert r.index.equals(f.index), "returns/factors calendars not aligned"
print("\nOK: data layer works and calendars are aligned.")
