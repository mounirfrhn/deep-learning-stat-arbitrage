"""End-to-end benchmark run: data -> residuals -> mean-reversion -> backtest.

Validates the full non-DL pipeline. Usage:  python scripts/run_benchmark.py
"""

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import backtest, data_loader as dl, factors as fac, policy, signals  # noqa: E402


def main(config_path: str = "config.yaml") -> None:
    cfg = yaml.safe_load(open(config_path))

    ds = dl.load_dataset(cfg)
    residuals = fac.build_residuals(ds["returns"], ds["factors"], cfg)

    lookback = cfg["signal"]["lookback"]
    lev = cfg["policy"]["max_gross_leverage"]
    tc = cfg["policy"]["transaction_cost_bps"]

    for model_name, resid in residuals.items():
        sig = signals.mean_reversion_signal(resid, lookback)
        w = policy.signals_to_weights(sig, lev)
        bt = backtest.run_backtest(w, resid, tc)
        s = bt["summary"]
        print(f"\n=== {model_name} residuals + mean-reversion ===")
        print(f"  Sharpe (ann):      {s['sharpe']:.2f}")
        print(f"  Return (ann):      {s['ann_return']:.2%}")
        print(f"  Vol (ann):         {s['ann_vol']:.2%}")
        print(f"  Max drawdown:      {s['max_drawdown']:.2%}")
        print(f"  Avg daily turnover:{s['avg_daily_turnover']:.2f}")
        print(f"  Trading days:      {s['n_days']}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "config.yaml")
