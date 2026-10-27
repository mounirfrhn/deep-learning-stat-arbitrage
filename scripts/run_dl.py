"""Train the CNN+Transformer and compare it to the mean-reversion benchmark
on the SAME out-of-sample test window.

Usage:  python scripts/run_dl.py [config.yaml] [fama_french|pca]
"""

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import backtest, data_loader as dl, factors as fac, metrics, policy, signals, train  # noqa: E402


def main(config_path: str = "config.yaml", model_family: str = "fama_french") -> None:
    cfg = yaml.safe_load(open(config_path))

    print("Loading data + building residuals ...")
    ds = dl.load_dataset(cfg)
    residuals = fac.build_residuals(ds["returns"], ds["factors"], cfg)[model_family]

    print(f"\nTraining CNN+Transformer on '{model_family}' residuals ...")
    model, res = train.train_model(residuals, cfg)

    t0, t1 = res["split_dates"]["test"]
    print(f"\nTest window: {t0.date()} -> {t1.date()}")

    # Benchmark (mean-reversion) on the SAME test window, for a fair comparison.
    sig = signals.mean_reversion_signal(residuals, cfg["signal"]["lookback"])
    w = policy.signals_to_weights(sig, cfg["policy"]["max_gross_leverage"])
    bt = backtest.run_backtest(w, residuals, cfg["policy"]["transaction_cost_bps"])
    bench_test = bt["returns"].loc[t0:t1]
    bench_summary = metrics.summary(bench_test[bench_test != 0])

    dl_s = res["test_summary"]
    print("\n================= OUT-OF-SAMPLE (test) =================")
    print(f"{'metric':<16}{'CNN+Transformer':>18}{'mean-reversion':>18}")
    for key, label in [("sharpe", "Sharpe"), ("ann_return", "Return"), ("ann_vol", "Vol"), ("max_drawdown", "MaxDD")]:
        fmt = (lambda v: f"{v:.2f}") if key == "sharpe" else (lambda v: f"{v:.2%}")
        print(f"{label:<16}{fmt(dl_s[key]):>18}{fmt(bench_summary[key]):>18}")
    print("========================================================")
    print(f"(validation Sharpe used for model selection: {res['val_sharpe']:+.2f})")


if __name__ == "__main__":
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    family = sys.argv[2] if len(sys.argv) > 2 else "fama_french"
    main(cfg_path, family)
