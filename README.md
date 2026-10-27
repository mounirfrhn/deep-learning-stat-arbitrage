# Deep Learning Statistical Arbitrage — Free-Data Reproduction

A from-scratch, free-data reproduction of the pipeline in
**Guijarro-Ordóñez, Pelger & Zanotti (2021), _"Deep Learning Statistical
Arbitrage"_** ([arXiv:2106.04028](https://arxiv.org/abs/2106.04028)).

The strategy has three stages:

1. **Arbitrage portfolios as factor residuals.** Systematic risk is removed from
   daily stock returns with a rolling factor model; the leftover **residuals**
   are (by construction) hedged and expected to mean-revert.
2. **Signal extraction with a convolutional transformer.** A CNN+Transformer
   reads the recent residual "price" path of each stock and outputs a trading
   signal — benchmarked against classical mean-reversion models.
3. **Optimal trading policy.** Signals are turned into portfolio weights by
   directly **maximizing the out-of-sample Sharpe ratio** under leverage and
   transaction-cost constraints.

## Why this is a reproduction, not a replica

The original paper uses **CRSP/Compustat** (proprietary) for ~550 of the largest
US stocks 1978–2016, plus 46 firm characteristics from Chen et al. (2022). This
project uses only **free data**, so some deliberate changes are made:

| Paper | This project |
|---|---|
| CRSP daily returns, monthly point-in-time top-500 by mkt cap | Current **S&P 500** via `yfinance` (⚠️ *survivorship bias*, see below) |
| Fama-French factors + RF (Kenneth French library) | **Same** — Kenneth French Data Library (free) |
| Factor models: FF, PCA, **IPCA** | FF + **PCA** only — IPCA dropped (needs the paid 46 characteristics) |
| CNN+Transformer signal + joint allocation | **Same architecture**, reimplemented in PyTorch |

### Known limitations (stated on purpose)

- **Survivorship bias.** Using *today's* S&P 500 excludes firms that were
  delisted or dropped, which inflates backtested returns. The paper avoids this
  with point-in-time CRSP membership. Results here are **not** directly
  comparable to the paper's headline Sharpe ratios and should be read as a
  methods demonstration.
- **IPCA omitted.** The conditional latent-factor model needs firm
  characteristics that aren't freely available; it's listed as future work.
- **Universe size & history** are smaller than the paper's.

## Project structure

```
src/
  universe.py      # S&P 500 constituents
  data_loader.py   # prices (yfinance/Stooq) + Fama-French factors, parquet cache
  factors.py       # rolling residuals: Fama-French (60d) and PCA (252d) — out-of-sample
  signals.py       # classical mean-reversion benchmarks            [in progress]
  models.py        # CNN+Transformer signal model (PyTorch)         [in progress]
  policy.py        # allocation head, Sharpe-maximizing loss        [in progress]
  backtest.py      # rolling out-of-sample backtest, transaction costs [in progress]
  metrics.py       # Sharpe, mean return, turnover, drawdown        [in progress]
notebooks/         # exploration + results/figures
tests/             # pytest: shapes, no look-ahead, metric sanity
papers/            # the original paper (PDF)
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# sanity-check the data layer (small universe, runs in seconds)
python scripts/smoke_data.py
```

Configuration (universe, dates, windows, model hyperparameters) lives in
[`config.yaml`](config.yaml).

## Methodology notes

- **No look-ahead bias.** Every factor model at day *t* is estimated using data
  only up to day *t−1*; residuals for day *t* are strictly out-of-sample.
- **Fama-French residuals** come from a rolling 60-day OLS of excess returns on
  the FF factors (Carhart 1997 style).
- **PCA residuals** follow Avellaneda & Lee (2010): eigen-portfolios from the
  trailing 252-day correlation matrix, loadings from a 60-day regression.

## References

- Guijarro-Ordóñez, Pelger, Zanotti (2021). *Deep Learning Statistical Arbitrage.* arXiv:2106.04028.
- Avellaneda & Lee (2010). *Statistical Arbitrage in the US Equities Market.*
- Kelly, Pruitt & Su (2019). *Characteristics Are Covariances (IPCA).*
- Chen, Pelger, Zhu (2022). *Deep Learning in Asset Pricing.*

---

*Reproduction project for educational / portfolio purposes. Not investment advice.*
