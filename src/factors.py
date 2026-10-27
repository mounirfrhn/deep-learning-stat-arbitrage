"""Residual (arbitrage-portfolio) construction via rolling factor models.

Given a panel of daily stock returns, we remove systematic risk using a factor
model estimated on a rolling window, and keep the residuals. These residuals are
the "arbitrage portfolios" the paper trades — they are (by construction) hedged
against the chosen factors and are expected to mean-revert.

Two factor families are implemented (IPCA is intentionally omitted — it needs
paid firm characteristics):

  * Fama-French:  excess returns regressed on observable FF factors
                  (rolling OLS, `ff_loadings_window` days).
  * PCA:          latent factors from the rolling return correlation matrix
                  (`pca_corr_window` days), loadings from a short regression
                  (`pca_loadings_window` days).

All residuals are strictly out-of-sample: the factor model at day *t* uses only
information up to day *t-1*, so there is no look-ahead bias.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _excess_returns(returns: pd.DataFrame, rf: pd.Series) -> pd.DataFrame:
    return returns.sub(rf.reindex(returns.index), axis=0)


def fama_french_residuals(
    returns: pd.DataFrame,
    factors: pd.DataFrame,
    loadings_window: int = 60,
) -> pd.DataFrame:
    """Daily out-of-sample residuals from a rolling Fama-French regression.

    For each day t we regress each stock's excess returns on the FF factors over
    the preceding ``loadings_window`` days (ending at t-1), then compute the
    residual of day t as r_t - alpha - beta' f_t. Carhart (1997) style.
    """
    rf = factors["RF"]
    factor_cols = [c for c in factors.columns if c != "RF"]
    f = factors[factor_cols]
    excess = _excess_returns(returns, rf)

    dates = returns.index
    resid = pd.DataFrame(index=dates, columns=returns.columns, dtype=float)

    f_vals = f.values
    ex_vals = excess.values
    n_days = len(dates)

    for t in range(loadings_window, n_days):
        lo, hi = t - loadings_window, t  # window [t-window, t-1]
        X = f_vals[lo:hi]
        X_design = np.column_stack([np.ones(len(X)), X])  # intercept + factors
        Y = ex_vals[lo:hi]  # (window, n_stocks)

        valid = ~np.isnan(Y).any(axis=0)  # stocks with full window history
        if not valid.any():
            continue

        # Solve OLS for all valid stocks at once.
        beta, *_ = np.linalg.lstsq(X_design, Y[:, valid], rcond=None)
        x_t = np.concatenate([[1.0], f_vals[t]])
        pred_t = x_t @ beta
        resid_t = ex_vals[t, valid] - pred_t
        resid.iloc[t, np.where(valid)[0]] = resid_t

    return resid


def pca_residuals(
    returns: pd.DataFrame,
    factors: pd.DataFrame,
    num_factors: int = 5,
    corr_window: int = 252,
    loadings_window: int = 60,
) -> pd.DataFrame:
    """Daily out-of-sample residuals from a rolling PCA factor model.

    Following Avellaneda & Lee (2010): estimate the eigen-portfolios from the
    trailing ``corr_window`` correlation matrix, regress each stock on those
    factor returns over ``loadings_window`` days, and residualize day t.
    """
    rf = factors["RF"]
    excess = _excess_returns(returns, rf)
    dates = returns.index
    resid = pd.DataFrame(index=dates, columns=returns.columns, dtype=float)

    ex = excess.values
    n_days = len(dates)
    start = max(corr_window, loadings_window)

    for t in range(start, n_days):
        win = ex[t - corr_window : t]  # (corr_window, n_stocks)
        valid = ~np.isnan(win).any(axis=0)
        if valid.sum() <= num_factors:
            continue
        W = win[:, valid]

        # Standardize, build correlation, take top eigenvectors as factor weights.
        mu = W.mean(axis=0)
        sd = W.std(axis=0)
        sd[sd == 0] = 1.0
        Z = (W - mu) / sd
        corr = np.corrcoef(Z, rowvar=False)
        corr = np.nan_to_num(corr)
        eigvals, eigvecs = np.linalg.eigh(corr)
        top = eigvecs[:, -num_factors:]            # (n_valid, k)
        weights = top / sd[:, None]                # eigen-portfolio weights

        # Factor returns over the loadings window and at day t.
        reg_win = ex[t - loadings_window : t][:, valid]  # (loadings_window, n_valid)
        f_hist = reg_win @ weights                        # (loadings_window, k)
        f_t = ex[t, valid] @ weights                      # (k,)

        X_design = np.column_stack([np.ones(len(f_hist)), f_hist])
        beta, *_ = np.linalg.lstsq(X_design, reg_win, rcond=None)  # (k+1, n_valid)
        pred_t = np.concatenate([[1.0], f_t]) @ beta
        resid_t = ex[t, valid] - pred_t
        resid.iloc[t, np.where(valid)[0]] = resid_t

    return resid


def build_residuals(returns: pd.DataFrame, factors: pd.DataFrame, cfg: dict) -> dict:
    """Build residual panels for every factor model listed in config."""
    fcfg = cfg["factors"]
    out: dict[str, pd.DataFrame] = {}
    if "fama_french" in fcfg["models"]:
        out["fama_french"] = fama_french_residuals(
            returns, factors, fcfg["ff_loadings_window"]
        )
    if "pca" in fcfg["models"]:
        out["pca"] = pca_residuals(
            returns,
            factors,
            fcfg["pca_num_factors"],
            fcfg["pca_corr_window"],
            fcfg["pca_loadings_window"],
        )
    return out
