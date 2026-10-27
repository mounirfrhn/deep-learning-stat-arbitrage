"""Training of the CNN+Transformer arbitrage model with a Sharpe objective.

Pipeline here:
  1. build_dataset  : residual panel -> per-day, per-stock input windows + targets
  2. portfolio_returns : differentiable map from raw per-stock weights to the
                         daily portfolio return (dollar-neutral, L1-normalized)
  3. sharpe_loss    : the training objective, -Sharpe over a set of days
  4. train_model    : chronological train/val/test split, mini-batches of DAYS,
                      Adam, early stopping on validation Sharpe

Why batches of *days* (not stock-samples): the objective is the Sharpe ratio of
the daily *portfolio* return, which couples all stocks on a given day. So one
training example is a whole day's cross-section of residual windows.

No look-ahead: the window for decision day d is residuals up to d; the target is
the residual of d+1. The train/val/test split is strictly chronological.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from src import metrics
from src.models import ConvTransformerSignal

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# 1. Dataset construction
# ---------------------------------------------------------------------------
def build_dataset(residuals: pd.DataFrame, lookback: int):
    """Turn a (days x stocks) residual panel into training tensors.

    For each decision day d (with a full lookback window and a next day):
      X[d]    = cumulative residuals over the window [d-L+1, d]  (shape N x L)
      Y[d]    = residual of day d+1                              (shape N)
      M[d]    = mask of stocks valid on that day (full window + target present)

    Returns (X, Y, M, dates) with
      X: (n_days, N, L) float32, Y: (n_days, N) float32, M: (n_days, N) bool.
    """
    r = residuals.values.astype(np.float32)        # (T, N)
    T, N = r.shape
    L = lookback

    X_list, Y_list, M_list, date_idx = [], [], [], []
    for d in range(L - 1, T - 1):
        window = r[d - L + 1 : d + 1, :]            # (L, N)
        target = r[d + 1, :]                        # (N,)
        valid = ~np.isnan(window).any(axis=0) & ~np.isnan(target)  # (N,)

        win = np.nan_to_num(window, nan=0.0)
        cum = np.cumsum(win, axis=0)                # integrate residuals -> "price"
        X_list.append(cum.T)                        # (N, L)
        Y_list.append(np.nan_to_num(target, nan=0.0))
        M_list.append(valid)
        date_idx.append(residuals.index[d + 1])     # the day the P&L is realized

    X = torch.tensor(np.stack(X_list))              # (n_days, N, L)
    Y = torch.tensor(np.stack(Y_list))              # (n_days, N)
    M = torch.tensor(np.stack(M_list))              # (n_days, N) bool
    return X, Y, M, pd.DatetimeIndex(date_idx)


# ---------------------------------------------------------------------------
# 2 & 3. Portfolio returns + Sharpe loss (differentiable)
# ---------------------------------------------------------------------------
def portfolio_returns(
    model: torch.nn.Module, X: torch.Tensor, Y: torch.Tensor, M: torch.Tensor
) -> torch.Tensor:
    """Daily portfolio returns for a block of days. Shape (n_days,), differentiable.

    For each day: run the model per stock, zero out invalid stocks, make the book
    dollar-neutral, L1-normalize to unit gross leverage, and dot with next-day
    residuals.
    """
    n_days, N, L = X.shape
    raw = model(X.reshape(n_days * N, L)).reshape(n_days, N)  # (n_days, N)

    mask = M.float()
    raw = raw * mask                                          # drop invalid stocks

    # Dollar-neutral: subtract the per-day mean over valid stocks only.
    n_valid = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
    mean = (raw.sum(dim=1, keepdim=True)) / n_valid
    centered = (raw - mean) * mask

    # L1-normalize to unit gross leverage.
    gross = centered.abs().sum(dim=1, keepdim=True).clamp(min=1e-8)
    weights = centered / gross                                # sum|w| = 1, sum w = 0

    daily = (weights * Y).sum(dim=1)                          # (n_days,)
    return daily


def sharpe_loss(returns: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Negative annualized Sharpe ratio (to minimize)."""
    mean = returns.mean()
    std = returns.std(unbiased=False).clamp(min=eps)
    return -(mean / std) * (TRADING_DAYS ** 0.5)


# ---------------------------------------------------------------------------
# 4. Training with a chronological split
# ---------------------------------------------------------------------------
def chronological_split(n_days: int, val_frac: float, test_frac: float):
    n_test = int(n_days * test_frac)
    n_val = int(n_days * val_frac)
    n_train = n_days - n_val - n_test
    idx = np.arange(n_days)
    return idx[:n_train], idx[n_train : n_train + n_val], idx[n_train + n_val :]


def train_model(
    residuals: pd.DataFrame,
    cfg: dict,
    val_frac: float = 0.15,
    test_frac: float = 0.2,
    verbose: bool = True,
):
    """Train the CNN+Transformer on a residual panel. Returns (model, result)."""
    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)  # the mini-batch shuffling below uses NumPy
    lookback = cfg["signal"]["lookback"]
    tcfg = cfg["train"]
    device = torch.device(tcfg.get("device", "cpu"))

    X, Y, M, dates = build_dataset(residuals, lookback)
    X, Y, M = X.to(device), Y.to(device), M.to(device)
    tr, va, te = chronological_split(len(X), val_frac, test_frac)

    model = ConvTransformerSignal(lookback=lookback).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=tcfg["lr"])

    batch_days = min(tcfg["batch_size"], len(tr))
    best_val, best_state, patience, bad = -1e9, None, 10, 0

    for epoch in range(tcfg["epochs"]):
        model.train()
        perm = np.random.permutation(tr)
        for i in range(0, len(perm), batch_days):
            b = perm[i : i + batch_days]
            opt.zero_grad()
            ret = portfolio_returns(model, X[b], Y[b], M[b])
            loss = sharpe_loss(ret)
            loss.backward()
            opt.step()

        # Validation Sharpe (model selection / early stopping).
        model.eval()
        with torch.no_grad():
            val_sharpe = -sharpe_loss(portfolio_returns(model, X[va], Y[va], M[va])).item()
        if val_sharpe > best_val:
            best_val, best_state, bad = val_sharpe, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
        if verbose and (epoch % 5 == 0 or epoch == tcfg["epochs"] - 1):
            print(f"  epoch {epoch:3d} | val Sharpe {val_sharpe:+.3f} | best {best_val:+.3f}")
        if bad >= patience:
            if verbose:
                print(f"  early stop at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Out-of-sample evaluation on the untouched test block.
    model.eval()
    with torch.no_grad():
        test_ret = portfolio_returns(model, X[te], Y[te], M[te]).cpu().numpy()
    test_series = pd.Series(test_ret, index=dates[te], name="strategy_return")

    result = {
        "val_sharpe": best_val,
        "test_summary": metrics.summary(test_series),
        "test_returns": test_series,
        "split_dates": {
            "train": (dates[tr][0], dates[tr][-1]),
            "val": (dates[va][0], dates[va][-1]),
            "test": (dates[te][0], dates[te][-1]),
        },
    }
    return model, result
