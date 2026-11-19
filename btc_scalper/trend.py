"""Pre-registered trend test (docs/hypotheses/H2_trend.md): a daily time-series momentum rule.

Signal at the close of day t; the position is held from the open of day t+1 to the open of t+2.
Research only: no orders. A pass leads to a paper run under the Risk Governor.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DAYS_PER_YEAR = 365


@dataclass
class TrendResult:
    strategy: pd.Series   # net daily returns
    benchmark: pd.Series  # buy-and-hold net daily returns
    position: pd.Series   # 0/1 held over each day's open-to-open return
    switches: int


def trend_returns(daily: pd.DataFrame, lookback: int = 28, fee: float = 0.004,
                  slippage: float = 0.0001) -> TrendResult:
    if len(daily) < lookback + 3:
        raise ValueError("Not enough daily bars for the lookback")
    cost = fee + slippage
    open_ret = daily["open"].shift(-1) / daily["open"] - 1          # open[d] -> open[d+1]
    signal = (daily["close"] / daily["close"].shift(lookback) - 1 > 0).astype(float)
    signal[daily["close"].shift(lookback).isna()] = np.nan
    position = signal.shift(1)                                       # decided at close d-1
    keep = position.notna() & open_ret.notna()
    position, open_ret = position[keep], open_ret[keep]
    prev = position.shift(1).fillna(0.0)                             # start flat: entry is charged
    turnover = (position - prev).abs()
    strat = position * open_ret - turnover * cost
    bench = open_ret.copy()
    bench.iloc[0] -= cost
    return TrendResult(strat, bench, position, int(turnover.sum()))


def sharpe(r: pd.Series) -> float:
    sd = r.std()
    return float(r.mean() / sd * np.sqrt(DAYS_PER_YEAR)) if sd > 0 else float("nan")


def max_dd(r: pd.Series) -> float:
    eq = (1 + r).cumprod()
    return float((eq / eq.cummax() - 1).min())


def block_bootstrap_sharpe_diff(a: pd.Series, b: pd.Series, block: int = 20, n: int = 10_000,
                                seed: int = 0) -> tuple[float, np.ndarray]:
    """Paired circular block bootstrap of Sharpe(a) - Sharpe(b). Returns (P(diff <= 0), diffs)."""
    x, y = a.to_numpy(), b.to_numpy()
    m = len(x)
    rng = np.random.default_rng(seed)
    k = int(np.ceil(m / block))
    starts = rng.integers(0, m, size=(n, k))
    idx = ((starts[:, :, None] + np.arange(block)) % m).reshape(n, -1)[:, :m]
    xs, ys = x[idx], y[idx]

    def sr(z):
        sd = z.std(axis=1, ddof=1)
        return np.where(sd > 0, z.mean(axis=1) / np.where(sd > 0, sd, 1) * np.sqrt(DAYS_PER_YEAR), np.nan)

    diffs = sr(xs) - sr(ys)
    return float(np.mean(diffs <= 0)), diffs
