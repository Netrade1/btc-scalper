import numpy as np
import pandas as pd

from btc_scalper.trend import block_bootstrap_sharpe_diff, trend_returns


def daily(n=200, seed=0):
    rng = np.random.default_rng(seed)
    close = 50_000 * np.exp(np.cumsum(rng.normal(0, 0.03, n)))
    open_ = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.002, n))
    idx = pd.date_range("2025-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame({"open": open_, "high": np.maximum(open_, close) * 1.01,
                         "low": np.minimum(open_, close) * 0.99, "close": close}, index=idx)


def test_positions_use_only_past_prices():
    d = daily()
    base = trend_returns(d, lookback=10).position
    cut = d.index[120]
    d2 = d.copy()
    d2.loc[d2.index > cut, ["open", "close"]] *= 3.0   # rewrite the future
    pos2 = trend_returns(d2, lookback=10).position
    common = base.index[base.index <= cut]
    pd.testing.assert_series_equal(base[common], pos2[common])


def test_costs_charged_on_every_switch():
    d = daily()
    free = trend_returns(d, lookback=10, fee=0.0, slippage=0.0)
    paid = trend_returns(d, lookback=10, fee=0.004, slippage=0.0001)
    drag = (free.strategy - paid.strategy).sum()
    assert np.isclose(drag, paid.switches * 0.0041)
    assert paid.switches > 0
    assert np.isclose(free.benchmark.iloc[0] - paid.benchmark.iloc[0], 0.0041)


def test_bootstrap_detects_clear_difference_and_not_noise():
    rng = np.random.default_rng(1)
    b = pd.Series(rng.normal(0, 0.02, 700))
    p_good, _ = block_bootstrap_sharpe_diff(b + 0.004, b + rng.normal(0, 0.005, 700), n=2000)
    p_same, _ = block_bootstrap_sharpe_diff(b, b + rng.normal(0, 0.001, 700), n=2000)
    assert p_good < 0.01
    assert 0.05 < p_same < 0.95
