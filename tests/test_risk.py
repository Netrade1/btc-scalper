import pandas as pd
import pytest

from btc_scalper.config import ExecutionConfig, RiskConfig, StrategyConfig, SystemConfig
from btc_scalper.models import Signal, Trade
from btc_scalper.risk import RiskGovernor
from btc_scalper.strategy import make_signal, round_trip_cost_r

T0 = pd.Timestamp("2024-01-01", tz="UTC")


def sig(p=0.5, stop=100.0, direction=1):
    return Signal(T0, direction, p, 1 - p, p, 0.3, stop, "edge")


def test_sizing_risk_kelly_and_leverage_caps():
    rg = RiskGovernor(RiskConfig(risk_per_trade=0.01, kelly_fraction=0.25, max_leverage=3.0), 300)
    intent, _ = rg.size_order(sig(p=0.5), 10_000, 100, 0.002, 24, 2.0)
    # Kelly f* = (0.5*2-0.5)/2 = 0.25 -> 0.25*0.25 = 6.25% > 1% cap, so risk = 1% = 1000
    assert intent.quantity * 100 == pytest.approx(1000)
    intent, _ = rg.size_order(sig(p=0.5, stop=1.0), 10_000, 100, 0.002, 24, 2.0)
    assert intent.quantity * 10_000 == pytest.approx(3 * 100_000)   # leverage-capped
    assert rg.size_order(sig(p=0.30), 10_000, 100, 0.002, 24, 2.0)[0] is None  # negative Kelly


def test_kill_switch_latches_and_blocks():
    rg = RiskGovernor(RiskConfig(max_drawdown=0.10, daily_loss_limit=0.5), 300)
    rg.on_bar(T0, 100_000)
    assert "kill_switch" in rg.on_bar(T0 + pd.Timedelta(minutes=5), 89_000)
    rg.on_bar(T0 + pd.Timedelta(minutes=10), 120_000)
    assert rg.pre_trade_check(T0 + pd.Timedelta(minutes=10), 0.002) == (False, "kill_switch")
    rg.reset_kill_switch()
    assert rg.pre_trade_check(T0 + pd.Timedelta(minutes=10), 0.002)[0]


def test_daily_loss_limit_resets_next_day():
    rg = RiskGovernor(RiskConfig(daily_loss_limit=0.02), 300)
    rg.on_bar(T0, 100_000)
    rg.on_bar(T0 + pd.Timedelta(hours=1), 97_900)
    assert rg.pre_trade_check(T0 + pd.Timedelta(hours=1), 0.002)[1] == "daily_loss_limit"
    rg.on_bar(T0 + pd.Timedelta(days=1), 97_900)
    rg.gap_detected = False
    assert rg.pre_trade_check(T0 + pd.Timedelta(days=1), 0.002)[0]


def test_consecutive_losses_cooldown_and_vol_band():
    rg = RiskGovernor(RiskConfig(max_consecutive_losses=2, cooldown_bars=10), 300)
    loss = Trade("x", 1, 1, T0, T0, 1, 1, -1, 0, 0, -1, -1, 0, 1, "STOP")
    rg.on_trade_closed(loss, T0)
    rg.on_trade_closed(loss, T0)
    assert rg.pre_trade_check(T0 + pd.Timedelta(minutes=5), 0.002)[1] == "loss_cooldown"
    assert rg.pre_trade_check(T0 + pd.Timedelta(hours=1), 0.002)[0]
    assert rg.pre_trade_check(T0 + pd.Timedelta(hours=1), 0.0001)[1] == "volatility_too_low"


def test_data_gap_blocks_entries():
    rg = RiskGovernor(RiskConfig(), 300)
    rg.on_bar(T0, 100_000)
    rg.on_bar(T0 + pd.Timedelta(hours=2), 100_000)
    assert rg.pre_trade_check(T0 + pd.Timedelta(hours=2), 0.002)[1] == "data_gap"


def test_signal_requires_edge_after_costs():
    st, ex = StrategyConfig(allow_short=True), ExecutionConfig(taker_fee=0.0005)
    cost = round_trip_cost_r(10_000, 15.0, ex)
    assert cost == pytest.approx(2 * 0.0006 * 10_000 / 15)
    s = make_signal(T0, 0.45, 0.2, 10.0, 10_000, st, ex)   # E = 0.35 - 0.8 < 0
    assert s.direction == 0 and s.reason == "edge_below_threshold"
    s = make_signal(T0, 0.2, 0.62, 100.0, 10_000, st, ex)
    assert s.direction == -1
    st.allow_short = False
    assert make_signal(T0, 0.2, 0.62, 100.0, 10_000, st, ex).direction == 0


def test_config_validation():
    with pytest.raises(ValueError):
        SystemConfig.from_dict({"risk": {"risk_per_tarde": 0.01}})
    with pytest.raises(ValueError):
        SystemConfig.from_dict({"strategy": {"allow_short": True}})   # spot default cannot short
    cfg = SystemConfig.load(None)
    assert cfg.data.market_type == "spot" and not cfg.strategy.allow_short
    assert cfg.execution.taker_fee == 0.0010


def test_default_spot_costs_are_about_one_r_on_5m():
    """Shipped spot costs: 2 x (0.10% + 1bp) on a 1.5 x ATR stop with ATR ~0.15% of price."""
    cost = round_trip_cost_r(60_000, 1.5 * 0.0015 * 60_000, ExecutionConfig())
    assert 0.9 < cost < 1.1
