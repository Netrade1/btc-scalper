"""End-to-end: the system must not trade noise, and must find a real edge."""
import json

import pandas as pd

from btc_scalper.backtest import run_backtest
from btc_scalper.data import generate_synthetic_ohlcv
from btc_scalper.exchange import CCXTClient
from btc_scalper.live import LiveRunner
from btc_scalper.metrics import max_drawdown, performance_summary

from .fakes import FakeExchange


def test_pure_noise_produces_no_meaningful_trading(fast_cfg):
    res = run_backtest(generate_synthetic_ohlcv(12_000, seed=3, alpha=0.0), fast_cfg)
    assert res.summary["trades"] <= 10
    assert res.summary["max_drawdown"] < 0.05


def test_noise_stays_flat_at_spot_costs_too(fast_spot_cfg):
    res = run_backtest(generate_synthetic_ohlcv(12_000, seed=3, alpha=0.0), fast_spot_cfg)
    assert res.summary["trades"] <= 5


def test_cost_sensitivity_same_edge_two_cost_levels(fast_cfg, fast_spot_cfg):
    raw = generate_synthetic_ohlcv(12_000, seed=7, alpha=1.5)
    perp = run_backtest(raw, fast_cfg).summary
    spot = run_backtest(raw, fast_spot_cfg).summary
    assert perp["trades"] > spot["trades"]               # higher costs -> far fewer qualifying setups
    assert perp["total_return"] > spot["total_return"]


def test_planted_edge_is_found_and_monetised(fast_cfg, tmp_path):
    res = run_backtest(generate_synthetic_ohlcv(12_000, seed=7, alpha=1.5), fast_cfg, str(tmp_path))
    s = res.summary
    assert s["trades"] >= 30
    assert s["total_return"] > 0 and s["expectancy_r"] > 0
    assert s["max_drawdown"] < fast_cfg.risk.max_drawdown
    for f in ("equity.csv", "trades.csv", "summary.json", "audit.jsonl", "model_diagnostics.csv"):
        assert (tmp_path / f).exists()
    events = {json.loads(l)["event"] for l in open(tmp_path / "audit.jsonl")}
    assert {"model_retrained", "order_submitted", "trade_closed"} <= events
    from btc_scalper.audit import verify_ledger
    assert verify_ledger(str(tmp_path / "audit.jsonl"))[0]


def test_metrics_basics():
    eq = pd.Series([100, 110, 99, 120], index=pd.date_range("2024", periods=4, freq="5min"))
    assert abs(max_drawdown(eq)[0] - 0.1) < 1e-12
    s = performance_summary(eq, [], 300, 100)
    assert abs(s["total_return"] - 0.2) < 1e-12


def test_paper_runner_dedups_bars_and_restores_state(fast_cfg, tmp_path):
    fast_cfg.live.state_file = str(tmp_path / "state.json")
    fast_cfg.live.audit_file = str(tmp_path / "audit.jsonl")
    bars = generate_synthetic_ohlcv(2100, seed=5, alpha=1.0)
    tf_ms = 300_000
    now = int(bars.index[2050].timestamp() * 1000) + tf_ms + 1
    ex = FakeExchange(bars, now_ms=now)
    client = CCXTClient(fast_cfg.data, exchange=ex)
    clock = lambda: pd.Timestamp(ex.now_ms, unit="ms", tz="UTC")  # noqa: E731
    runner = LiveRunner(fast_cfg, client, clock=clock)
    assert runner.run_cycle() == 1
    assert runner.run_cycle() == 0              # same bar again: no double processing
    ex.now_ms += 5 * tf_ms                       # five new bars (e.g. after a hiccup)
    assert runner.run_cycle() == 5
    last = runner.last_ts

    runner2 = LiveRunner(fast_cfg, client, clock=clock)   # simulated restart
    assert runner2.last_ts == last
    assert runner2.risk.state.equity == runner.risk.state.equity
    assert runner2.run_cycle() == 0


def test_live_mode_is_refused(fast_cfg, tmp_path):
    import pytest

    from btc_scalper.cli import main

    fast_cfg.live.state_file = str(tmp_path / "s.json")
    with pytest.raises(PermissionError):
        LiveRunner(fast_cfg, client=None, live=True)
    with pytest.raises(PermissionError):
        main(["live", "--i-accept-the-risk"])


def test_paper_runner_blocks_decisions_on_stale_feed(fast_cfg, tmp_path):
    fast_cfg.live.state_file = str(tmp_path / "state.json")
    fast_cfg.live.audit_file = str(tmp_path / "audit.jsonl")
    bars = generate_synthetic_ohlcv(2100, seed=5)
    ex = FakeExchange(bars, now_ms=int(bars.index[-1].timestamp() * 1000) + 300_001)
    client = CCXTClient(fast_cfg.data, exchange=ex)
    late = pd.Timestamp(ex.now_ms, unit="ms", tz="UTC") + pd.Timedelta(minutes=30)
    runner = LiveRunner(fast_cfg, client, clock=lambda: late)
    runner.run_cycle()
    events = [json.loads(l) for l in open(tmp_path / "audit.jsonl")]
    blocked = [e for e in events if e["event"] == "integrity_gate_blocked"]
    assert blocked and any(r.startswith("stale_feed") for r in blocked[0]["reasons"])


def test_warm_start_builds_forward_track_record(fast_cfg, tmp_path):
    fast_cfg.live.state_file = str(tmp_path / "state.json")
    fast_cfg.live.audit_file = str(tmp_path / "audit.jsonl")
    fast_cfg.live.warmup_bars = 4000
    bars = generate_synthetic_ohlcv(4000, seed=5, alpha=1.0)
    ex = FakeExchange(bars, now_ms=int(bars.index[-1].timestamp() * 1000) + 300_001)
    runner = LiveRunner(fast_cfg, CCXTClient(fast_cfg.data, exchange=ex),
                        clock=lambda: pd.Timestamp(ex.now_ms, unit="ms", tz="UTC"))
    runner.warm_start()
    assert len(runner.predictor.oos["long"]) > 1000
    runner.run_cycle()
    assert runner.predictor.diagnostics[-1].extra["oos_n"] > 500
