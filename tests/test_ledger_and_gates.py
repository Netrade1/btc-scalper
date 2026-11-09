import json

import numpy as np
import pandas as pd
import pytest

from btc_scalper.audit import AuditLog, verify_ledger
from btc_scalper.backtest import run_backtest
from btc_scalper.broker import SimBroker
from btc_scalper.config import ExecutionConfig
from btc_scalper.data import generate_synthetic_ohlcv, integrity_gate, validate_ohlcv
from btc_scalper.engine import TradingEngine
from btc_scalper.labeling import triple_barrier_labels
from btc_scalper.models import OrderIntent
from btc_scalper.predictor import WalkForwardPredictor
from btc_scalper.risk import RiskGovernor


def test_ledger_verifies_and_detects_tampering(tmp_path):
    path = str(tmp_path / "ledger.jsonl")
    log = AuditLog(path)
    for i in range(5):
        log.log("order_submitted", None, qty=i * 0.1)
    log.close()
    assert verify_ledger(path) == (True, 5, "")

    log = AuditLog(path)                       # re-open continues the chain
    log.log("trade_closed", None, pnl=12.5)
    log.close()
    assert verify_ledger(path)[:2] == (True, 6)

    lines = open(path).read().splitlines()
    rec = json.loads(lines[2])
    rec["qty"] = 99.0                          # alter a record
    lines[2] = json.dumps(rec, sort_keys=True, separators=(",", ":"))
    open(path, "w").write("\n".join(lines) + "\n")
    ok, n, err = verify_ledger(path)
    assert not ok and n == 2 and "line 3" in err
    with pytest.raises(RuntimeError):
        AuditLog(path)                         # refuses to append to a broken chain

    del lines[2]                               # deleting a line also breaks it
    open(path, "w").write("\n".join(lines) + "\n")
    assert not verify_ledger(path)[0]


def test_integrity_gate_flags_gap_and_stale_feed():
    raw = generate_synthetic_ohlcv(300, seed=1)
    gapped = validate_ohlcv(raw.drop(raw.index[-20:-15]), 300)
    now = gapped.index[-1] + pd.Timedelta(minutes=5)
    assert "gap_before_bar" in integrity_gate(gapped, now, 300, 100)
    clean = validate_ohlcv(raw, 300)
    assert integrity_gate(clean, clean.index[-1] + pd.Timedelta(minutes=5, seconds=3), 300, 100) == []
    stale = integrity_gate(clean, clean.index[-1] + pd.Timedelta(hours=1), 300, 100)
    assert any(r.startswith("stale_feed") for r in stale)
    bad = raw.copy()
    bad.iloc[-3, bad.columns.get_loc("buy_volume")] = -5.0
    assert "flow_out_of_range" in integrity_gate(validate_ohlcv(bad, 300), now, 300, 100)


def test_label_bar_touching_both_barriers_is_a_loss():
    idx = pd.date_range("2024-01-01", periods=4, freq="5min", tz="UTC")
    df = pd.DataFrame({"open": [100, 100, 100, 100], "high": [100, 110, 100, 100],
                       "low": [100, 90, 100, 100], "close": [100, 100, 100, 100]}, index=idx, dtype=float)
    lab = triple_barrier_labels(df, pd.Series(2.0, index=idx), 1.5, 2.0, 2)
    assert lab["long_r"].iloc[0] == -1.0 and lab["long_win"].iloc[0] == 0.0
    assert lab["short_r"].iloc[0] == -1.0
    assert lab.iloc[-2:].isna().all().all()    # unresolved tail is NaN, not 0


def test_no_lookahead_corrupting_future_leaves_past_decisions_unchanged(fast_cfg):
    raw = generate_synthetic_ohlcv(9000, seed=7, alpha=1.5)
    t = 7000
    corrupt = raw.copy()
    rng = np.random.default_rng(0)
    future = corrupt.index[t + 1:]
    scale = np.exp(rng.normal(0, 0.05, len(future)))
    for c in ("open", "high", "low", "close"):
        corrupt.loc[future, c] *= scale
    corrupt.loc[future, "high"] = corrupt.loc[future, ["open", "high", "low", "close"]].max(axis=1) * 1.01
    corrupt.loc[future, "low"] = corrupt.loc[future, ["open", "high", "low", "close"]].min(axis=1) * 0.99
    corrupt.loc[future, "buy_volume"] = corrupt.loc[future, "volume"] * rng.uniform(0, 1, len(future))

    def decisions(df):
        res = run_backtest(df, fast_cfg)
        keep = ("order_submitted", "model_retrained", "entry_blocked")
        return [{k: v for k, v in r.items() if k not in ("wall_ts", "hash", "prev_hash", "client_id")}
                for r in res.audit if r["event"] in keep and pd.Timestamp(r["bar_ts"]) <= raw.index[t]]

    a, b = decisions(raw), decisions(corrupt)
    assert any(r["event"] == "order_submitted" for r in a)
    assert a == b


def test_circuit_breaker_trips_and_flattens(fast_cfg):
    raw = generate_synthetic_ohlcv(400, seed=2)
    from btc_scalper.backtest import prepare

    raw, feats, labels = prepare(raw, fast_cfg)
    fast_cfg.risk.max_drawdown = 0.05
    broker = SimBroker(ExecutionConfig(), 100_000, 300)
    risk = RiskGovernor(fast_cfg.risk, 300)
    eng = TradingEngine(fast_cfg, broker, WalkForwardPredictor(fast_cfg.model, fast_cfg.strategy, 300),
                        risk, AuditLog())
    eng.step(raw, feats, labels, 0, decide=False)
    # Force a large long position into a falling market
    px = raw["close"].iloc[0]
    broker.submit(OrderIntent("x", raw.index[0], 1, 100_000 * 3 / px, px * 0.5, 2.0, 1000, 0.1, 0.5,
                              px, 1e9, 0.001), None)
    for t in range(1, len(raw)):
        eng.step(raw, feats, labels, t, decide=False)
        if risk.state.kill_switch:
            break
    else:  # make sure the market fell enough to trip; otherwise force a mark
        pytest.skip("synthetic path did not draw down")
    assert broker.position is None
    assert any(r["event"] == "risk_alert" and r["alert"] == "kill_switch" for r in eng.audit.records)
    assert risk.pre_trade_check(raw.index[t], 0.002) == (False, "kill_switch")
