import pandas as pd
import pytest

from btc_scalper.broker import SimBroker
from btc_scalper.config import ExecutionConfig
from btc_scalper.models import Bar, OrderIntent

T0 = pd.Timestamp("2024-01-01", tz="UTC")


def bar(i, o, h, l, c, v=100.0):
    return Bar(T0 + pd.Timedelta(minutes=5 * i), o, h, l, c, v, v / 2)


def intent(side=1, qty=1.0, stop=100.0, rr=2.0, hold=24):
    return OrderIntent("id", T0, side, qty, stop, rr, hold, 0.2, 0.5, 10_000.0, 1e9, 0.001)


def frictionless():
    return ExecutionConfig(taker_fee=0.0, slippage_bps=0.0, impact_coef=0.0, funding_rate_8h=0.0)


def test_fills_at_next_open_and_anchors_bracket_to_fill():
    b = SimBroker(frictionless(), 100_000, 300)
    b.submit(intent(), bar(0, 10_000, 10_010, 9_990, 10_000))
    b.process_bar(bar(1, 10_050, 10_060, 10_040, 10_050))
    p = b.position
    assert p.entry_price == 10_050 and p.stop_price == 9_950 and p.take_profit == 10_250


def test_ambiguous_bar_assumes_stop_first():
    b = SimBroker(frictionless(), 100_000, 300)
    b.submit(intent(), bar(0, 10_000, 10_000, 10_000, 10_000))
    trades = b.process_bar(bar(1, 10_000, 10_500, 9_800, 10_000))
    assert trades[0].exit_reason == "STOP" and trades[0].r_multiple == pytest.approx(-1.0)


def test_gap_through_stop_fills_at_open():
    b = SimBroker(frictionless(), 100_000, 300)
    b.submit(intent(), bar(0, 10_000, 10_000, 10_000, 10_000))
    b.process_bar(bar(1, 10_000, 10_010, 9_990, 10_000))
    trades = b.process_bar(bar(2, 9_700, 9_750, 9_650, 9_700))
    assert trades[0].exit_reason == "STOP_GAP" and trades[0].exit_price == 9_700
    assert trades[0].r_multiple == pytest.approx(-3.0)


def test_short_target_and_time_stop():
    b = SimBroker(frictionless(), 100_000, 300)
    b.submit(intent(side=-1), bar(0, 10_000, 10_000, 10_000, 10_000))
    trades = b.process_bar(bar(1, 10_000, 10_010, 9_790, 9_800))
    assert trades[0].exit_reason == "TARGET" and trades[0].r_multiple == pytest.approx(2.0)

    b.submit(intent(hold=2), bar(2, 10_000, 10_000, 10_000, 10_000))
    assert b.process_bar(bar(3, 10_000, 10_010, 9_990, 10_000)) == []
    trades = b.process_bar(bar(4, 10_000, 10_010, 9_990, 10_020))
    assert trades[0].exit_reason == "TIME_STOP" and trades[0].exit_price == 10_020


def test_costs_are_charged_on_actual_prices():
    cfg = ExecutionConfig(taker_fee=0.001, slippage_bps=10.0, impact_coef=0.0, funding_rate_8h=0.0)
    b = SimBroker(cfg, 100_000, 300)
    b.submit(intent(), bar(0, 10_000, 10_000, 10_000, 10_000))
    trades = b.process_bar(bar(1, 10_000, 10_500, 10_000, 10_400))
    t = trades[0]
    assert t.entry_price == pytest.approx(10_010)          # +10 bps slippage
    assert t.exit_price == pytest.approx(10_210)           # target = fill + 2R, limit fill
    assert t.fees == pytest.approx(10_010 * 0.001 + 10_210 * 0.001)
    assert b.equity(10_000) == pytest.approx(100_000 + t.net_pnl)
