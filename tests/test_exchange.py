import pandas as pd
import pytest

from btc_scalper import safety
from btc_scalper.audit import AuditLog
from btc_scalper.config import DataConfig, ExecutionConfig
from btc_scalper.data import DataIntegrityError, generate_synthetic_ohlcv
from btc_scalper.exchange import CCXTClient, LiveRouter
from btc_scalper.models import Bar, OrderIntent

from .fakes import FakeExchange

T0 = pd.Timestamp("2024-01-01", tz="UTC")


def mkbar(c=65_000.0):
    return Bar(T0, c, c + 10, c - 10, c, 100, 50)


def mkintent(side=1):
    return OrderIntent("cid1", T0, side, 0.5, 300.0, 2.0, 3, 0.2, 0.5, 65_000.0, 100, 0.002)


def _bars_and_exchange():
    bars = generate_synthetic_ohlcv(50, seed=1, start="2024-01-01")
    last_open_ms = int(bars.index[-1].timestamp() * 1000)
    return bars, FakeExchange(bars, now_ms=last_open_ms + 60_000)  # newest bar only 1 min old


def test_client_drops_forming_bar_and_uses_real_taker_volume():
    bars, ex = _bars_and_exchange()
    df = CCXTClient(DataConfig(exchange="binance"), exchange=ex).fetch_closed_bars(100)
    assert df.index[-1] == bars.index[-2]
    assert (df.index == bars.index[:-1]).all()
    assert df["buy_volume"].to_numpy() == pytest.approx(bars["buy_volume"].iloc[:-1].to_numpy())


def test_missing_taker_volume_fails_closed_unless_opted_in():
    bars, ex = _bars_and_exchange()
    with pytest.raises(DataIntegrityError):
        CCXTClient(DataConfig(exchange="kraken"), exchange=ex).fetch_closed_bars(100)
    df = CCXTClient(DataConfig(exchange="kraken", allow_flow_proxy=True), exchange=ex).fetch_closed_bars(100)
    assert df["buy_volume"].between(0, df["volume"]).all() and df["buy_volume_is_proxy"].all()


def test_live_router_refuses_while_paper_only():
    assert safety.PAPER_TRADING_ONLY
    ex = FakeExchange()
    client = CCXTClient(DataConfig(exchange="binance", symbol="BTC/USDT:USDT"), exchange=ex)
    with pytest.raises(PermissionError):
        LiveRouter(client, ExecutionConfig(), AuditLog())


def make_broker(monkeypatch, **kw):
    # Router logic is exercised against a fake exchange only, by lifting the lock in-test.
    monkeypatch.setattr(safety, "PAPER_TRADING_ONLY", False)
    ex = FakeExchange(**kw)
    client = CCXTClient(DataConfig(exchange="binance", symbol="BTC/USDT:USDT"), exchange=ex)
    client.bar_seconds = 300
    return ex, LiveRouter(client, ExecutionConfig(), AuditLog())


def test_submit_places_reduce_only_bracket_and_books_stop(monkeypatch):
    ex, br = make_broker(monkeypatch)
    br.submit(mkintent(), mkbar())
    assert br.position.quantity == pytest.approx(0.5)
    kinds = [o["params"] for o in ex.created]
    assert kinds[1]["stopLossPrice"] == pytest.approx(64_700) and kinds[1]["reduceOnly"]
    assert kinds[2]["takeProfitPrice"] == pytest.approx(65_600)
    ex.trigger(br.sl_id)
    trades = br.process_bar(mkbar(64_700))
    assert trades[0].exit_reason == "STOP" and br.position is None
    assert ex.orders[ex.created[2]["id"]]["status"] == "canceled"
    assert trades[0].r_multiple < -1.0     # -1R plus fees
    assert br.reconcile() == ""


def test_time_stop_closes_with_reduce_only_market(monkeypatch):
    ex, br = make_broker(monkeypatch)
    br.submit(mkintent(side=-1), mkbar())
    for _ in range(2):
        assert br.process_bar(mkbar()) == []
    trades = br.process_bar(mkbar())
    assert trades[0].exit_reason == "TIME_STOP" and abs(ex.position) < 1e-12
    assert ex.created[-1]["params"] == {"reduceOnly": True}


def test_bracket_failure_flattens_immediately(monkeypatch):
    ex, br = make_broker(monkeypatch, fail_on_stop_order=True)
    br.submit(mkintent(), mkbar())
    assert br.position is None and abs(ex.position) < 1e-12
    assert br.trades[-1].exit_reason == "BRACKET_FAILURE"


def test_reconcile_detects_orphan_position(monkeypatch):
    ex, br = make_broker(monkeypatch)
    ex.position = 0.3
    assert "exchange position" in br.reconcile()
