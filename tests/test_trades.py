import numpy as np
import pandas as pd
import pytest

from btc_scalper.config import DataConfig
from btc_scalper.exchange import CCXTClient
from btc_scalper.trades import TradeBarStore

from .fakes import FakeExchange

T0 = int(pd.Timestamp("2024-01-01", tz="UTC").timestamp() * 1000)
BAR = 300_000


def tape(n=4000, seed=0, empty_bar=None):
    rng = np.random.default_rng(seed)
    ts = np.sort(rng.integers(T0, T0 + 20 * BAR, n))
    if empty_bar is not None:  # remove all trades in one bar
        ts = ts[(ts < T0 + empty_bar * BAR) | (ts >= T0 + (empty_bar + 1) * BAR)]
    ts[100:105] = ts[100]       # several trades share a timestamp (page boundaries)
    px = 65_000 + np.cumsum(rng.normal(0, 5, len(ts)))
    return [{"id": str(i), "timestamp": int(t), "price": float(p), "amount": float(a),
             "side": "buy" if b else "sell"}
            for i, (t, p, a, b) in enumerate(zip(ts, px, rng.uniform(0.001, 0.5, len(ts)), rng.random(len(ts)) < 0.55))]


def reference_bars(trades):
    df = pd.DataFrame(trades)
    df["bar"] = df.timestamp // BAR * BAR
    df["buy"] = np.where(df.side == "buy", df.amount, 0.0)
    g = df.groupby("bar")
    return pd.DataFrame({"open": g.price.first(), "high": g.price.max(), "low": g.price.min(),
                         "close": g.price.last(), "volume": g.amount.sum(), "buy_volume": g.buy.sum()})


def store_for(ex, tmp_path, page_limit=97):
    return TradeBarStore(ex, "BTC/USD", 300, str(tmp_path / "bars.csv"), page_limit=page_limit,
                         now_ms=lambda: T0 + 20 * BAR + 10_000)


def test_bars_match_reference_aggregation_across_small_pages(tmp_path):
    ex = FakeExchange()
    ex.tape = tape()
    n = store_for(ex, tmp_path).update(start=pd.Timestamp(T0, unit="ms", tz="UTC"))
    got = store_for(ex, tmp_path).load()
    ref = reference_bars(ex.tape)
    assert n == 20 and len(got) == 20
    assert got["trades"].sum() == len(ex.tape)            # no trade lost or double counted
    np.testing.assert_allclose(got[["open", "high", "low", "close", "volume", "buy_volume"]].to_numpy(),
                               ref.to_numpy())


def test_backfill_is_resumable_and_identical(tmp_path):
    ex = FakeExchange()
    ex.tape = tape(seed=1)
    full = store_for(ex, tmp_path / "a").update(start=pd.Timestamp(T0, unit="ms", tz="UTC"))
    s = store_for(ex, tmp_path / "b")
    s.update(start=pd.Timestamp(T0, unit="ms", tz="UTC"), max_pages=7)     # interrupted
    partial = len(s.load())
    assert 0 < partial < full
    s.update()                                                            # resume
    pd.testing.assert_frame_equal(s.load(), store_for(ex, tmp_path / "a").load())


def test_empty_bar_is_flat_with_zero_volume(tmp_path):
    ex = FakeExchange()
    ex.tape = tape(seed=2, empty_bar=8)
    store_for(ex, tmp_path).update(start=pd.Timestamp(T0, unit="ms", tz="UTC"))
    df = store_for(ex, tmp_path).load()
    assert len(df) == 20
    row = df.iloc[8]
    assert row.volume == 0 and row.open == row.close == df.iloc[7].close


def test_maker_side_venues_are_inverted(tmp_path):
    ex = FakeExchange()
    ex.tape = tape(seed=3)
    s = TradeBarStore(ex, "BTC/USD", 300, str(tmp_path / "m.csv"), side_is_maker=True,
                      now_ms=lambda: T0 + 20 * BAR + 10_000)
    s.update(start=pd.Timestamp(T0, unit="ms", tz="UTC"))
    df = s.load()
    ref = reference_bars(ex.tape)
    np.testing.assert_allclose(df["buy_volume"].to_numpy(), (ref.volume - ref.buy_volume).to_numpy())


def test_client_uses_trade_store_for_non_binance(tmp_path):
    ex = FakeExchange()
    ex.markets["BTC/USD"] = dict(ex.markets["BTC/USDT"], symbol="BTC/USD")
    ex.tape = tape(seed=4)
    ex.now_ms = T0 + 20 * BAR + 10_000
    client = CCXTClient(DataConfig(exchange="kraken", symbol="BTC/USD", trade_store_dir=str(tmp_path)), exchange=ex)
    with pytest.raises(ValueError, match="fetch-trades"):
        client.fetch_closed_bars(10)                      # empty store: must backfill first
    client.trade_store().update(start=pd.Timestamp(T0, unit="ms", tz="UTC"))
    df = client.fetch_closed_bars(10)
    assert len(df) == 10 and not df.attrs["issues"]


class FlakyExchange(FakeExchange):
    """Returns an empty page now and then, like a public API hiccup."""

    def __init__(self, empty_every=5, always_empty_after=None):
        super().__init__()
        self.calls, self.empty_every, self.always_empty_after = 0, empty_every, always_empty_after

    def fetch_trades(self, symbol, since=None, limit=1000):
        self.calls += 1
        if self.calls % self.empty_every == 0:
            return []
        if self.always_empty_after is not None and since is not None and since > self.always_empty_after:
            return []
        return super().fetch_trades(symbol, since, limit)


def test_empty_page_mid_history_is_retried_not_treated_as_caught_up(tmp_path, monkeypatch):
    monkeypatch.setattr("btc_scalper.trades.time.sleep", lambda s: None)
    ex = FlakyExchange(empty_every=4)
    ex.tape = tape(seed=5)
    s = store_for(ex, tmp_path)
    s.update(start=pd.Timestamp(T0, unit="ms", tz="UTC"))
    got = s.load()
    assert len(got) == 20 and got["trades"].sum() == len(ex.tape)
    np.testing.assert_allclose(got["volume"].to_numpy(), reference_bars(ex.tape)["volume"].to_numpy())


def test_persistent_empty_feed_fails_closed_without_fake_bars(tmp_path, monkeypatch):
    monkeypatch.setattr("btc_scalper.trades.time.sleep", lambda s: None)
    ex = FlakyExchange(empty_every=10**9, always_empty_after=T0 + 5 * BAR)
    ex.tape = tape(seed=6)
    s = store_for(ex, tmp_path)
    with pytest.raises(RuntimeError, match="no data"):
        s.update(start=pd.Timestamp(T0, unit="ms", tz="UTC"))
    assert len(s.load()) <= 6          # only real bars were written; nothing invented to "now"


def test_page_of_duplicate_timestamps_does_not_stall(tmp_path):
    ex = FakeExchange()
    tr = tape(seed=7)
    for t in tr[200:600]:
        t["timestamp"] = tr[200]["timestamp"]   # 400 trades in the same millisecond
    tr.sort(key=lambda t: t["timestamp"])
    ex.tape = tr
    s = store_for(ex, tmp_path, page_limit=150)
    s.update(start=pd.Timestamp(T0, unit="ms", tz="UTC"))
    assert len(s.load()) == 20


class FakeKrakenNative(FakeExchange):
    """Kraken's raw Trades endpoint: nanosecond ``since``/``last`` cursor, float-second times."""

    id = "kraken"

    def __init__(self, tape_):
        super().__init__()
        # nanosecond stamps: trades sharing a millisecond get distinct sub-ms offsets
        self.ns = [t["timestamp"] * 1_000_000 + k for k, t in enumerate(tape_)]
        self.rows = [[str(t["price"]), str(t["amount"]), ns / 1e9, "b" if t["side"] == "buy" else "s", "m", "",
                      int(t["id"])] for t, ns in zip(tape_, self.ns)]

    def market(self, symbol):
        return {"id": "XXBTZUSD"}

    def fetch_trades(self, *a, **k):
        raise AssertionError("Kraken store must page natively, not via ccxt")

    def publicGetTrades(self, params):
        since, count = int(params["since"]), int(params["count"])
        idx = [i for i, ns in enumerate(self.ns) if ns > since][:count]
        rows = [self.rows[i] for i in idx]
        return {"result": {"XXBTZUSD": rows, "last": str(self.ns[idx[-1]] if idx else since)}}


def test_kraken_burst_of_trades_in_one_millisecond(tmp_path):
    base = tape(n=3000, seed=5)
    burst_ms = T0 + 7 * BAR + 1234
    burst = [{"id": str(10_000 + i), "timestamp": burst_ms, "price": 65_000.0, "amount": 0.0001,
              "side": "buy"} for i in range(2500)]
    trades = sorted(base + burst, key=lambda t: t["timestamp"])
    for i, t in enumerate(trades):
        t["id"] = str(i)
    ex = FakeKrakenNative(trades)
    n = store_for(ex, tmp_path, page_limit=1000).update(start=pd.Timestamp(T0, unit="ms", tz="UTC"))
    got = store_for(ex, tmp_path).load()
    assert n == 20
    assert got["trades"].sum() == len(trades)           # all 2500 burst trades counted, none stalled
    np.testing.assert_allclose(got["volume"].to_numpy(), reference_bars(trades)["volume"].to_numpy())


def test_range_with_end_stops_at_end(tmp_path):
    ex = FakeExchange()
    end_ms = T0 + 9 * BAR
    # a quiet minute before the end: the old loop crept through it 1 ms per request
    ex.tape = [t for t in tape(seed=6) if not end_ms - 60_000 <= t["timestamp"] < end_ms]
    calls = []
    real = ex.fetch_trades
    ex.fetch_trades = lambda *a, **k: calls.append(1) or real(*a, **k)
    s = store_for(ex, tmp_path)
    end = pd.Timestamp(end_ms, unit="ms", tz="UTC")
    n = s.update(start=pd.Timestamp(T0, unit="ms", tz="UTC"), end=end)
    got = s.load()
    assert len(calls) < 50
    assert n == 9 and got.index[-1] == pd.Timestamp(T0 + 8 * BAR, unit="ms", tz="UTC")
    ref = reference_bars([t for t in ex.tape if t["timestamp"] < end.timestamp() * 1000])
    np.testing.assert_allclose(got["close"].to_numpy(), ref["close"].to_numpy())
