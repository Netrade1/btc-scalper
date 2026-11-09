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
