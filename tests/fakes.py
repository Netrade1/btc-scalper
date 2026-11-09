"""In-memory stand-in for a ccxt derivatives exchange."""
from __future__ import annotations

import itertools

import pandas as pd


class FakeExchange:
    def __init__(self, bars: pd.DataFrame | None = None, now_ms: int | None = None,
                 fail_on_stop_order: bool = False):
        self.bars = bars
        self.now_ms = now_ms
        self.orders: dict[str, dict] = {}
        self.position = 0.0  # signed base qty
        self.ids = itertools.count(1)
        self.fail_on_stop_order = fail_on_stop_order
        self.last_price = 65_000.0
        self.created: list[dict] = []
        self.markets = {"BTC/USDT": {
            "id": "BTCUSDT", "symbol": "BTC/USDT", "contract": False, "linear": None,
            "quote": "USDT", "limits": {"amount": {"min": 0.0001}, "cost": {"min": 5.0}},
        }, "BTC/USDT:USDT": {
            "id": "BTCUSDT", "symbol": "BTC/USDT:USDT", "contract": True, "linear": True,
            "contractSize": 1.0, "settle": "USDT", "quote": "USDT",
            "limits": {"amount": {"min": 0.001}, "cost": {"min": 5.0}},
        }}

    # market data
    def load_markets(self):
        return self.markets

    def market(self, symbol):
        return self.markets[symbol]

    def milliseconds(self):
        return self.now_ms

    def fetch_ohlcv(self, symbol, timeframe, since=None, limit=1000):
        df = self.bars
        rows = [[int(t.timestamp() * 1000), r.open, r.high, r.low, r.close, r.volume]
                for t, r in zip(df.index, df.itertuples())]
        if since is not None:
            rows = [r for r in rows if r[0] >= since]
        return rows[:limit]

    def publicGetKlines(self, req):
        """Binance raw kline format: index 9 is taker-buy base volume."""
        since, limit = req.get("startTime"), req.get("limit", 1000)
        out = []
        for t, r in zip(self.bars.index, self.bars.itertuples()):
            ms = int(t.timestamp() * 1000)
            if since is not None and ms < since:
                continue
            out.append([ms, str(r.open), str(r.high), str(r.low), str(r.close), str(r.volume),
                        ms + 299_999, "0", 0, str(r.buy_volume), "0", "0"])
            if len(out) >= limit:
                break
        return out

    fapiPublicGetKlines = publicGetKlines

    # trading
    def amount_to_precision(self, symbol, amount):
        return f"{float(amount):.3f}"

    def price_to_precision(self, symbol, price):
        return f"{float(price):.1f}"

    def fetch_balance(self):
        return {"total": {"USDT": 100_000.0}}

    def create_order(self, symbol, type_, side, amount, price=None, params=None):
        params = params or {}
        oid = str(next(self.ids))
        sign = 1 if side == "buy" else -1
        self.created.append({"id": oid, "type": type_, "side": side, "amount": amount, "params": params})
        if "stopLossPrice" in params or "takeProfitPrice" in params:
            if self.fail_on_stop_order and "stopLossPrice" in params:
                raise RuntimeError("exchange rejected stop order")
            trig = params.get("stopLossPrice", params.get("takeProfitPrice"))
            o = {"id": oid, "status": "open", "filled": 0.0, "amount": float(amount), "side": side,
                 "trigger": float(trig), "average": None}
        else:
            self.position += sign * float(amount)
            o = {"id": oid, "status": "closed", "filled": float(amount), "amount": float(amount), "side": side,
                 "average": self.last_price, "fee": {"cost": self.last_price * float(amount) * 0.0005}}
        self.orders[oid] = o
        return dict(o)

    def trigger(self, oid: str):
        """Simulate an exchange-side protective order firing."""
        o = self.orders[oid]
        o.update(status="closed", filled=o["amount"], average=o["trigger"],
                 fee={"cost": o["trigger"] * o["amount"] * 0.0005})
        self.position += (1 if o["side"] == "buy" else -1) * o["amount"]

    def fetch_order(self, oid, symbol=None):
        return dict(self.orders[oid])

    def cancel_order(self, oid, symbol=None):
        if self.orders[oid]["status"] == "open":
            self.orders[oid]["status"] = "canceled"

    def fetch_positions(self, symbols=None):
        if abs(self.position) < 1e-12:
            return []
        return [{"symbol": "BTC/USDT:USDT", "contracts": abs(self.position),
                 "side": "long" if self.position > 0 else "short"}]
