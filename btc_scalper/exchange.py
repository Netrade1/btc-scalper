"""CCXT integration: closed-bar data feed and a real-order broker.

Fixes relative to the original live loop:
  * The newest candle returned by ``fetch_ohlcv`` is still forming. Trading on
    it is exactly the repainting the original claimed to avoid. We drop any bar
    whose close time is in the future.
  * Real taker-buy volume comes from Binance raw klines (index 9). If the venue
    does not publish it, data loading fails closed unless
    ``data.allow_flow_proxy`` is set; it never silently uses a 50/50 split.
  * Orders are actually sent (the original had ``create_market_order``
    commented out yet still "filled" in memory), sized to exchange precision
    and limits, and protected by exchange-side reduce-only stop/take-profit
    orders. If the protective orders cannot be placed, the position is
    flattened immediately.
  * Exchange position state is reconciled every bar; any mismatch halts new
    entries for operator review.

``LiveRouter`` is the reviewed real-order implementation, but it refuses to
construct while ``safety.PAPER_TRADING_ONLY`` is True (the shipped default).
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any

import pandas as pd

from .audit import AuditLog
from .config import DataConfig, ExecutionConfig, LiveConfig, timeframe_seconds
from . import safety
from .data import DataIntegrityError, validate_ohlcv
from .models import Bar, OrderIntent, Position, Trade

logger = logging.getLogger(__name__)


def _retry(fn, *args, attempts: int = 4, **kwargs):
    import ccxt

    delay = 1.0
    for i in range(attempts):
        try:
            return fn(*args, **kwargs)
        except (ccxt.NetworkError, ccxt.ExchangeNotAvailable, ccxt.RequestTimeout) as e:
            if i == attempts - 1:
                raise
            logger.warning("Network error (%s); retry %d in %.0fs", e, i + 1, delay)
            time.sleep(delay)
            delay *= 2


class CCXTClient:
    """Thin wrapper around a ccxt exchange for market data."""

    def __init__(self, data_cfg: DataConfig, live_cfg: LiveConfig | None = None, authenticated: bool = False,
                 exchange: Any = None):
        self.cfg = data_cfg
        self.bar_seconds = timeframe_seconds(data_cfg.timeframe)
        if exchange is not None:
            self.exchange = exchange
        else:
            import ccxt

            params: dict[str, Any] = {"enableRateLimit": True, "options": {"defaultType": data_cfg.market_type}}
            if authenticated:
                assert live_cfg is not None
                key, secret = os.environ.get(live_cfg.api_key_env), os.environ.get(live_cfg.api_secret_env)
                if not key or not secret:
                    raise RuntimeError(f"Set {live_cfg.api_key_env} and {live_cfg.api_secret_env} for live trading")
                params.update(apiKey=key, secret=secret)
                if os.environ.get(live_cfg.api_password_env):
                    params["password"] = os.environ[live_cfg.api_password_env]
            self.exchange = getattr(ccxt, data_cfg.exchange)(params)
            if live_cfg is not None and live_cfg.sandbox:
                self.exchange.set_sandbox_mode(True)
        _retry(self.exchange.load_markets)
        self.market = self.exchange.market(data_cfg.symbol)

    # ------------------------------------------------------------------ data
    def _binance_klines(self, since: int | None, limit: int) -> list | None:
        """Raw Binance klines include taker-buy base volume (column 9)."""
        if self.cfg.exchange not in {"binance", "binanceusdm"}:
            return None
        method = "fapiPublicGetKlines" if self.market.get("linear") else "publicGetKlines"
        if not hasattr(self.exchange, method):
            return None
        req = {"symbol": self.market["id"], "interval": self.cfg.timeframe, "limit": limit}
        if since is not None:
            req["startTime"] = since
        rows = _retry(getattr(self.exchange, method), req)
        return [[int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5]), float(r[9])]
                for r in rows]

    def _fetch_page(self, since: int | None, limit: int) -> list:
        try:
            rows = self._binance_klines(since, limit)
            if rows is not None:
                return rows
        except Exception as e:  # fall back to unified OHLCV (flow then needs the explicit proxy opt-in)
            logger.warning("Raw klines unavailable (%s); falling back to fetch_ohlcv", e)
        rows = _retry(self.exchange.fetch_ohlcv, self.cfg.symbol, self.cfg.timeframe, since, limit)
        return [r + [float("nan")] for r in rows]

    def fetch_closed_bars(self, n_bars: int, page_limit: int = 1000) -> pd.DataFrame:
        """The last ``n_bars`` fully closed bars, paginating as needed."""
        now_ms = int(self.exchange.milliseconds())
        tf_ms = self.bar_seconds * 1000
        since = now_ms - (n_bars + 2) * tf_ms
        rows: list = []
        while since < now_ms:
            page = self._fetch_page(since, page_limit)
            if not page:
                break
            rows.extend(page)
            nxt = int(page[-1][0]) + tf_ms
            if nxt <= since:
                break
            since = nxt
        return self._to_frame(rows, now_ms).iloc[-n_bars:]

    def fetch_range(self, start: pd.Timestamp, end: pd.Timestamp | None = None, page_limit: int = 1000) -> pd.DataFrame:
        now_ms = int(self.exchange.milliseconds())
        end_ms = int((end or pd.Timestamp.now(tz="UTC")).timestamp() * 1000)
        since, tf_ms, rows = int(start.timestamp() * 1000), self.bar_seconds * 1000, []
        while since < min(end_ms, now_ms):
            page = self._fetch_page(since, page_limit)
            if not page:
                break
            rows.extend(page)
            since = int(page[-1][0]) + tf_ms
            logger.info("Fetched %d bars (to %s)", len(rows), pd.Timestamp(since, unit="ms", tz="UTC"))
        df = self._to_frame(rows, now_ms)
        return df[df.index < pd.Timestamp(end_ms, unit="ms", tz="UTC")]

    def _to_frame(self, rows: list, now_ms: int) -> pd.DataFrame:
        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume", "buy_volume"])
        df = df.drop_duplicates("ts").sort_values("ts")
        # Drop the still-forming bar: closed only if open_time + timeframe <= now.
        df = df[df["ts"] + self.bar_seconds * 1000 <= now_ms]
        df.index = pd.DatetimeIndex(pd.to_datetime(df.pop("ts"), unit="ms", utc=True), name="timestamp")
        if df["buy_volume"].isna().all():
            df = df.drop(columns=["buy_volume"])
        if df.empty:
            raise DataIntegrityError("Exchange returned no closed bars")
        return validate_ohlcv(df, self.bar_seconds, self.cfg.allow_flow_proxy)


class LiveRouter:
    """Routes orders to a derivatives venue with exchange-side protective orders.

    BLOCKED: raises PermissionError while ``safety.PAPER_TRADING_ONLY`` is True.

    Equity is tracked locally from the starting balance plus realised fills
    (venues disagree on whether "total" balance includes unrealised PnL), and
    position state is reconciled against the exchange every bar.
    """

    def __init__(self, client: CCXTClient, exec_cfg: ExecutionConfig, audit: AuditLog,
                 initial_equity: float | None = None):
        if safety.PAPER_TRADING_ONLY:
            raise PermissionError(
                "PAPER_TRADING_ONLY: live order routing is disabled. It requires a passing replay on real "
                "history, a passing paper run, and Risk Chair approval before safety.py is changed.")
        self.client, self.ex, self.cfg, self.audit = client, client.exchange, exec_cfg, audit
        self.symbol = client.cfg.symbol
        self.market = client.market
        if not self.market.get("contract"):
            raise ValueError("Live order routing supports derivatives (swap/future) only; use paper mode for spot")
        self.contract_size = float(self.market.get("contractSize") or 1.0)
        self.position: Position | None = None
        self.sl_id: str | None = None
        self.tp_id: str | None = None
        self.trades: list[Trade] = []
        self.reconciliation_error: str = ""
        if initial_equity is None:
            bal = _retry(self.ex.fetch_balance)
            settle = self.market.get("settle") or self.market.get("quote")
            initial_equity = float(bal.get("total", {}).get(settle) or 0.0)
        self.realized_equity = float(initial_equity)

    # ------------------------------------------------------------- helpers
    def is_flat(self) -> bool:
        return self.position is None

    def _contracts(self, base_qty: float) -> float:
        return float(self.ex.amount_to_precision(self.symbol, base_qty / self.contract_size))

    def _fee_of(self, order: dict, price: float, qty: float) -> float:
        fee = (order or {}).get("fee") or {}
        if fee.get("cost") is not None:
            return float(fee["cost"])
        return price * qty * self.cfg.taker_fee

    def _resolve(self, order: dict) -> dict:
        """Ensure we know a market order's average price and filled amount."""
        if order.get("average") and order.get("filled"):
            return order
        for _ in range(5):
            o = _retry(self.ex.fetch_order, order["id"], self.symbol)
            if o.get("status") == "closed" and o.get("filled"):
                return o
            time.sleep(0.5)
        return o

    def exchange_position_qty(self) -> float:
        """Signed base-currency position on the exchange."""
        positions = _retry(self.ex.fetch_positions, [self.symbol])
        qty = 0.0
        for p in positions:
            if p.get("symbol") != self.symbol:
                continue
            contracts = float(p.get("contracts") or 0.0)
            sign = -1 if p.get("side") == "short" else 1
            qty += sign * contracts * self.contract_size
        return qty

    def reconcile(self) -> str:
        """Compare local and exchange positions. Returns a mismatch description or ''."""
        qty = self.exchange_position_qty()
        local = 0.0 if self.position is None else self.position.side * self.position.quantity
        tol = max(abs(local), abs(qty), 1e-12) * 0.01 + 1e-9
        if abs(qty - local) > tol and not (self.position is not None and abs(qty) < 1e-12):
            return f"exchange position {qty} != local {local}"
        return ""

    # ----------------------------------------------------------------- orders
    def submit(self, intent: OrderIntent, bar: Bar) -> None:
        if self.position is not None:
            raise RuntimeError("Already in a position")
        side = "buy" if intent.side > 0 else "sell"
        amount = self._contracts(intent.quantity)
        limits = self.market.get("limits", {})
        min_amt = (limits.get("amount") or {}).get("min") or 0.0
        min_cost = (limits.get("cost") or {}).get("min") or 0.0
        if amount <= 0 or amount < min_amt or amount * self.contract_size * intent.ref_price < min_cost:
            self.audit.log("order_rejected_local", intent.timestamp, reason="below_exchange_minimum",
                           amount=amount, min_amount=min_amt, min_cost=min_cost)
            return
        # No automatic retry on order placement: a retried market order could
        # double-fill. A failed/unknown submission is caught by reconciliation.
        order = self.ex.create_order(self.symbol, "market", side, amount, None,
                                     {"clientOrderId": intent.client_id})
        order = self._resolve(order)
        filled = float(order.get("filled") or 0.0) * self.contract_size
        fill_px = float(order.get("average") or order.get("price") or intent.ref_price)
        if filled <= 0:
            self.audit.log("order_unfilled", intent.timestamp, order_id=order.get("id"))
            return
        fee = self._fee_of(order, fill_px, filled)
        self.realized_equity -= fee
        stop = float(self.ex.price_to_precision(self.symbol, fill_px - intent.side * intent.stop_distance))
        tp = float(self.ex.price_to_precision(
            self.symbol, fill_px + intent.side * intent.reward_risk * intent.stop_distance))
        self.position = Position(
            side=intent.side, quantity=filled, entry_price=fill_px, stop_price=stop, take_profit=tp,
            entry_time=bar.timestamp, risk_amount=filled * intent.stop_distance,
            max_holding_bars=intent.max_holding_bars, expected_r=intent.expected_r,
            client_id=intent.client_id, fees_paid=fee,
        )
        self.audit.log("order_filled", intent.timestamp, order_id=order.get("id"), side=side,
                       quantity=filled, price=fill_px, fee=fee)
        exit_side = "sell" if intent.side > 0 else "buy"
        contracts = self._contracts(filled)
        try:
            sl = self.ex.create_order(self.symbol, "market", exit_side, contracts, None,
                                      {"stopLossPrice": stop, "reduceOnly": True})
            self.sl_id = sl["id"]
            tpo = self.ex.create_order(self.symbol, "market", exit_side, contracts, None,
                                       {"takeProfitPrice": tp, "reduceOnly": True})
            self.tp_id = tpo["id"]
            self.audit.log("bracket_placed", intent.timestamp, stop=stop, take_profit=tp,
                           sl_id=self.sl_id, tp_id=self.tp_id)
        except Exception as e:
            logger.critical("Failed to place protective orders (%s): flattening", e)
            self.audit.log("bracket_failed", intent.timestamp, error=str(e))
            self.close_position(bar, "BRACKET_FAILURE")

    def _cancel(self, order_id: str | None) -> None:
        if not order_id:
            return
        try:
            self.ex.cancel_order(order_id, self.symbol)
        except Exception as e:  # already filled/cancelled
            logger.debug("cancel %s: %s", order_id, e)

    def _order_if_filled(self, order_id: str | None) -> dict | None:
        if not order_id:
            return None
        try:
            o = _retry(self.ex.fetch_order, order_id, self.symbol)
        except Exception:
            return None
        return o if o.get("status") == "closed" and o.get("filled") else None

    def _book_trade(self, exit_px: float, exit_fee: float, ts: pd.Timestamp, reason: str) -> Trade:
        pos = self.position
        assert pos is not None
        gross = pos.side * pos.quantity * (exit_px - pos.entry_price)
        self.realized_equity += gross - exit_fee
        fees = pos.fees_paid + exit_fee
        net = gross - fees - pos.funding_paid
        trade = Trade(pos.client_id, pos.side, pos.quantity, pos.entry_time, ts, pos.entry_price, exit_px,
                      gross, fees, pos.funding_paid, net, net / pos.risk_amount if pos.risk_amount else 0.0,
                      pos.expected_r, pos.bars_held, reason)
        self.trades.append(trade)
        self.position, self.sl_id, self.tp_id = None, None, None
        return trade

    def process_bar(self, bar: Bar) -> list[Trade]:
        """Called after each bar closes: detect bracket exits, apply time stop, reconcile."""
        closed: list[Trade] = []
        if self.position is not None:
            pos = self.position
            pos.bars_held += 1
            funding = abs(pos.quantity * bar.close) * self.cfg.funding_rate_8h * self.client.bar_seconds / 28800
            pos.funding_paid += funding
            self.realized_equity -= funding  # estimate; exchange settles actual funding
            exch_qty = self.exchange_position_qty()
            if abs(exch_qty) < 1e-12:
                for oid, other, reason in ((self.sl_id, self.tp_id, "STOP"), (self.tp_id, self.sl_id, "TARGET")):
                    o = self._order_if_filled(oid)
                    if o:
                        self._cancel(other)
                        px = float(o.get("average") or o.get("price"))
                        closed.append(self._book_trade(px, self._fee_of(o, px, pos.quantity), bar.timestamp, reason))
                        break
                else:
                    self._cancel(self.sl_id)
                    self._cancel(self.tp_id)
                    closed.append(self._book_trade(bar.close, bar.close * pos.quantity * self.cfg.taker_fee,
                                                   bar.timestamp, "EXTERNAL_CLOSE"))
            elif pos.bars_held >= pos.max_holding_bars:
                t = self.close_position(bar, "TIME_STOP")
                if t:
                    closed.append(t)
        self.reconciliation_error = self.reconcile()
        return closed

    def close_position(self, bar: Bar, reason: str) -> Trade | None:
        if self.position is None:
            return None
        self._cancel(self.sl_id)
        self._cancel(self.tp_id)
        pos = self.position
        side = "sell" if pos.side > 0 else "buy"
        order = self.ex.create_order(self.symbol, "market", side, self._contracts(pos.quantity), None,
                                     {"reduceOnly": True})
        order = self._resolve(order)
        px = float(order.get("average") or order.get("price") or bar.close)
        trade = self._book_trade(px, self._fee_of(order, px, pos.quantity), bar.timestamp, reason)
        self.audit.log("position_closed", bar.timestamp, reason=reason, price=px)
        return trade

    def equity(self, mark: float) -> float:
        return self.realized_equity + (self.position.unrealized(mark) if self.position else 0.0)

    def to_dict(self) -> dict:
        return {"realized_equity": self.realized_equity, "sl_id": self.sl_id, "tp_id": self.tp_id,
                "position": self.position.to_dict() if self.position else None}

    def load_dict(self, d: dict) -> None:
        self.realized_equity = d["realized_equity"]
        self.sl_id, self.tp_id = d.get("sl_id"), d.get("tp_id")
        self.position = Position.from_dict(d["position"]) if d.get("position") else None
