"""Broker interface and an event-driven bar simulator.

Fill model (SimBroker), designed to be pessimistic:
  * Orders decided at the close of bar t fill at the OPEN of bar t+1 (the
    original filled at the same close it used for the decision).
  * Entry/stop/time-stop fills pay fixed slippage plus square-root impact
    ``impact_coef * vol * sqrt(qty / bar_volume)``.
  * The bracket is anchored to the actual fill price, so 1R is exact.
  * Gaps through a stop fill at the open, not at the stop price.
  * If one bar touches both stop and target, the stop is assumed first.
  * Take-profit fills at the limit price; fees are charged at the real exit
    price (the original charged exit fees on the bar close).
  * Perpetual funding accrues on notional while the position is open.
"""
from __future__ import annotations

import math
from typing import Protocol

import pandas as pd

from .config import ExecutionConfig
from .models import Bar, OrderIntent, Position, Trade


class Broker(Protocol):
    position: Position | None

    def is_flat(self) -> bool: ...
    def submit(self, intent: OrderIntent, bar: Bar) -> None: ...
    def process_bar(self, bar: Bar) -> list[Trade]: ...
    def close_position(self, bar: Bar, reason: str) -> Trade | None: ...
    def equity(self, mark: float) -> float: ...


class SimBroker:
    def __init__(self, cfg: ExecutionConfig, initial_capital: float, bar_seconds: int):
        self.cfg = cfg
        self.bar_seconds = bar_seconds
        self.cash = initial_capital
        self.position: Position | None = None
        self.pending: OrderIntent | None = None
        self.trades: list[Trade] = []
        self.total_fees = 0.0

    def is_flat(self) -> bool:
        return self.position is None and self.pending is None

    def submit(self, intent: OrderIntent, bar: Bar) -> None:
        if not self.is_flat():
            raise RuntimeError("SimBroker supports one position/order at a time")
        self.pending = intent

    def _slip(self, price: float, qty: float, side: int, ref_volume: float, vol: float) -> float:
        """Price after adverse slippage for an order of ``side`` (+1 buy, -1 sell)."""
        impact = 0.0
        if ref_volume > 0 and vol > 0:
            impact = self.cfg.impact_coef * vol * math.sqrt(qty / ref_volume)
        return price * (1 + side * (self.cfg.slippage_bps / 1e4 + impact))

    def _fee(self, price: float, qty: float) -> float:
        fee = price * qty * self.cfg.taker_fee
        self.total_fees += fee
        return fee

    def process_bar(self, bar: Bar) -> list[Trade]:
        closed: list[Trade] = []
        just_filled = False
        if self.pending is not None:
            it, self.pending = self.pending, None
            fill = self._slip(bar.open, it.quantity, it.side, it.ref_volume, it.atr_pct)
            fee = self._fee(fill, it.quantity)
            self.cash -= fee
            self.position = Position(
                side=it.side, quantity=it.quantity, entry_price=fill,
                stop_price=fill - it.side * it.stop_distance,
                take_profit=fill + it.side * it.reward_risk * it.stop_distance,
                entry_time=bar.timestamp, risk_amount=it.quantity * it.stop_distance,
                max_holding_bars=it.max_holding_bars, expected_r=it.expected_r,
                client_id=it.client_id, fees_paid=fee,
            )
            just_filled = True

        pos = self.position
        if pos is None:
            return closed
        s = pos.side
        exit_px, reason = None, ""
        vol = (bar.high - bar.low) / bar.close if bar.close > 0 else 0.0
        adverse_open = (bar.open - pos.stop_price) * s <= 0
        favourable_open = (bar.open - pos.take_profit) * s >= 0
        if not just_filled and adverse_open:
            exit_px, reason = self._slip(bar.open, pos.quantity, -s, bar.volume, vol), "STOP_GAP"
        elif (bar.low <= pos.stop_price) if s > 0 else (bar.high >= pos.stop_price):
            exit_px, reason = self._slip(pos.stop_price, pos.quantity, -s, bar.volume, vol), "STOP"
        elif not just_filled and favourable_open:
            exit_px, reason = bar.open, "TARGET_GAP"
        elif (bar.high >= pos.take_profit) if s > 0 else (bar.low <= pos.take_profit):
            exit_px, reason = pos.take_profit, "TARGET"

        # Funding accrues for the portion of the bar the position was open.
        funding = abs(pos.quantity * bar.close) * self.cfg.funding_rate_8h * self.bar_seconds / (8 * 3600)
        pos.funding_paid += funding
        self.cash -= funding
        pos.bars_held += 1

        if exit_px is None and pos.bars_held >= pos.max_holding_bars:
            exit_px, reason = self._slip(bar.close, pos.quantity, -s, bar.volume, vol), "TIME_STOP"
        if exit_px is not None:
            closed.append(self._close(exit_px, bar.timestamp, reason))
        return closed

    def _close(self, exit_px: float, ts: pd.Timestamp, reason: str) -> Trade:
        pos = self.position
        assert pos is not None
        fee = self._fee(exit_px, pos.quantity)
        gross = pos.side * pos.quantity * (exit_px - pos.entry_price)
        self.cash += gross - fee
        fees = pos.fees_paid + fee
        net = gross - fees - pos.funding_paid
        trade = Trade(
            client_id=pos.client_id, side=pos.side, quantity=pos.quantity, entry_time=pos.entry_time,
            exit_time=ts, entry_price=pos.entry_price, exit_price=exit_px, gross_pnl=gross, fees=fees,
            funding=pos.funding_paid, net_pnl=net, r_multiple=net / pos.risk_amount if pos.risk_amount else 0.0,
            expected_r=pos.expected_r, bars_held=pos.bars_held, exit_reason=reason,
        )
        self.trades.append(trade)
        self.position = None
        return trade

    def close_position(self, bar: Bar, reason: str) -> Trade | None:
        self.pending = None
        if self.position is None:
            return None
        vol = (bar.high - bar.low) / bar.close if bar.close > 0 else 0.0
        px = self._slip(bar.close, self.position.quantity, -self.position.side, bar.volume, vol)
        return self._close(px, bar.timestamp, reason)

    def equity(self, mark: float) -> float:
        return self.cash + (self.position.unrealized(mark) if self.position else 0.0)

    # Persistence for paper trading
    def to_dict(self) -> dict:
        return {"cash": self.cash, "total_fees": self.total_fees,
                "position": self.position.to_dict() if self.position else None,
                "pending": None if self.pending is None else {
                    **{k: getattr(self.pending, k) for k in self.pending.__dataclass_fields__},
                    "timestamp": self.pending.timestamp.isoformat()}}

    def load_dict(self, d: dict) -> None:
        self.cash = d["cash"]
        self.total_fees = d.get("total_fees", 0.0)
        self.position = Position.from_dict(d["position"]) if d.get("position") else None
        p = d.get("pending")
        self.pending = OrderIntent(**{**p, "timestamp": pd.Timestamp(p["timestamp"])}) if p else None
