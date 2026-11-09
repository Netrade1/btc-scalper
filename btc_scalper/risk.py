"""Risk governor: the hard veto between the model and the market.

Controls (each one is a reason string in the audit log when it blocks a trade):
  * kill switch on mark-to-market drawdown from peak (latches; needs manual reset)
  * daily loss limit on MTM equity vs. the UTC-day open (resets next day)
  * consecutive-loss cooldown
  * volatility regime band on ATR%
  * stale / gapped data gate
  * manual halt (e.g. after an exchange reconciliation mismatch)
Sizing: fixed-fractional risk, capped by fractional Kelly, scaled down as
drawdown deepens, and capped by a max notional leverage.
"""
from __future__ import annotations

import logging
import math
import uuid
from dataclasses import asdict, dataclass

import pandas as pd

from .config import RiskConfig
from .models import OrderIntent, Signal, Trade

logger = logging.getLogger(__name__)


@dataclass
class RiskState:
    equity: float
    peak_equity: float
    day: str = ""
    day_start_equity: float = 0.0
    kill_switch: bool = False
    kill_reason: str = ""
    day_halted: bool = False
    manual_halt: str = ""
    consecutive_losses: int = 0
    cooldown_until: str = ""
    last_bar_ts: str = ""


class RiskGovernor:
    def __init__(self, cfg: RiskConfig, bar_seconds: int, state: RiskState | None = None):
        self.cfg = cfg
        self.bar_seconds = bar_seconds
        self.state = state or RiskState(equity=cfg.initial_capital, peak_equity=cfg.initial_capital)
        self.gap_detected = False

    # ---------------------------------------------------------------- updates
    @property
    def drawdown(self) -> float:
        s = self.state
        return 0.0 if s.peak_equity <= 0 else max(0.0, (s.peak_equity - s.equity) / s.peak_equity)

    def on_bar(self, ts: pd.Timestamp, equity: float) -> list[str]:
        """Update with mark-to-market equity at bar close. Returns newly raised alerts."""
        s, alerts = self.state, []
        if s.last_bar_ts:
            gap = (ts - pd.Timestamp(s.last_bar_ts)).total_seconds()
            self.gap_detected = gap > self.cfg.max_bar_gap_multiple * self.bar_seconds
        s.last_bar_ts = ts.isoformat()
        s.equity = equity
        s.peak_equity = max(s.peak_equity, equity)
        day = ts.strftime("%Y-%m-%d")
        if day != s.day:
            s.day, s.day_start_equity, s.day_halted = day, equity, False
        if not s.kill_switch and self.drawdown >= self.cfg.max_drawdown:
            s.kill_switch = True
            s.kill_reason = f"drawdown {self.drawdown:.2%} >= {self.cfg.max_drawdown:.2%}"
            logger.critical("KILL SWITCH: %s", s.kill_reason)
            alerts.append("kill_switch")
        if not s.day_halted and s.day_start_equity > 0:
            day_loss = (s.day_start_equity - equity) / s.day_start_equity
            if day_loss >= self.cfg.daily_loss_limit:
                s.day_halted = True
                logger.warning("Daily loss limit hit (%.2f%%); no new entries until next UTC day", day_loss * 100)
                alerts.append("daily_loss_limit")
        return alerts

    def on_trade_closed(self, trade: Trade, ts: pd.Timestamp) -> None:
        s = self.state
        if trade.net_pnl < 0:
            s.consecutive_losses += 1
            if s.consecutive_losses >= self.cfg.max_consecutive_losses:
                until = ts + pd.Timedelta(seconds=self.cfg.cooldown_bars * self.bar_seconds)
                s.cooldown_until = until.isoformat()
                s.consecutive_losses = 0
                logger.warning("%d consecutive losses: cooling down until %s", self.cfg.max_consecutive_losses, until)
        else:
            s.consecutive_losses = 0

    def halt(self, reason: str) -> None:
        self.state.manual_halt = reason
        logger.critical("TRADING HALTED: %s", reason)

    def reset_kill_switch(self) -> None:
        """Operator action only. Re-bases the peak to current equity."""
        s = self.state
        s.kill_switch, s.kill_reason, s.manual_halt = False, "", ""
        s.peak_equity = s.equity

    # ----------------------------------------------------------------- checks
    def pre_trade_check(self, ts: pd.Timestamp, atr_pct: float) -> tuple[bool, str]:
        s = self.state
        if s.kill_switch:
            return False, "kill_switch"
        if s.manual_halt:
            return False, f"manual_halt:{s.manual_halt}"
        if s.day_halted:
            return False, "daily_loss_limit"
        if s.cooldown_until and ts < pd.Timestamp(s.cooldown_until):
            return False, "loss_cooldown"
        if self.gap_detected:
            return False, "data_gap"
        if not math.isfinite(atr_pct):
            return False, "no_volatility_estimate"
        if atr_pct < self.cfg.min_atr_pct:
            return False, "volatility_too_low"
        if atr_pct > self.cfg.max_atr_pct:
            return False, "volatility_too_high"
        return True, "ok"

    def size_order(self, signal: Signal, price: float, ref_volume: float, atr_pct: float,
                   horizon: int, reward_risk: float) -> tuple[OrderIntent | None, str]:
        if signal.direction == 0:
            return None, "no_signal"
        equity = self.state.equity
        b, p = reward_risk, signal.probability
        kelly = (p * b - (1 - p)) / b
        if kelly <= 0:
            return None, "negative_kelly"
        risk_frac = min(self.cfg.risk_per_trade, self.cfg.kelly_fraction * kelly)
        if self.cfg.drawdown_throttle:
            risk_frac *= max(0.0, 1.0 - self.drawdown / self.cfg.max_drawdown)
        qty = equity * risk_frac / signal.stop_distance
        qty = min(qty, self.cfg.max_leverage * equity / price)
        if not (qty > 0 and math.isfinite(qty)):
            return None, "zero_size"
        ts = signal.timestamp
        client_id = f"bs{int(ts.timestamp())}{'L' if signal.direction > 0 else 'S'}{uuid.uuid4().hex[:6]}"
        return OrderIntent(
            client_id=client_id, timestamp=ts, side=signal.direction, quantity=qty,
            stop_distance=signal.stop_distance, reward_risk=reward_risk, max_holding_bars=horizon,
            expected_r=signal.expectancy_r, probability=p, ref_price=price, ref_volume=ref_volume,
            atr_pct=atr_pct,
        ), "ok"

    # ------------------------------------------------------------ persistence
    def to_dict(self) -> dict:
        return asdict(self.state)

    @classmethod
    def from_dict(cls, cfg: RiskConfig, bar_seconds: int, d: dict) -> "RiskGovernor":
        return cls(cfg, bar_seconds, RiskState(**d))
