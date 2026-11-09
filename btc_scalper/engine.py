"""One decision step per closed bar. Backtest, paper and live all call this."""
from __future__ import annotations

import logging
import math

import pandas as pd

from .audit import AuditLog
from .broker import Broker
from .config import SystemConfig
from .models import Bar
from .predictor import WalkForwardPredictor
from .risk import RiskGovernor
from .strategy import make_signal

logger = logging.getLogger(__name__)


def bar_at(raw: pd.DataFrame, t: int) -> Bar:
    r = raw.iloc[t]
    return Bar(raw.index[t], float(r["open"]), float(r["high"]), float(r["low"]), float(r["close"]),
               float(r["volume"]), float(r["buy_volume"]))


class TradingEngine:
    def __init__(self, cfg: SystemConfig, broker: Broker, predictor: WalkForwardPredictor,
                 risk: RiskGovernor, audit: AuditLog):
        self.cfg, self.broker, self.predictor, self.risk, self.audit = cfg, broker, predictor, risk, audit
        self._last_block_reason = ""

    def step(self, raw: pd.DataFrame, features: pd.DataFrame, labels: pd.DataFrame, t: int,
             decide: bool = True) -> float:
        """Process bar t (already closed). Returns mark-to-market equity at its close."""
        bar = bar_at(raw, t)
        ts = bar.timestamp

        for trade in self.broker.process_bar(bar):
            self.risk.on_trade_closed(trade, ts)
            self.audit.log("trade_closed", ts, **trade.to_dict())

        equity = self.broker.equity(bar.close)
        for alert in self.risk.on_bar(ts, equity):
            self.audit.log("risk_alert", ts, alert=alert, equity=equity, drawdown=self.risk.drawdown,
                           detail=self.risk.state.kill_reason)
        if self.risk.state.kill_switch and not self.broker.is_flat():
            trade = self.broker.close_position(bar, "KILL_SWITCH")
            if trade:
                self.risk.on_trade_closed(trade, ts)
                self.audit.log("trade_closed", ts, **trade.to_dict())
            equity = self.broker.equity(bar.close)
            self.risk.state.equity = equity

        if decide:
            self._decide(raw, features, labels, t, bar)
        return equity

    def _decide(self, raw, features, labels, t: int, bar: Bar) -> None:
        ts = bar.timestamp
        # Keep the retrain cadence regardless of whether we are allowed to trade.
        if self.predictor.needs_retrain(ts):
            for d in self.predictor.fit(features, labels, t):
                self.audit.log("model_retrained", ts, side=d.side, n_train=d.n_train,
                               n_calibration=d.n_calibration, auc=d.auc, brier=d.brier,
                               base_rate=d.base_rate, healthy=d.healthy, **d.extra)
        # Predict every bar (not only when flat) so the forward track record is complete.
        p_long, p_short = self.predictor.predict(features, t)
        if not self.broker.is_flat():
            return
        frow = features.iloc[t]
        atr_pct = float(frow["atr_pct"])
        ok, reason = self.risk.pre_trade_check(ts, atr_pct)
        if not ok:
            if reason != self._last_block_reason:
                self.audit.log("entry_blocked", ts, reason=reason)
            self._last_block_reason = reason
            return
        self._last_block_reason = ""
        signal = make_signal(ts, p_long, p_short, float(frow["atr"]), bar.close,
                             self.cfg.strategy, self.cfg.execution)
        if signal.direction == 0:
            return
        intent, why = self.risk.size_order(signal, bar.close, bar.volume, atr_pct,
                                           self.cfg.strategy.max_holding_bars, self.cfg.strategy.reward_risk)
        if intent is None:
            self.audit.log("order_vetoed", ts, reason=why, probability=signal.probability,
                           expectancy_r=signal.expectancy_r)
            return
        self.audit.log("order_submitted", ts, client_id=intent.client_id, side=intent.side,
                       quantity=intent.quantity, ref_price=intent.ref_price,
                       stop_distance=intent.stop_distance, probability=intent.probability,
                       expected_r=intent.expected_r, p_long=p_long if math.isfinite(p_long) else None,
                       p_short=p_short if math.isfinite(p_short) else None)
        self.broker.submit(intent, bar)
