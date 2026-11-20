"""Live / paper trading daemon.

Each cycle: sleep until just after the next bar close -> fetch closed bars ->
replay every bar not yet processed through the same ``TradingEngine.step``
used in backtests -> persist state atomically.

* paper mode (the only mode enabled): orders fill in ``SimBroker`` against
  live closed bars, so the fill model is identical to the backtest.
* live mode is refused (PermissionError) while ``safety.PAPER_TRADING_ONLY``.
* Integrity gate: gaps, duplicates, malformed bars, bad flow or a stale feed
  in the feature window mean no new decision on that bar (exits still run).
* Authentication, permission and bad-symbol errors halt immediately. Network
  errors spend an error budget and halt when it runs out.
* When the drawdown circuit breaker trips, the position is flattened and the
  loop stops.
* Bars are de-duplicated by timestamp, so the original's bug of re-running the
  model 5 times per 5-minute bar (60 s polling) cannot recur.
* State (risk governor, positions, model track record, last bar) survives restarts.
  The kill switch stays latched across restarts until ``--reset-kill-switch``.
* SIGINT/SIGTERM finish the current cycle and exit cleanly. Open positions keep
  their exchange-side stop and target.
* Too many consecutive errors halt new entries.
"""
from __future__ import annotations

import json
import logging
import os
import signal
import time

import pandas as pd

from . import safety
from .audit import AuditLog
from .broker import SimBroker
from .config import SystemConfig
from .data import DataIntegrityError, integrity_gate
from .engine import TradingEngine
from .features import compute_features
from .labeling import triple_barrier_labels
from .predictor import WalkForwardPredictor
from .risk import RiskGovernor

logger = logging.getLogger(__name__)


def atomic_write_json(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(data, fh, default=str)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


class LiveRunner:
    def __init__(self, cfg: SystemConfig, client, live: bool = False, reset_kill_switch: bool = False,
                 clock=None):
        if live and safety.PAPER_TRADING_ONLY:
            raise PermissionError("PAPER_TRADING_ONLY: live mode is disabled")
        self.cfg, self.client, self.live = cfg, client, live
        self.clock = clock or (lambda: pd.Timestamp.now(tz="UTC"))
        self.audit = AuditLog(cfg.live.audit_file, keep_in_memory=False)
        self.predictor = WalkForwardPredictor(cfg.model, cfg.strategy, cfg.bar_seconds)
        state = self._load_state()

        if live:
            from .exchange import LiveRouter

            self.broker = LiveRouter(client, cfg.execution, self.audit)
            initial = self.broker.realized_equity
        else:
            self.broker = SimBroker(cfg.execution, cfg.risk.initial_capital, cfg.bar_seconds)
            initial = cfg.risk.initial_capital

        if state:
            self.risk = RiskGovernor.from_dict(cfg.risk, cfg.bar_seconds, state["risk"])
            self.broker.load_dict(state["broker"])
            self.predictor.oos_load_dict(state.get("oos", {}))
            self.last_ts = pd.Timestamp(state["last_ts"]) if state.get("last_ts") else None
            logger.info("Restored state: last bar %s, equity %.2f", self.last_ts, self.risk.state.equity)
        else:
            cfg.risk.initial_capital = initial
            self.risk = RiskGovernor(cfg.risk, cfg.bar_seconds)
            self.last_ts = None
        if reset_kill_switch:
            self.risk.reset_kill_switch()
            self.audit.log("kill_switch_reset", None, equity=self.risk.state.equity)

        if live:
            mismatch = self.broker.reconcile()
            if mismatch:
                self.risk.halt(f"startup reconciliation: {mismatch}")
                self.audit.log("reconciliation_mismatch", None, detail=mismatch)

        self.engine = TradingEngine(cfg, self.broker, self.predictor, self.risk, self.audit)
        self._stop = False
        self._errors = 0

    # ---------------------------------------------------------------- state
    def _load_state(self) -> dict | None:
        path = self.cfg.live.state_file
        if not os.path.exists(path):
            return None
        with open(path) as fh:
            state = json.load(fh)
        if state.get("mode") != ("live" if self.live else "paper"):
            raise RuntimeError(f"State file {path} belongs to {state.get('mode')} mode; use a separate state_file")
        return state

    def _save_state(self) -> None:
        atomic_write_json(self.cfg.live.state_file, {
            "mode": "live" if self.live else "paper",
            "symbol": self.cfg.data.symbol,
            "last_ts": self.last_ts.isoformat() if self.last_ts is not None else None,
            "risk": self.risk.to_dict(),
            "broker": self.broker.to_dict(),
            "oos": self.predictor.oos_to_dict(),
        })

    # ----------------------------------------------------------------- loop
    def _handle_signal(self, signum, _frame) -> None:
        logger.warning("Signal %s received: stopping after this cycle", signum)
        self._stop = True

    def seconds_to_next_close(self, now: float | None = None) -> float:
        now = time.time() if now is None else now
        bs = self.cfg.bar_seconds
        return bs - (now % bs) + self.cfg.live.poll_buffer_seconds

    def warm_start(self) -> None:
        """Cold start only: replay the predictor walk-forward over recent history.

        This builds the out-of-sample track record the health gate needs, so the
        system does not have to incubate for days. It is causal (same fit/predict
        sequence as the backtest) and places no trades.
        """
        n = max(self.cfg.live.warmup_bars, self.cfg.data.history_bars)
        raw = self.client.fetch_closed_bars(n)
        feats = compute_features(raw, self.cfg.features)
        st = self.cfg.strategy
        labels = triple_barrier_labels(raw, feats["atr"], st.stop_atr_mult, st.reward_risk, st.max_holding_bars)
        start = self.cfg.model.min_train_samples + self.predictor.purge_gap
        for t in range(start, len(raw)):
            if self.predictor.needs_retrain(raw.index[t]):
                self.predictor.fit(feats, labels, t)
            self.predictor.predict(feats, t)
        self.predictor.last_train_ts = None  # force a fresh fit on the first live bar
        n_oos = {s: len(v) for s, v in self.predictor.oos.items()}
        logger.info("Warm start over %d bars: forward track record %s", len(raw), n_oos)
        self.audit.log("warm_start", raw.index[-1], bars=len(raw), oos_records=n_oos)

    def run_cycle(self) -> int:
        """Process all newly closed bars. Returns how many were processed."""
        raw = self.client.fetch_closed_bars(self.cfg.data.history_bars)
        feats = compute_features(raw, self.cfg.features)
        st = self.cfg.strategy
        labels = triple_barrier_labels(raw, feats["atr"], st.stop_atr_mult, st.reward_risk, st.max_holding_bars)
        if self.last_ts is None:
            todo = [len(raw) - 1]  # cold start: begin at the latest closed bar
        else:
            todo = [i for i in range(len(raw)) if raw.index[i] > self.last_ts]
        lookback = self.cfg.features.zscore_window + self.cfg.features.vwap_window + 2
        problems = integrity_gate(raw, self.clock(), self.cfg.bar_seconds, lookback,
                                  self.cfg.data.max_staleness_bars) if todo else []
        if problems:
            logger.warning("Integrity gate: no new decisions (%s)", ", ".join(problems))
            self.audit.log("integrity_gate_blocked", raw.index[-1], reasons=problems)
        for k, t in enumerate(todo):
            last = k == len(todo) - 1
            # Catch up on missed bars (exits, risk) but only decide on the newest one.
            self.engine.step(raw, feats, labels, t, decide=last and not problems)
            self.last_ts = raw.index[t]
        if self.live and getattr(self.broker, "reconciliation_error", ""):
            err = self.broker.reconciliation_error
            if not self.risk.state.manual_halt:
                self.risk.halt(f"reconciliation: {err}")
                self.audit.log("reconciliation_mismatch", self.last_ts, detail=err)
        if todo:
            pos = self.broker.position
            logger.info("Bar %s close=%.2f equity=%.2f dd=%.2f%% position=%s",
                        self.last_ts, raw["close"].iloc[-1], self.risk.state.equity, self.risk.drawdown * 100,
                        f"{'LONG' if pos.side > 0 else 'SHORT'} {pos.quantity:.4f}@{pos.entry_price:.2f}" if pos else "flat")
        self._save_state()
        return len(todo)

    def run(self, max_cycles: int | None = None) -> None:
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)
        mode = "LIVE" if self.live else "PAPER"
        logger.info("Starting %s trading %s %s on %s", mode, self.cfg.data.symbol, self.cfg.data.timeframe,
                    self.cfg.data.exchange)
        self.audit.log("daemon_start", None, mode=mode, config=self.cfg.to_dict())
        if self.last_ts is None and not any(self.predictor.oos.values()):
            self.warm_start()
        try:
            import ccxt

            fatal: tuple = (ccxt.AuthenticationError, ccxt.PermissionDenied, ccxt.BadSymbol,
                            ccxt.AccountSuspended, PermissionError, DataIntegrityError)
        except ImportError:  # pragma: no cover
            fatal = (PermissionError, DataIntegrityError)
        cycles = 0
        while not self._stop and (max_cycles is None or cycles < max_cycles):
            try:
                n = self.run_cycle()
                self._errors = 0
                cycles += 1
                wait = self.seconds_to_next_close() if n else 5.0  # bar not published yet: retry soon
                if self.risk.state.kill_switch:
                    logger.critical("Circuit breaker tripped (%s): stopping loop", self.risk.state.kill_reason)
                    self.audit.log("loop_halted", self.last_ts, reason="kill_switch",
                                   detail=self.risk.state.kill_reason)
                    break
            except fatal as e:
                logger.critical("Fatal error, halting: %r", e)
                self.risk.halt(f"fatal: {e!r}")
                self.audit.log("loop_halted", self.last_ts, reason="fatal_error", error=repr(e))
                break
            except Exception as e:  # network etc.: spend the error budget
                self._errors += 1
                logger.exception("Cycle failed (%d consecutive): %s", self._errors, e)
                self.audit.log("cycle_error", self.last_ts, error=repr(e), consecutive=self._errors)
                if self._errors >= self.cfg.live.max_consecutive_errors:
                    self.risk.halt(f"{self._errors} consecutive errors")
                    self.audit.log("loop_halted", self.last_ts, reason="error_budget_exhausted")
                    break
                wait = min(60.0, 2.0 ** min(self._errors, 6))
            end = time.time() + wait
            while not self._stop and time.time() < end and (max_cycles is None or cycles < max_cycles):
                time.sleep(min(1.0, max(0.0, end - time.time())))
        self._save_state()
        self.audit.log("daemon_stop", self.last_ts)
        self.audit.close()
        logger.info("Stopped.")
