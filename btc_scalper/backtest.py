"""Event-driven walk-forward backtest using the exact live decision path."""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field

import pandas as pd

from .audit import AuditLog
from .broker import SimBroker
from .config import SystemConfig
from .data import validate_ohlcv
from .engine import TradingEngine
from .features import compute_features
from .labeling import triple_barrier_labels
from .metrics import performance_summary
from .predictor import WalkForwardPredictor
from .risk import RiskGovernor

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    summary: dict
    equity: pd.Series
    trades: pd.DataFrame
    model_diagnostics: pd.DataFrame
    audit: list = field(default_factory=list)


def prepare(raw: pd.DataFrame, cfg: SystemConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = validate_ohlcv(raw, cfg.bar_seconds, cfg.data.allow_flow_proxy)
    feats = compute_features(raw, cfg.features)
    st = cfg.strategy
    labels = triple_barrier_labels(raw, feats["atr"], st.stop_atr_mult, st.reward_risk, st.max_holding_bars)
    return raw, feats, labels


def run_backtest(raw: pd.DataFrame, cfg: SystemConfig, out_dir: str | None = None) -> BacktestResult:
    raw, feats, labels = prepare(raw, cfg)
    audit = AuditLog(os.path.join(out_dir, "audit.jsonl") if out_dir else None)
    broker = SimBroker(cfg.execution, cfg.risk.initial_capital, cfg.bar_seconds)
    predictor = WalkForwardPredictor(cfg.model, cfg.strategy, cfg.bar_seconds)
    risk = RiskGovernor(cfg.risk, cfg.bar_seconds)
    engine = TradingEngine(cfg, broker, predictor, risk, audit)

    equity = []
    exposure = 0
    n = len(raw)
    for t in range(n):
        # Never open a trade we cannot see the end of within the data.
        equity.append(engine.step(raw, feats, labels, t, decide=t < n - 1))
        exposure += broker.position is not None
        if t and t % 5000 == 0:
            logger.info("Backtest %d/%d bars | equity %.2f | trades %d", t, n, equity[-1], len(broker.trades))
    if broker.position is not None:  # mark-out at the end
        from .engine import bar_at
        broker.close_position(bar_at(raw, n - 1), "END_OF_DATA")
        equity[-1] = broker.equity(float(raw["close"].iloc[-1]))
    audit.close()

    eq = pd.Series(equity, index=raw.index, name="equity")
    summary = performance_summary(eq, broker.trades, cfg.bar_seconds, cfg.risk.initial_capital,
                                  exposure, cfg.risk.risk_per_trade)
    diags = pd.DataFrame([asdict(d) for d in predictor.diagnostics])
    if len(diags):
        summary["model_retrains"] = int(len(diags) // 2)
        summary["model_mean_auc"] = float(diags["auc"].mean())
        summary["model_healthy_share"] = float(diags["healthy"].mean())
    summary["kill_switch"] = risk.state.kill_switch
    trades = pd.DataFrame([t.to_dict() for t in broker.trades])

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        eq.to_csv(os.path.join(out_dir, "equity.csv"))
        trades.to_csv(os.path.join(out_dir, "trades.csv"), index=False)
        diags.to_csv(os.path.join(out_dir, "model_diagnostics.csv"), index=False)
        with open(os.path.join(out_dir, "summary.json"), "w") as fh:
            json.dump(summary, fh, indent=2, default=str)
        with open(os.path.join(out_dir, "config.json"), "w") as fh:
            json.dump(cfg.to_dict(), fh, indent=2)
    return BacktestResult(summary, eq, trades, diags, audit.records or [])
