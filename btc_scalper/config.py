"""Typed, validated configuration for every layer of the system.

Every tunable lives here so a run is fully described by one TOML file
(see ``config.example.toml``). Unknown keys are rejected so typos fail loudly.
"""
from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

_TF_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}


def timeframe_seconds(timeframe: str) -> int:
    """'5m' -> 300, '1h' -> 3600."""
    unit = timeframe[-1]
    if unit not in _TF_UNITS or not timeframe[:-1].isdigit():
        raise ValueError(f"Unsupported timeframe: {timeframe!r}")
    return int(timeframe[:-1]) * _TF_UNITS[unit]


@dataclass
class DataConfig:
    exchange: str = "binance"
    symbol: str = "BTC/USDT"          # ccxt unified symbol ("BTC/USDT:USDT" = linear perpetual)
    timeframe: str = "5m"
    market_type: str = "spot"         # "spot", "swap" or "future" (ccxt defaultType)
    history_bars: int = 3600          # bars pulled each live cycle (must cover model training window)
    allow_flow_proxy: bool = False    # FAIL_CLOSED: refuse data without real taker-buy volume
    max_staleness_bars: float = 2.0   # integrity gate: newest closed bar must be this fresh


@dataclass
class FeatureConfig:
    atr_period: int = 14
    rsi_period: int = 14
    vwap_window: int = 48
    flow_window: int = 12
    vol_window: int = 48
    zscore_window: int = 96


@dataclass
class StrategyConfig:
    stop_atr_mult: float = 1.5        # stop distance = k * ATR
    reward_risk: float = 2.0          # take-profit = RR * stop distance
    max_holding_bars: int = 24        # time stop; also the label horizon
    min_edge_r: float = 0.10          # required expectancy (in R) after costs
    min_probability: float = 0.40     # floor on calibrated win probability
    allow_short: bool = False         # spot cannot be shorted without margin


@dataclass
class ModelConfig:
    train_window: int = 3000          # rolling training window (bars)
    min_train_samples: int = 800
    retrain_every: int = 288          # bars between refits (288 x 5m = 1 day)
    embargo_bars: int = 12            # extra gap on top of the label horizon
    calibration_fraction: float = 0.25
    min_auc: float = 0.53             # model health gate on the out-of-sample calibration slice
    calibration: str = "platt"        # "platt" or "isotonic"
    min_oos_samples: int = 1500       # resolved forward predictions needed before trading
    oos_window: int = 6000            # bars of forward track record used for recalibration
    min_brier_skill: float = 0.0      # forward Brier skill vs. base rate must exceed this
    n_estimators: int = 200
    max_depth: int = 3
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: float = 20.0
    reg_lambda: float = 5.0
    seed: int = 42


@dataclass
class RiskConfig:
    initial_capital: float = 100_000.0
    risk_per_trade: float = 0.0075    # max fraction of equity lost if the stop is hit
    kelly_fraction: float = 0.25      # fractional Kelly cap on the risked fraction
    max_leverage: float = 3.0         # notional / equity cap
    max_drawdown: float = 0.15        # hard kill switch on mark-to-market drawdown
    drawdown_throttle: bool = True    # scale risk down linearly as drawdown deepens
    daily_loss_limit: float = 0.03    # halt new entries for the rest of the UTC day
    max_consecutive_losses: int = 5
    cooldown_bars: int = 36
    min_atr_pct: float = 0.0008       # volatility regime band (ATR / price)
    max_atr_pct: float = 0.03
    max_bar_gap_multiple: float = 3.0 # stale-data gate


@dataclass
class ExecutionConfig:
    taker_fee: float = 0.0010         # Binance spot taker, per side
    slippage_bps: float = 1.0         # fixed half-spread + latency cost
    impact_coef: float = 0.1          # square-root market impact coefficient
    funding_rate_8h: float = 0.0      # perpetual funding paid while holding (0 for spot)


@dataclass
class LiveConfig:
    state_file: str = "state/live_state.json"
    audit_file: str = "logs/audit.jsonl"
    poll_buffer_seconds: float = 3.0
    warmup_bars: int = 8000           # cold start: replay the model walk-forward over this much history
    max_consecutive_errors: int = 10
    api_key_env: str = "EXCHANGE_API_KEY"
    api_secret_env: str = "EXCHANGE_API_SECRET"
    api_password_env: str = "EXCHANGE_API_PASSWORD"
    sandbox: bool = False


@dataclass
class SystemConfig:
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    live: LiveConfig = field(default_factory=LiveConfig)

    @property
    def bar_seconds(self) -> int:
        return timeframe_seconds(self.data.timeframe)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "SystemConfig":
        cfg = cls()
        for section, values in raw.items():
            if not hasattr(cfg, section):
                raise ValueError(f"Unknown config section [{section}]")
            target = getattr(cfg, section)
            allowed = {f.name: f for f in fields(target)}
            for key, value in values.items():
                if key not in allowed:
                    raise ValueError(f"Unknown config key [{section}].{key}")
                setattr(target, key, type(getattr(target, key))(value))
        cfg.validate()
        return cfg

    @classmethod
    def load(cls, path: str | Path | None) -> "SystemConfig":
        if path is None:
            cfg = cls()
            cfg.validate()
            return cfg
        with open(path, "rb") as fh:
            return cls.from_dict(tomllib.load(fh))

    def validate(self) -> None:
        r, s, m = self.risk, self.strategy, self.model
        checks = [
            (0 < r.risk_per_trade <= 0.05, "risk.risk_per_trade must be in (0, 0.05]"),
            (0 < r.kelly_fraction <= 1, "risk.kelly_fraction must be in (0, 1]"),
            (0 < r.max_drawdown < 1, "risk.max_drawdown must be in (0, 1)"),
            (0 < r.daily_loss_limit < 1, "risk.daily_loss_limit must be in (0, 1)"),
            (r.max_leverage > 0, "risk.max_leverage must be > 0"),
            (r.min_atr_pct < r.max_atr_pct, "risk.min_atr_pct must be < max_atr_pct"),
            (s.stop_atr_mult > 0 and s.reward_risk > 0, "strategy stop/RR must be > 0"),
            (s.max_holding_bars >= 1, "strategy.max_holding_bars must be >= 1"),
            (0 < m.calibration_fraction < 0.5, "model.calibration_fraction must be in (0, 0.5)"),
            (m.train_window >= m.min_train_samples, "model.train_window must be >= min_train_samples"),
            (m.calibration in {"platt", "isotonic"}, "model.calibration must be platt or isotonic"),
            (self.data.market_type in {"spot", "swap", "future"}, "data.market_type invalid"),
        ]
        for ok, msg in checks:
            if not ok:
                raise ValueError(msg)
        timeframe_seconds(self.data.timeframe)
        if self.data.market_type == "spot" and s.allow_short:
            raise ValueError("Spot markets cannot be shorted: set strategy.allow_short = false")
        needed = m.train_window + s.max_holding_bars + m.embargo_bars + self.features.zscore_window
        if self.data.history_bars < needed:
            raise ValueError(f"data.history_bars must be >= {needed} to train the model live")
