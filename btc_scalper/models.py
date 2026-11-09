"""Core value objects shared by every layer."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class Bar:
    timestamp: pd.Timestamp   # bar OPEN time (exchange convention)
    open: float
    high: float
    low: float
    close: float
    volume: float
    buy_volume: float         # taker-buy volume (real or proxied)


@dataclass(frozen=True)
class Signal:
    timestamp: pd.Timestamp
    direction: int            # +1 long, -1 short, 0 flat
    p_long: float
    p_short: float
    probability: float
    expectancy_r: float       # expected R after costs for the chosen side
    stop_distance: float
    reason: str


@dataclass(frozen=True)
class OrderIntent:
    client_id: str
    timestamp: pd.Timestamp   # decision time (close of the signal bar)
    side: int                 # +1 buy, -1 sell
    quantity: float
    stop_distance: float      # bracket is re-anchored on the actual fill price
    reward_risk: float
    max_holding_bars: int
    expected_r: float
    probability: float
    ref_price: float
    ref_volume: float
    atr_pct: float


@dataclass
class Position:
    side: int
    quantity: float
    entry_price: float
    stop_price: float
    take_profit: float
    entry_time: pd.Timestamp
    risk_amount: float        # quantity * stop distance (1R in currency)
    max_holding_bars: int
    expected_r: float
    client_id: str
    bars_held: int = 0
    fees_paid: float = 0.0
    funding_paid: float = 0.0

    def unrealized(self, mark: float) -> float:
        return self.side * self.quantity * (mark - self.entry_price)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["entry_time"] = self.entry_time.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Position":
        d = dict(d)
        d["entry_time"] = pd.Timestamp(d["entry_time"])
        return cls(**d)


@dataclass(frozen=True)
class Trade:
    client_id: str
    side: int
    quantity: float
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    gross_pnl: float
    fees: float
    funding: float
    net_pnl: float
    r_multiple: float
    expected_r: float
    bars_held: int
    exit_reason: str

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["entry_time"] = self.entry_time.isoformat()
        d["exit_time"] = self.exit_time.isoformat()
        return d


@dataclass
class ModelDiagnostics:
    timestamp: pd.Timestamp
    side: str
    n_train: int
    n_calibration: int
    auc: float
    brier: float
    base_rate: float
    healthy: bool
    extra: dict[str, Any] = field(default_factory=dict)
