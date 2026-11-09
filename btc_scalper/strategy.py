"""Turns calibrated probabilities into a trade decision via cost-aware expectancy."""
from __future__ import annotations

import math

import pandas as pd

from .config import ExecutionConfig, StrategyConfig
from .models import Signal


def round_trip_cost_r(price: float, stop_distance: float, ex: ExecutionConfig) -> float:
    """Fees + slippage for entry and exit, expressed in R (multiples of the stop distance)."""
    per_side = ex.taker_fee + ex.slippage_bps / 1e4
    return 2 * per_side * price / stop_distance


def make_signal(ts: pd.Timestamp, p_long: float, p_short: float, atr: float, price: float,
                st: StrategyConfig, ex: ExecutionConfig) -> Signal:
    stop = st.stop_atr_mult * atr if math.isfinite(atr) else float("nan")
    if not (math.isfinite(stop) and stop > 0):
        return Signal(ts, 0, p_long, p_short, float("nan"), 0.0, float("nan"), "invalid_atr")
    cost = round_trip_cost_r(price, stop, ex)
    rr = st.reward_risk
    best = (0, float("nan"), -math.inf)
    candidates = [(1, p_long)] + ([(-1, p_short)] if st.allow_short else [])
    for direction, p in candidates:
        if not math.isfinite(p):
            continue
        e = p * rr - (1 - p) - cost
        if e > best[2]:
            best = (direction, p, e)
    direction, p, e = best
    if direction == 0:
        return Signal(ts, 0, p_long, p_short, float("nan"), 0.0, stop, "model_unavailable")
    if p < st.min_probability:
        return Signal(ts, 0, p_long, p_short, p, e, stop, "probability_below_floor")
    if e < st.min_edge_r:
        return Signal(ts, 0, p_long, p_short, p, e, stop, "edge_below_threshold")
    return Signal(ts, direction, p_long, p_short, p, e, stop, "edge")
