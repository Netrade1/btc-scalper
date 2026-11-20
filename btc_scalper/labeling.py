"""Triple-barrier labels that match the trade the system actually takes.

The original script trained on "is the next close higher?" but traded a
1.5 ATR stop / 3 ATR target bracket, then plugged that next-bar probability
into a 2:1 expectancy formula. Those are different bets. Here the label for
bar s answers exactly: *if we enter at the next bar's open with this bracket,
does the take-profit get hit before the stop within the holding horizon?*

Same-bar ambiguity (both barriers touched) counts as a loss, matching the
simulator. Rows whose outcome is not fully known yet are NaN.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _shift_up(a: np.ndarray, k: int) -> np.ndarray:
    out = np.full_like(a, np.nan, dtype=float)
    if k < len(a):
        out[: len(a) - k] = a[k:]
    return out


def triple_barrier_labels(
    df: pd.DataFrame, atr: pd.Series, stop_mult: float, reward_risk: float, horizon: int
) -> pd.DataFrame:
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    n = len(df)
    entry = _shift_up(o, 1)
    dist = stop_mult * atr.to_numpy(float)
    valid = np.isfinite(entry) & np.isfinite(dist) & (dist > 0)

    out = {}
    for name, side in (("long", 1), ("short", -1)):
        stop = entry - side * dist
        tp = entry + side * reward_risk * dist
        r = np.full(n, np.nan)
        dur = np.full(n, np.nan)
        done = ~valid
        for k in range(1, horizon + 1):
            hk, lk, ck = _shift_up(h, k), _shift_up(l, k), _shift_up(c, k)
            avail = np.isfinite(hk) & ~done
            if side == 1:
                hit_sl = avail & (lk <= stop)
                hit_tp = avail & ~hit_sl & (hk >= tp)
            else:
                hit_sl = avail & (hk >= stop)
                hit_tp = avail & ~hit_sl & (lk <= tp)
            r[hit_sl] = -1.0
            r[hit_tp] = reward_risk
            dur[hit_sl | hit_tp] = k
            done = done | hit_sl | hit_tp
            if k == horizon:
                timeout = avail & ~hit_sl & ~hit_tp
                r[timeout] = side * (ck[timeout] - entry[timeout]) / dist[timeout]
                dur[timeout] = k
        known = np.arange(n) + horizon <= n - 1
        r[~known] = np.nan
        dur[~known] = np.nan
        win = np.where(np.isnan(r), np.nan, (r >= reward_risk - 1e-12).astype(float))
        out[f"{name}_r"] = r
        out[f"{name}_win"] = win
        out[f"{name}_bars"] = dur  # bars until resolution (drives label overlap)
    return pd.DataFrame(out, index=df.index)
