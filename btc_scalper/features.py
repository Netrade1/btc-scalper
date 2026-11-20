"""Causal, stationary feature engineering.

Rules enforced here (and checked by ``tests/test_causality.py``):
  * a feature at bar t uses only bars <= t (no ``shift(-k)``, no centred windows);
  * only finite rolling windows, so a value does not depend on where the data
    window starts. The live system pulls a rolling window each cycle; with
    cumulative CVD/VWAP or EWMs (as in the original script) live features would
    drift away from the ones the model was backtested on;
  * features are scale-free (returns, z-scores, ATR multiples), so the model
    sees the same distribution at 20k as at 100k BTC.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FeatureConfig

FEATURE_COLUMNS = [
    "ret_1", "ret_3", "ret_6", "ret_12", "ret_24",
    "atr_pct", "rv", "rv_ratio", "rsi_c", "vwap_dist",
    "ofi_1", "ofi_fast", "ofi_slow", "ofi_accel",
    "vol_z", "clv", "body_atr", "range_atr", "mom_z",
    "hour_sin", "hour_cos",
]


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.where(b != 0)


def compute_features(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    o, h, l, c, v = (df[k] for k in ("open", "high", "low", "close", "volume"))
    bv = df["buy_volume"]
    f = pd.DataFrame(index=df.index)

    logc = np.log(c)
    for k in (1, 3, 6, 12, 24):
        f[f"ret_{k}"] = logc.diff(k)

    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    atr = tr.rolling(cfg.atr_period).mean()
    f["atr"] = atr                      # raw, used for stops (not a model input)
    f["atr_pct"] = atr / c

    r1 = f["ret_1"]
    rv = r1.rolling(cfg.vol_window).std()
    f["rv"] = rv
    f["rv_ratio"] = _safe_div(r1.rolling(max(cfg.vol_window // 4, 3)).std(), rv)

    d = c.diff()
    gain = d.clip(lower=0).rolling(cfg.rsi_period).mean()
    loss = (-d).clip(lower=0).rolling(cfg.rsi_period).mean()
    rsi = 100 - 100 / (1 + _safe_div(gain, loss))
    rsi = rsi.where(loss != 0, np.where(gain > 0, 100.0, 50.0))
    f["rsi_c"] = (rsi - 50) / 50

    typical = (h + l + c) / 3
    vwap = _safe_div((typical * v).rolling(cfg.vwap_window).sum(), v.rolling(cfg.vwap_window).sum())
    f["vwap_dist"] = _safe_div(c - vwap, atr)

    delta = 2 * bv - v                  # buy - sell volume
    f["ofi_1"] = _safe_div(delta, v)
    f["ofi_fast"] = _safe_div(delta.rolling(cfg.flow_window).sum(), v.rolling(cfg.flow_window).sum())
    f["ofi_slow"] = _safe_div(delta.rolling(cfg.vwap_window).sum(), v.rolling(cfg.vwap_window).sum())
    f["ofi_accel"] = f["ofi_fast"] - f["ofi_slow"]

    logv = np.log(v.where(v > 0))
    f["vol_z"] = _safe_div(logv - logv.rolling(cfg.zscore_window).mean(), logv.rolling(cfg.zscore_window).std())

    rng = h - l
    f["clv"] = _safe_div(c - l, rng) * 2 - 1
    f["body_atr"] = _safe_div(c - o, atr)
    f["range_atr"] = _safe_div(rng, atr)
    f["mom_z"] = _safe_div(f["ret_12"], rv * np.sqrt(12))

    if isinstance(df.index, pd.DatetimeIndex):
        hour = df.index.hour + df.index.minute / 60
        f["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        f["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    else:
        f["hour_sin"] = 0.0
        f["hour_cos"] = 0.0

    # NaNs are kept (rows are skipped at decision time) rather than dropna()'d,
    # so feature rows stay aligned 1:1 with price bars.
    return f.replace([np.inf, -np.inf], np.nan)
