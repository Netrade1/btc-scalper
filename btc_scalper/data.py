"""Market data: validation, order-flow proxies, CSV I/O and a synthetic generator.

The synthetic generator has regime-switching volatility, fat tails and an
optional *planted* order-flow edge (``alpha``). With ``alpha=0`` the market is
pure noise and an honest system should find nothing; with ``alpha>0`` it should
find the edge. Both cases are exercised in the test-suite.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

OHLCV = ["open", "high", "low", "close", "volume"]


def add_buy_volume_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """Estimate taker-buy volume when the venue does not publish it.

    Uses the close-location value: a bar closing at its high is treated as
    100% buyer-initiated, at its low as 100% seller-initiated. (The original
    script set buy = sell = 0.5 * volume, which made every flow feature zero
    and then NaN, so ``dropna()`` silently deleted the whole live dataset.)
    """
    out = df.copy()
    rng = (out["high"] - out["low"]).to_numpy()
    clv = np.where(rng > 0, (out["close"] - out["low"]).to_numpy() / np.where(rng > 0, rng, 1.0), 0.5)
    out["buy_volume"] = out["volume"] * clv
    out["buy_volume_is_proxy"] = True
    return out


class DataIntegrityError(ValueError):
    """Data is unusable; FAIL_CLOSED means we stop rather than guess."""


def validate_ohlcv(df: pd.DataFrame, bar_seconds: int | None = None,
                   allow_flow_proxy: bool = False) -> pd.DataFrame:
    """Sort, de-duplicate and sanity-check bars.

    Problems are recorded in ``out.attrs["issues"]`` (bar timestamp -> reason)
    so the live integrity gate can refuse to decide on damaged data. Missing
    taker-buy volume raises unless ``allow_flow_proxy`` is explicitly set.
    """
    issues: dict[str, list[pd.Timestamp]] = {}
    missing = [c for c in OHLCV if c not in df.columns]
    if missing:
        raise ValueError(f"Missing OHLCV columns: {missing}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Index must be a DatetimeIndex of bar open times")
    out = df.sort_index()
    out.index = out.index.as_unit("us")  # one resolution regardless of source
    dup = out.index.duplicated(keep="last")
    if dup.any():
        issues["duplicate_bars"] = list(out.index[dup])
    out = out[~dup]
    out = out.astype({c: float for c in OHLCV})
    bad = (
        (out[["open", "high", "low", "close"]] <= 0).any(axis=1)
        | (out["high"] < out[["open", "close"]].max(axis=1) - 1e-9)
        | (out["low"] > out[["open", "close"]].min(axis=1) + 1e-9)
        | (out["volume"] < 0)
        | out[OHLCV].isna().any(axis=1)
    )
    if bad.any():
        logger.warning("Dropping %d malformed bars", int(bad.sum()))
        issues["malformed_bars"] = list(out.index[bad])
        out = out[~bad]
    if "buy_volume" not in out.columns or out["buy_volume"].isna().all():
        if not allow_flow_proxy:
            raise DataIntegrityError(
                "No taker-buy volume in data. Order-flow features would be fake. Use a venue that "
                "publishes it (Binance klines) or set data.allow_flow_proxy = true knowingly.")
        logger.warning("Using close-location PROXY for taker-buy volume (data.allow_flow_proxy = true)")
        out = add_buy_volume_proxy(out)
    else:
        bv = out["buy_volume"].astype(float)
        bad_flow = bv.isna() | (bv < 0) | (bv > out["volume"] * (1 + 1e-9))
        if bad_flow.any():
            issues["flow_out_of_range"] = list(out.index[bad_flow])
        out["buy_volume"] = np.minimum(bv.clip(lower=0), out["volume"]).fillna(out["volume"] / 2)
    if bar_seconds is not None and len(out) > 1:
        gaps = out.index.to_series().diff().dt.total_seconds()
        gap_idx = out.index[(gaps > bar_seconds * 1.5).to_numpy()]
        if len(gap_idx):
            logger.warning("Data has %d gaps larger than one bar", len(gap_idx))
            issues["gap_before_bar"] = list(gap_idx)
    if len(out) == 0:
        raise DataIntegrityError("No valid bars after validation")
    out.attrs["issues"] = issues
    return out


def integrity_gate(df: pd.DataFrame, now: pd.Timestamp, bar_seconds: int, lookback_bars: int,
                   max_staleness_bars: float = 2.0) -> list[str]:
    """Reasons NOT to make a new decision on this data (empty list = OK).

    Checks the window the features actually use: gaps, duplicates, malformed
    bars or out-of-range flow inside it, and a stale newest bar.
    """
    reasons = []
    if len(df) < lookback_bars:
        reasons.append(f"insufficient_history:{len(df)}<{lookback_bars}")
    recent_start = df.index[-min(lookback_bars, len(df))]
    for kind, stamps in df.attrs.get("issues", {}).items():
        if any(ts >= recent_start for ts in stamps):
            reasons.append(kind)
    age = (now - (df.index[-1] + pd.Timedelta(seconds=bar_seconds))).total_seconds()
    if age > max_staleness_bars * bar_seconds:
        reasons.append(f"stale_feed:{age:.0f}s")
    return reasons


def load_csv(path: str, bar_seconds: int | None = None, allow_flow_proxy: bool = False) -> pd.DataFrame:
    """Load OHLCV from CSV. Needs a ``timestamp`` column (ISO or epoch ms)."""
    df = pd.read_csv(path)
    ts_col = next((c for c in ("timestamp", "time", "date", "datetime") if c in df.columns), None)
    if ts_col is None:
        raise ValueError("CSV needs a timestamp column")
    ts = df[ts_col]
    idx = pd.to_datetime(ts, unit="ms", utc=True) if pd.api.types.is_numeric_dtype(ts) else pd.to_datetime(ts, utc=True)
    df = df.drop(columns=[ts_col]).set_index(pd.DatetimeIndex(idx, name="timestamp"))
    return validate_ohlcv(df, bar_seconds, allow_flow_proxy)


def save_csv(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index_label="timestamp")


def generate_synthetic_ohlcv(
    n_bars: int = 20_000,
    bar_seconds: int = 300,
    seed: int = 7,
    alpha: float = 0.0,
    flow_persistence: float = 0.9,
    base_vol: float = 0.0015,
    start: str = "2024-01-01",
) -> pd.DataFrame:
    """Synthetic BTC-like bars.

    alpha : strength of the planted edge. Order-flow imbalance is persistent
            (AR(1) with ``flow_persistence``) and the next bar's expected return
            is ``alpha * vol * imbalance`` of the current bar, i.e. an order-flow
            continuation effect that lasts several bars.
    """
    rng = np.random.default_rng(seed)
    # Two-state Markov volatility regime (calm / stressed)
    regime = np.zeros(n_bars, dtype=int)
    for i in range(1, n_bars):
        stay = 0.995 if regime[i - 1] == 0 else 0.98
        regime[i] = regime[i - 1] if rng.random() < stay else 1 - regime[i - 1]
    vol = base_vol * np.where(regime == 1, 2.5, 1.0)
    # Intraday seasonality of volatility
    idx = pd.date_range(start=start, periods=n_bars, freq=f"{bar_seconds}s", tz="UTC")
    hour = idx.hour.to_numpy() + idx.minute.to_numpy() / 60
    vol = vol * (1 + 0.3 * np.sin(2 * np.pi * (hour - 8) / 24))

    # Persistent order-flow imbalance in [-1, 1]
    ofi = np.zeros(n_bars)
    noise = rng.normal(0, 0.25 * np.sqrt(1 - flow_persistence ** 2), n_bars)
    for i in range(1, n_bars):
        ofi[i] = np.clip(flow_persistence * ofi[i - 1] + noise[i], -0.95, 0.95)
    # What the exchange reports is a noisy per-bar reading of the latent flow.
    observed = np.clip(ofi + rng.normal(0, 0.15, n_bars), -0.95, 0.95)

    shocks = rng.standard_t(df=4, size=n_bars) / np.sqrt(2.0)  # unit-variance t(4)
    drift = np.zeros(n_bars)
    drift[1:] = alpha * vol[1:] * ofi[:-1]
    rets = drift + vol * shocks

    close = 65_000.0 * np.exp(np.cumsum(rets))
    open_ = np.empty(n_bars)
    open_[0] = 65_000.0
    open_[1:] = close[:-1] * np.exp(rng.normal(0, 0.05, n_bars - 1) * vol[1:])
    wick = np.abs(rng.normal(0, 0.6, (2, n_bars))) * vol
    high = np.maximum(open_, close) * np.exp(wick[0])
    low = np.minimum(open_, close) * np.exp(-wick[1])
    volume = rng.lognormal(mean=4.0, sigma=0.5, size=n_bars) * np.where(regime == 1, 1.8, 1.0)
    buy_volume = volume * (0.5 + 0.5 * observed)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume, "buy_volume": buy_volume},
        index=pd.DatetimeIndex(idx, name="timestamp"),
    )
