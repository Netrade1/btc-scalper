import pytest

from btc_scalper.config import SystemConfig

_FAST_MODEL = {"train_window": 1500, "min_train_samples": 600, "retrain_every": 288,
               "n_estimators": 80, "min_oos_samples": 1000}

# Cost levels. "perp" is a perpetual/VIP taker (0.05%, long/short); "spot" is the shipped
# default (Binance spot taker 0.10%, long-only).
PERP = {"data": {"market_type": "swap", "symbol": "BTC/USDT:USDT", "history_bars": 2000},
        "strategy": {"allow_short": True},
        "execution": {"taker_fee": 0.0005, "funding_rate_8h": 0.0001}}


@pytest.fixture
def fast_cfg() -> SystemConfig:
    """Perp cost level with smaller model windows so tests run quickly (mechanics tests)."""
    return SystemConfig.from_dict({**PERP, "model": _FAST_MODEL})


@pytest.fixture
def fast_spot_cfg() -> SystemConfig:
    """Shipped spot defaults (long-only, 0.10% taker) with small model windows."""
    return SystemConfig.from_dict({"model": _FAST_MODEL, "data": {"history_bars": 2000}})
