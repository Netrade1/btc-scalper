"""No look-ahead: features, labels and model fits must not see the future."""
import numpy as np
import pandas as pd

from btc_scalper.backtest import prepare
from btc_scalper.config import FeatureConfig
from btc_scalper.data import generate_synthetic_ohlcv
from btc_scalper.features import compute_features
from btc_scalper.labeling import triple_barrier_labels
from btc_scalper.predictor import WalkForwardPredictor


def test_features_identical_on_truncated_history():
    raw = generate_synthetic_ohlcv(1500, seed=1)
    cfg = FeatureConfig()
    full = compute_features(raw, cfg)
    for t in (300, 800, 1499):
        trunc = compute_features(raw.iloc[: t + 1], cfg)
        pd.testing.assert_series_equal(full.iloc[t], trunc.iloc[t], check_names=False, rtol=1e-9)


def test_features_independent_of_window_start():
    """Live pulls a rolling window: the start point must not change values."""
    raw = generate_synthetic_ohlcv(1500, seed=2)
    cfg = FeatureConfig()
    a = compute_features(raw, cfg).iloc[-1]
    b = compute_features(raw.iloc[500:], cfg).iloc[-1]
    np.testing.assert_allclose(a.to_numpy(float), b.to_numpy(float), rtol=1e-8)


def test_labels_only_use_horizon_and_are_nan_when_unresolved():
    raw = generate_synthetic_ohlcv(800, seed=3)
    atr = compute_features(raw, FeatureConfig())["atr"]
    H = 24
    full = triple_barrier_labels(raw, atr, 1.5, 2.0, H)
    s = 400
    trunc = triple_barrier_labels(raw.iloc[: s + H + 1], atr.iloc[: s + H + 1], 1.5, 2.0, H)
    assert full.iloc[s].equals(trunc.iloc[s])
    assert full.iloc[-H:].isna().all().all()


def test_predictor_ignores_labels_after_purge_cutoff(fast_cfg):
    raw, feats, labels = prepare(generate_synthetic_ohlcv(3000, seed=4, alpha=1.0), fast_cfg)
    t = 2500
    p1 = WalkForwardPredictor(fast_cfg.model, fast_cfg.strategy, fast_cfg.bar_seconds)
    p1.fit(feats, labels, t)
    corrupted = labels.copy()
    cutoff = t - p1.purge_gap
    corrupted.iloc[cutoff + 1:] = 1.0 - corrupted.iloc[cutoff + 1:].fillna(0)
    p2 = WalkForwardPredictor(fast_cfg.model, fast_cfg.strategy, fast_cfg.bar_seconds)
    p2.fit(feats, corrupted, t)
    raw_1 = p1.models["long"].predict_proba(feats[p1.columns].iloc[[t]])[:, 1]
    raw_2 = p2.models["long"].predict_proba(feats[p2.columns].iloc[[t]])[:, 1]
    np.testing.assert_allclose(raw_1, raw_2)
