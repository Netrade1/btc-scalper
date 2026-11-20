"""Purged walk-forward gradient-boosting predictor with probability calibration.

Leakage controls:
  * Purging: a label at row s depends on bars s+1..s+H, so at decision time t
    only rows s <= t - H - embargo are used for training.
  * Inside each fit, the calibration slice is separated from the fit slice by
    another H + embargo rows, so calibration is genuinely out-of-sample.
  * Retraining happens on a fixed cadence (not every bar): refitting XGBoost
    every bar is slow and adds noise.

Two models are trained: P(long bracket wins) and P(short bracket wins). Raw
tree scores become probabilities in two stages, because the expectancy maths
downstream needs probabilities that hold up *going forward*:

  1. Slice calibration: Platt (or isotonic) fitted on a purged hold-out slice
     at the end of the training window.
  2. Forward recalibration: every prediction the system makes is recorded.
     Once those labels resolve, they form a genuine walk-forward track record
     of what past models said vs. what happened. A Platt map is fitted on
     that record, which absorbs model decay and selection bias. (Without it,
     a 62% forecast realised ~42% in testing.)

Health gate: a model may trade only when (a) its hold-out AUC beats
``min_auc`` and its 95% lower confidence bound beats 0.5, using an effective
sample size that corrects for overlapping labels, and (b) the forward track
record has at least ``min_oos_samples`` resolved predictions whose AUC lower
bound also beats 0.5, and whose Brier skill against the base rate beats
``min_brier_skill``. The skill is cross-fitted: the recalibration map is fitted
on the older half of the record and scored on the newer half. New models have to
earn the right to trade.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .config import ModelConfig, StrategyConfig
from .features import FEATURE_COLUMNS
from .models import ModelDiagnostics

logger = logging.getLogger(__name__)
SIDES = ("long", "short")


def auc_standard_error(auc: float, n_pos: int, n_neg: int) -> float:
    """Hanley & McNeil (1982) standard error of the ROC AUC."""
    if n_pos < 2 or n_neg < 2:
        return float("inf")
    q1, q2 = auc / (2 - auc), 2 * auc ** 2 / (1 + auc)
    var = (auc * (1 - auc) + (n_pos - 1) * (q1 - auc ** 2) + (n_neg - 1) * (q2 - auc ** 2)) / (n_pos * n_neg)
    return float(np.sqrt(max(var, 0.0)))


class PlattCalibrator:
    """Logistic map on the logit of the raw score: smooth and hard to overfit."""

    def __init__(self):
        self.lr = LogisticRegression(C=1.0)

    @staticmethod
    def _logit(p: np.ndarray) -> np.ndarray:
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p)).reshape(-1, 1)

    def fit(self, raw: np.ndarray, y) -> "PlattCalibrator":
        self.lr.fit(self._logit(raw), np.asarray(y))
        return self

    def predict(self, raw: np.ndarray) -> np.ndarray:
        return self.lr.predict_proba(self._logit(raw))[:, 1]


def make_calibrator(kind: str):
    if kind == "isotonic":
        return IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    return PlattCalibrator()


def make_classifier(cfg: ModelConfig):
    try:
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=cfg.n_estimators, max_depth=cfg.max_depth, learning_rate=cfg.learning_rate,
            subsample=cfg.subsample, colsample_bytree=cfg.colsample_bytree,
            min_child_weight=cfg.min_child_weight, reg_lambda=cfg.reg_lambda,
            random_state=cfg.seed, eval_metric="logloss", tree_method="hist", n_jobs=2, verbosity=0,
        )
    except ImportError:  # pragma: no cover - fallback when xgboost is absent
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(
            max_iter=cfg.n_estimators, max_depth=cfg.max_depth, learning_rate=cfg.learning_rate,
            l2_regularization=cfg.reg_lambda, random_state=cfg.seed,
        )


class WalkForwardPredictor:
    def __init__(self, model_cfg: ModelConfig, strategy_cfg: StrategyConfig, bar_seconds: int,
                 feature_columns: list[str] | None = None):
        self.cfg = model_cfg
        self.horizon = strategy_cfg.max_holding_bars
        self.bar_seconds = bar_seconds
        self.columns = list(feature_columns or FEATURE_COLUMNS)
        self.models: dict[str, object] = {}
        self.calibrators: dict[str, object] = {}
        self.healthy: dict[str, bool] = {s: False for s in SIDES}
        self.last_train_ts: pd.Timestamp | None = None
        self.diagnostics: list[ModelDiagnostics] = []
        self.last_train_cutoff_ts: pd.Timestamp | None = None
        self._cache: dict[pd.Timestamp, tuple[float, float]] = {}
        # Forward (out-of-sample) record: ts -> stage-1 probability, per side.
        self.oos: dict[str, dict[pd.Timestamp, float]] = {s: {} for s in SIDES}
        self.recalibrators: dict[str, PlattCalibrator | None] = {s: None for s in SIDES}
        self.slice_healthy: dict[str, bool] = {s: False for s in SIDES}

    @property
    def purge_gap(self) -> int:
        return self.horizon + self.cfg.embargo_bars

    def needs_retrain(self, ts: pd.Timestamp) -> bool:
        if self.last_train_ts is None:
            return True
        return (ts - self.last_train_ts).total_seconds() >= self.cfg.retrain_every * self.bar_seconds

    def fit(self, features: pd.DataFrame, labels: pd.DataFrame, t: int) -> list[ModelDiagnostics]:
        """Fit both side models using only information available at the close of bar t."""
        ts = features.index[t]
        cutoff = t - self.purge_gap
        self.last_train_ts = ts
        self._cache.clear()
        out: list[ModelDiagnostics] = []
        if cutoff < 0:
            for s in SIDES:
                self.healthy[s] = False
            return out
        start = max(0, cutoff - self.cfg.train_window + 1)
        X_all = features.iloc[start: cutoff + 1][self.columns]
        self.last_train_cutoff_ts = features.index[cutoff]
        for side in SIDES:
            y_all = labels.iloc[start: cutoff + 1][f"{side}_win"]
            mask = X_all.notna().all(axis=1) & y_all.notna()
            X, y = X_all[mask], y_all[mask].astype(int)
            dur = labels.iloc[start: cutoff + 1][f"{side}_bars"][mask]
            overlap = float(dur.mean()) if len(dur) else 1.0
            diag = self._fit_side(side, X, y, ts, overlap)
            self.slice_healthy[side] = diag.healthy
            diag.extra.update(self._fit_forward_recalibration(side, labels, features.index[cutoff], overlap))
            self.healthy[side] = self.slice_healthy[side] and diag.extra["oos_healthy"]
            diag.healthy = self.healthy[side]
            out.append(diag)
        self.diagnostics.extend(out)
        return out

    def _fit_side(self, side: str, X: pd.DataFrame, y: pd.Series, ts: pd.Timestamp,
                  mean_duration: float = 1.0) -> ModelDiagnostics:
        n = len(X)
        n_cal = int(n * self.cfg.calibration_fraction)
        n_fit = n - n_cal - self.purge_gap
        if n < self.cfg.min_train_samples or n_fit < self.cfg.min_train_samples // 2 or n_cal < 100:
            self.healthy[side] = False
            return ModelDiagnostics(ts, side, max(n_fit, 0), n_cal, float("nan"), float("nan"),
                                    float(y.mean()) if n else float("nan"), False, {"reason": "insufficient_data"})
        X_fit, y_fit = X.iloc[:n_fit], y.iloc[:n_fit]
        X_cal, y_cal = X.iloc[n - n_cal:], y.iloc[n - n_cal:]
        if y_fit.nunique() < 2 or y_cal.nunique() < 2:
            self.healthy[side] = False
            return ModelDiagnostics(ts, side, n_fit, n_cal, float("nan"), float("nan"), float(y.mean()),
                                    False, {"reason": "single_class"})
        model = make_classifier(self.cfg)
        model.fit(X_fit, y_fit)
        raw = model.predict_proba(X_cal)[:, 1]
        auc = float(roc_auc_score(y_cal, raw))
        # Overlapping labels are serially correlated: a label lasting d bars shares
        # most of its path with its neighbours. Use the effective sample size
        # n / mean_duration (average label uniqueness) for the confidence bound.
        overlap = max(1.0, mean_duration)
        n_pos = int(y_cal.sum())
        se = auc_standard_error(auc, int(n_pos / overlap), int((len(y_cal) - n_pos) / overlap))
        auc_lcb = auc - 1.645 * se
        cal = make_calibrator(self.cfg.calibration).fit(raw, y_cal)
        cal_p = cal.predict(raw)
        brier = float(np.mean((cal_p - y_cal.to_numpy()) ** 2))
        healthy = auc >= self.cfg.min_auc and auc_lcb > 0.5
        self.models[side], self.calibrators[side], self.healthy[side] = model, cal, healthy
        logger.debug("Retrained %s model at %s: AUC=%.3f (LCB %.3f) Brier=%.4f healthy=%s",
                     side, ts, auc, auc_lcb, brier, healthy)
        return ModelDiagnostics(ts, side, n_fit, n_cal, auc, brier, float(y_cal.mean()), healthy,
                                {"auc_lcb": auc_lcb, "label_overlap": overlap})

    def _fit_forward_recalibration(self, side: str, labels: pd.DataFrame, cutoff_ts: pd.Timestamp,
                                   overlap: float) -> dict:
        rec = self.oos[side]
        # Drop records too old to matter.
        horizon_ts = cutoff_ts - pd.Timedelta(seconds=self.cfg.oos_window * self.bar_seconds)
        for k in [k for k in rec if k < horizon_ts]:
            del rec[k]
        info = {"oos_n": 0, "oos_auc": float("nan"), "oos_auc_lcb": float("nan"),
                "oos_brier_skill": float("nan"), "oos_healthy": False}
        self.recalibrators[side] = None
        if not rec:
            return info
        ser = pd.Series(rec).sort_index()
        ser = ser[ser.index <= cutoff_ts]
        y = labels[f"{side}_win"].reindex(ser.index)
        m = y.notna() & np.isfinite(ser)
        ser, y = ser[m], y[m].astype(int)
        info["oos_n"] = int(len(ser))
        if len(ser) < self.cfg.min_oos_samples or y.nunique() < 2:
            return info
        auc = float(roc_auc_score(y, ser))
        n_pos = int(y.sum())
        se = auc_standard_error(auc, int(n_pos / overlap), int((len(y) - n_pos) / overlap))
        half = len(ser) // 2
        x, yv = ser.to_numpy(), y.to_numpy()
        bss = float("nan")
        if np.unique(yv[:half]).size == 2:
            p_hold = PlattCalibrator().fit(x[:half], yv[:half]).predict(x[half:])
            base = yv[:half].mean()
            ref = np.mean((base - yv[half:]) ** 2)
            bss = float(1 - np.mean((p_hold - yv[half:]) ** 2) / ref) if ref > 0 else float("nan")
        lcb = auc - 1.645 * se
        healthy = bool(lcb > 0.5 and np.isfinite(bss) and bss > self.cfg.min_brier_skill)
        info.update(oos_auc=auc, oos_auc_lcb=lcb, oos_brier_skill=bss, oos_healthy=healthy)
        self.recalibrators[side] = PlattCalibrator().fit(x, yv)
        return info

    def oos_to_dict(self) -> dict:
        return {s: {k.isoformat(): v for k, v in rec.items()} for s, rec in self.oos.items()}

    def oos_load_dict(self, d: dict) -> None:
        for s in SIDES:
            self.oos[s] = {pd.Timestamp(k): float(v) for k, v in d.get(s, {}).items()}

    def predict(self, features: pd.DataFrame, t: int) -> tuple[float, float]:
        """Calibrated (p_long, p_short) for bar t. NaN when the model can't be trusted.

        Predictions for the rest of the current retrain block are computed in one
        vectorised call and cached. That is still causal: feature row r only uses
        bars <= r, and the model was fit before r.
        """
        ts = features.index[t]
        if ts in self._cache:
            return self._cache[ts]
        if self.last_train_ts is None:
            return float("nan"), float("nan")
        limit = self.last_train_ts + pd.Timedelta(seconds=self.cfg.retrain_every * self.bar_seconds)
        end = int(features.index.searchsorted(limit, side="left"))
        end = max(end, t + 1)
        block = features.iloc[t:end][self.columns]
        ok = block.notna().all(axis=1).to_numpy()
        probs = {}
        for side in SIDES:
            p = np.full(len(block), np.nan)
            if side in self.models and ok.any():
                raw = self.models[side].predict_proba(block[ok])[:, 1]
                stage1 = self.calibrators[side].predict(raw)
                # Record every forward prediction (healthy or not) for the track record.
                for idx, v in zip(block.index[ok], stage1):
                    self.oos[side][idx] = float(v)
                if self.healthy.get(side) and self.recalibrators[side] is not None:
                    p[ok] = self.recalibrators[side].predict(stage1)
            probs[side] = p
        for i, idx in enumerate(block.index):
            self._cache[idx] = (float(probs["long"][i]), float(probs["short"][i]))
        return self._cache[ts]
