"""Performance and risk analytics."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .models import Trade


def _norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def probabilistic_sharpe(returns: pd.Series, benchmark_sr: float = 0.0) -> float:
    """Bailey & Lopez de Prado PSR: P(true per-period Sharpe > benchmark), adjusting for skew/kurtosis."""
    r = returns.dropna()
    n = len(r)
    if n < 30 or r.std() == 0:
        return float("nan")
    sr = r.mean() / r.std()
    skew, kurt = float(r.skew()), float(r.kurt()) + 3.0
    denom = math.sqrt(max(1e-12, 1 - skew * sr + (kurt - 1) / 4 * sr ** 2))
    return _norm_cdf((sr - benchmark_sr) * math.sqrt(n - 1) / denom)


def max_drawdown(equity: pd.Series) -> tuple[float, int]:
    peak = equity.cummax()
    dd = (equity - peak) / peak
    underwater = (dd < 0).astype(int).to_numpy()
    longest = run = 0
    for u in underwater:
        run = run + 1 if u else 0
        longest = max(longest, run)
    return float(-dd.min()) if len(dd) else 0.0, longest


def bootstrap_drawdowns(r_multiples: np.ndarray, risk_fraction: float, n_paths: int = 2000,
                        seed: int = 0) -> dict[str, float]:
    """Monte-Carlo resample of trade order: distribution of max drawdown at fixed fractional risk."""
    if len(r_multiples) < 5:
        return {}
    rng = np.random.default_rng(seed)
    sims = rng.choice(r_multiples, size=(n_paths, len(r_multiples)), replace=True)
    eq = np.cumprod(1 + risk_fraction * sims, axis=1)
    peak = np.maximum.accumulate(eq, axis=1)
    mdd = ((peak - eq) / peak).max(axis=1)
    return {"mdd_p50": float(np.percentile(mdd, 50)), "mdd_p95": float(np.percentile(mdd, 95)),
            "mdd_p99": float(np.percentile(mdd, 99))}


def performance_summary(equity: pd.Series, trades: list[Trade], bar_seconds: int,
                        initial_capital: float, exposure_bars: int | None = None,
                        risk_fraction: float = 0.0075) -> dict:
    ppy = 365 * 24 * 3600 / bar_seconds  # crypto trades 24/7
    rets = equity.pct_change().dropna()
    years = max(len(equity) / ppy, 1e-9)
    final = float(equity.iloc[-1]) if len(equity) else initial_capital
    total_ret = final / initial_capital - 1
    std = rets.std()
    downside = float(np.sqrt(np.mean(np.minimum(rets.to_numpy(), 0.0) ** 2))) if len(rets) else 0.0
    mdd, mdd_bars = max_drawdown(equity) if len(equity) else (0.0, 0)
    if final <= 0:
        cagr = -1.0
    else:
        log_growth = math.log(final / initial_capital) / years
        cagr = math.exp(min(log_growth, 700.0)) - 1  # avoid overflow on very short samples
    s = {
        "start": str(equity.index[0]) if len(equity) else None,
        "end": str(equity.index[-1]) if len(equity) else None,
        "bars": int(len(equity)),
        "initial_capital": initial_capital,
        "final_equity": final,
        "total_return": total_ret,
        "cagr": cagr,
        "ann_volatility": float(std * math.sqrt(ppy)) if std > 0 else 0.0,
        "sharpe": float(rets.mean() / std * math.sqrt(ppy)) if std > 0 else 0.0,
        "sortino": float(rets.mean() / downside * math.sqrt(ppy)) if downside and downside > 0 else 0.0,
        "max_drawdown": mdd,
        "max_drawdown_duration_days": mdd_bars * bar_seconds / 86400,
        "calmar": cagr / mdd if mdd > 0 else 0.0,
        "psr_vs_zero": probabilistic_sharpe(rets),
    }
    n = len(trades)
    s["trades"] = n
    if n:
        pnl = np.array([t.net_pnl for t in trades])
        r = np.array([t.r_multiple for t in trades])
        wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
        s.update({
            "win_rate": float((pnl > 0).mean()),
            "profit_factor": float(wins.sum() / -losses.sum()) if losses.sum() < 0 else float("inf"),
            "avg_win": float(wins.mean()) if len(wins) else 0.0,
            "avg_loss": float(losses.mean()) if len(losses) else 0.0,
            "expectancy_r": float(r.mean()),
            "expected_r_model": float(np.mean([t.expected_r for t in trades])),
            "r_std": float(r.std()),
            "avg_bars_held": float(np.mean([t.bars_held for t in trades])),
            "total_fees": float(sum(t.fees for t in trades)),
            "total_funding": float(sum(t.funding for t in trades)),
            "long_trades": int(sum(t.side > 0 for t in trades)),
            "short_trades": int(sum(t.side < 0 for t in trades)),
            "exit_reasons": pd.Series([t.exit_reason for t in trades]).value_counts().to_dict(),
            "trades_per_day": n / max(len(equity) * bar_seconds / 86400, 1e-9),
        })
        s["monte_carlo"] = bootstrap_drawdowns(r, risk_fraction)
    if exposure_bars is not None and len(equity):
        s["exposure"] = exposure_bars / len(equity)
    return s


def format_summary(s: dict) -> str:
    pct = {"total_return", "cagr", "ann_volatility", "max_drawdown", "win_rate", "exposure", "psr_vs_zero"}
    lines = []
    for k, v in s.items():
        if isinstance(v, dict):
            v = ", ".join(f"{a}={b:.2%}" if isinstance(b, float) else f"{a}={b}" for a, b in v.items())
        elif isinstance(v, float):
            v = f"{v:.2%}" if k in pct else f"{v:,.4f}"
        lines.append(f"  {k:<28} {v}")
    return "\n".join(lines)
