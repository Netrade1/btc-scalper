"""Walk-forward replay of real bars at several fee levels, plus model diagnostics.

Usage: python scripts/real_replay.py data/kraken_BTCUSD_5m_trades.csv [--fees 0.004 0.0026 0.001 0.0005]
"""
import argparse
import logging

import numpy as np

from btc_scalper.backtest import prepare, run_backtest
from btc_scalper.config import SystemConfig
from btc_scalper.data import load_csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--fees", type=float, nargs="+", default=[0.004, 0.0026, 0.001, 0.0005])
    ap.add_argument("--shorts", action="store_true", help="also allow shorts (perp-style)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.ERROR)
    base = SystemConfig.load(None)
    raw = load_csv(args.csv, base.bar_seconds)
    _, feats, labels = prepare(raw, base)
    atr_pct = float(feats["atr_pct"].median())
    print(f"Data: {raw.index[0]} -> {raw.index[-1]}  ({len(raw)} bars), median ATR {atr_pct:.3%}, "
          f"price {raw.close.iloc[0]:,.0f} -> {raw.close.iloc[-1]:,.0f}")
    print(f"Base rate TP-before-SL (1.5 ATR stop, 2R target, 24 bars): long {labels.long_win.mean():.1%}, "
          f"short {labels.short_win.mean():.1%} (break-even before costs: 33.3%)\n")
    print("| taker fee | cost (R) | trades | win % | avg R | return | max DD | Sharpe | healthy share | fwd AUC |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for fee in args.fees:
        over = {"execution": {"taker_fee": fee}}
        if args.shorts:
            over.update({"data": {"market_type": "swap"}, "strategy": {"allow_short": True}})
        cfg = SystemConfig.from_dict(over)
        res = run_backtest(raw, cfg)
        s, d = res.summary, res.model_diagnostics
        cost_r = 2 * (fee + cfg.execution.slippage_bps / 1e4) / (cfg.strategy.stop_atr_mult * atr_pct)
        fwd = [x.get("oos_auc") for x in d["extra"]] if len(d) else []
        fwd = np.nanmean([v for v in fwd if v is not None]) if fwd else float("nan")
        print(f"| {fee:.2%} | {cost_r:.1f} | {s['trades']} | {s.get('win_rate', 0):.0%} | "
              f"{s.get('expectancy_r', 0):+.2f} | {s['total_return']:+.1%} | {s['max_drawdown']:.1%} | "
              f"{s['sharpe']:+.2f} | {s.get('model_healthy_share', 0):.0%} | {fwd:.3f} |", flush=True)


if __name__ == "__main__":
    main()
