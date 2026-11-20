"""Walk-forward replay of real bars at several fee levels, plus model diagnostics.

Usage: python scripts/real_replay.py data/kraken_BTCUSD_5m_trades.csv [--timeframe 1h]
       [--fees 0.004 0.0026 0.001 0.0005] [--shorts]

The CSV may hold shorter bars than --timeframe; they are combined (5m -> 1h / 4h).
"""
import argparse
import logging

import numpy as np

from btc_scalper.backtest import prepare, run_backtest
from btc_scalper.config import SystemConfig
from btc_scalper.config import timeframe_seconds
from btc_scalper.data import load_csv, resample_bars


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--fees", type=float, nargs="+", default=[0.004, 0.0026, 0.001, 0.0005])
    ap.add_argument("--shorts", action="store_true", help="also allow shorts (perp-style)")
    ap.add_argument("--timeframe", default=None, help="e.g. 1h or 4h (default: the config's)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.ERROR)
    common = {"data": {}}
    if args.timeframe:
        common["data"]["timeframe"] = args.timeframe
        # Two years is only ~4,400 4h bars: shrink the windows so a forward track
        # record can exist at all. The health gate's thresholds are unchanged.
        if timeframe_seconds(args.timeframe) >= 4 * 3600:
            common["model"] = {"train_window": 1500, "min_train_samples": 600, "retrain_every": 72,
                               "min_oos_samples": 600, "oos_window": 3000}
    base = SystemConfig.from_dict(common)
    raw = load_csv(args.csv)
    src_seconds = int((raw.index[1] - raw.index[0]).total_seconds())
    if base.bar_seconds != src_seconds:
        raw = resample_bars(raw, base.bar_seconds)
    _, feats, labels = prepare(raw, base)
    atr_pct = float(feats["atr_pct"].median())
    print(f"Data: {raw.index[0]} -> {raw.index[-1]}  ({len(raw)} bars), median ATR {atr_pct:.3%}, "
          f"price {raw.close.iloc[0]:,.0f} -> {raw.close.iloc[-1]:,.0f}")
    print(f"Timeframe {base.data.timeframe}. Base rate TP-before-SL (1.5 ATR stop, 2R target, {base.strategy.max_holding_bars} bars): long {labels.long_win.mean():.1%}, "
          f"short {labels.short_win.mean():.1%} (break-even before costs: 33.3%)\n")
    print("| taker fee | cost (R) | trades | win % | avg R | return | max DD | Sharpe | healthy share | fwd AUC |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for fee in args.fees:
        over = {k: dict(v) for k, v in common.items()}
        over["execution"] = {"taker_fee": fee}
        if args.shorts:
            over["data"]["market_type"] = "swap"
            over["strategy"] = {"allow_short": True}
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
