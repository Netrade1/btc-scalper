"""Synthetic replay matrix: random walk vs. planted edge, at two cost levels.

Validates the *mechanics* (no trading on noise, edge capture, cost sensitivity).
It says nothing about whether a real market edge exists.
Usage: python scripts/replay_report.py [--bars 20000] [--seeds 7 11 23]
"""
import argparse
import logging

from btc_scalper.backtest import run_backtest
from btc_scalper.config import SystemConfig
from btc_scalper.data import generate_synthetic_ohlcv

COSTS = {
    "perp 0.05% L/S": {"data": {"market_type": "swap", "symbol": "BTC/USDT:USDT"},
                       "strategy": {"allow_short": True},
                       "execution": {"taker_fee": 0.0005, "funding_rate_8h": 0.0001}},
    "spot 0.10% long": {},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bars", type=int, default=20_000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[7, 11, 23])
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.0, 1.0])
    ap.add_argument("--costs", nargs="+", default=list(COSTS), help="subset of cost levels")
    ap.add_argument("--no-header", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.ERROR)
    if not args.no_header:
        print("| market | cost level | seed | trades | win % | avg R | return | max DD | Sharpe |")
        print("|---|---|---|---|---|---|---|---|---|")
    for alpha in args.alphas:
        for name, over in ((k, COSTS[k]) for k in args.costs):
            cfg = SystemConfig.from_dict(over)
            for seed in args.seeds:
                s = run_backtest(generate_synthetic_ohlcv(args.bars, seed=seed, alpha=alpha), cfg).summary
                mkt = "random walk" if alpha == 0 else f"planted edge a={alpha}"
                print(f"| {mkt} | {name} | {seed} | {s['trades']} | {s.get('win_rate', 0):.0%} | "
                      f"{s.get('expectancy_r', 0):+.2f} | {s['total_return']:+.1%} | {s['max_drawdown']:.1%} | "
                      f"{s['sharpe']:+.2f} |", flush=True)


if __name__ == "__main__":
    main()
