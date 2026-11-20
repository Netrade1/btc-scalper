"""Run a pre-registered trend test (docs/hypotheses/H2_trend.md, H3_trend14_holdout.md).

Usage: python scripts/trend_test.py data/kraken_BTCUSD_5m_trades.csv               # H2 (28 days)
       python scripts/trend_test.py data/holdout/kraken_BTCUSD_5m_trades.csv --lookback 14   # H3
"""
import argparse

from btc_scalper.data import load_csv, resample_bars
from btc_scalper.trend import block_bootstrap_sharpe_diff, max_dd, sharpe, trend_returns


def row(daily, lookback, fee):
    r = trend_returns(daily, lookback=lookback, fee=fee)
    p, _ = block_bootstrap_sharpe_diff(r.strategy, r.benchmark)
    return r, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--lookback", type=int, default=28, help="the pre-registered lookback in days")
    a = ap.parse_args()
    daily = resample_bars(load_csv(a.csv), 86400)
    r, p = row(daily, a.lookback, 0.004)
    s_sr, b_sr = sharpe(r.strategy), sharpe(r.benchmark)
    s_dd, b_dd = max_dd(r.strategy), max_dd(r.benchmark)
    s_tot, b_tot = (1 + r.strategy).prod() - 1, (1 + r.benchmark).prod() - 1
    print(f"Daily bars {daily.index[0].date()} -> {daily.index[-1].date()} ({len(daily)}); "
          f"evaluated {len(r.strategy)} days; time in market {r.position.mean():.0%}; switches {r.switches}\n")
    print("| | Sharpe | total return | max drawdown |\n|---|---|---|---|")
    print(f"| trend {a.lookback}d, 0.40% | {s_sr:+.2f} | {s_tot:+.1%} | {s_dd:.1%} |")
    print(f"| buy and hold | {b_sr:+.2f} | {b_tot:+.1%} | {b_dd:.1%} |\n")
    checks = [("Sharpe above buy-and-hold", s_sr > b_sr),
              (f"bootstrap P(diff <= 0) = {p:.3f} < 0.05", p < 0.05),
              ("drawdown no worse than buy-and-hold", s_dd >= b_dd)]
    for name, ok in checks:
        print(f"- [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"\nVERDICT ({a.lookback}d, 0.40%): {'PASS' if all(ok for _, ok in checks) else 'FAIL'}\n")
    print("By calendar year (reported, not used to decide):\n")
    print("| year | trend return | buy-and-hold return | time in market |\n|---|---|---|---|")
    for y, g in r.strategy.groupby(r.strategy.index.year):
        b = r.benchmark[g.index]
        print(f"| {y} | {(1 + g).prod() - 1:+.1%} | {(1 + b).prod() - 1:+.1%} | {r.position[g.index].mean():.0%} |")
    print()
    print("Sensitivity (reported, not used to decide):\n")
    print("| lookback | fee | Sharpe | total return | max DD | switches | P(diff<=0) |\n|---|---|---|---|---|---|---|")
    for lb in (14, 28, 56):
        for fee in (0.004, 0.0026, 0.0016):
            rr, pp = row(daily, lb, fee)
            print(f"| {lb}d | {fee:.2%} | {sharpe(rr.strategy):+.2f} | {(1 + rr.strategy).prod() - 1:+.1%} | "
                  f"{max_dd(rr.strategy):.1%} | {rr.switches} | {pp:.3f} |")


if __name__ == "__main__":
    main()
