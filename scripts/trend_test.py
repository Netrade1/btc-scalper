"""Run the pre-registered H2 trend test (docs/hypotheses/H2_trend.md).

Usage: python scripts/trend_test.py data/kraken_BTCUSD_5m_trades.csv
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
    a = ap.parse_args()
    daily = resample_bars(load_csv(a.csv), 86400)
    r, p = row(daily, 28, 0.004)
    s_sr, b_sr = sharpe(r.strategy), sharpe(r.benchmark)
    s_dd, b_dd = max_dd(r.strategy), max_dd(r.benchmark)
    s_tot, b_tot = (1 + r.strategy).prod() - 1, (1 + r.benchmark).prod() - 1
    print(f"Daily bars {daily.index[0].date()} -> {daily.index[-1].date()} ({len(daily)}); "
          f"evaluated {len(r.strategy)} days; time in market {r.position.mean():.0%}; switches {r.switches}\n")
    print("| | Sharpe | total return | max drawdown |\n|---|---|---|---|")
    print(f"| trend 28d, 0.40% | {s_sr:+.2f} | {s_tot:+.1%} | {s_dd:.1%} |")
    print(f"| buy and hold | {b_sr:+.2f} | {b_tot:+.1%} | {b_dd:.1%} |\n")
    checks = [("Sharpe above buy-and-hold", s_sr > b_sr),
              (f"bootstrap P(diff <= 0) = {p:.3f} < 0.05", p < 0.05),
              ("drawdown no worse than buy-and-hold", s_dd >= b_dd)]
    for name, ok in checks:
        print(f"- [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"\nH2 VERDICT: {'PASS' if all(ok for _, ok in checks) else 'FAIL'}\n")
    print("Sensitivity (reported, not used to decide):\n")
    print("| lookback | fee | Sharpe | total return | max DD | switches | P(diff<=0) |\n|---|---|---|---|---|---|---|")
    for lb in (14, 28, 56):
        for fee in (0.004, 0.0026, 0.0016):
            rr, pp = row(daily, lb, fee)
            print(f"| {lb}d | {fee:.2%} | {sharpe(rr.strategy):+.2f} | {(1 + rr.strategy).prod() - 1:+.1%} | "
                  f"{max_dd(rr.strategy):.1%} | {rr.switches} | {pp:.3f} |")


if __name__ == "__main__":
    main()
