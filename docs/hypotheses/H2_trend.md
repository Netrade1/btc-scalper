# H2: Time-series trend on BTC/USD spot (pre-registered)

Written and committed **before** the test was run. Do not edit after the run except to append results.

## Hypothesis
Holding BTC only while its price is above its level 28 days earlier (otherwise cash) earns a higher
risk-adjusted return than buy-and-hold, after Kraken's base taker fee. Basis: time-series momentum
(Moskowitz, Ooi & Pedersen 2012) and crypto momentum (Liu & Tsyvinski 2021).

## Fixed design (no tuning)
- Data: Kraken BTC/USD trades, 2024-11-12 to 2026-11-09, combined into UTC daily bars
  (`data/kraken_BTCUSD_5m_trades.csv`). Known before the run: BTC fell from about 89,000 to 74,000 over
  the window; the daily path was not examined.
- Signal at the close of day t: `close[t] / close[t-28] - 1 > 0` → long, else flat.
- Execution: trade at the open of day t+1. Daily return = open-to-open. Long-only, 100% notional, no leverage.
- Costs: 0.40% taker fee + 1 bp slippage per side, charged on every change in position.
- Benchmark: buy-and-hold over the same days (entry cost charged once).
- The first 28 days are warm-up for both legs.

## Pass rule (all three)
1. Strategy annualized Sharpe (365 days) > buy-and-hold Sharpe.
2. Circular block bootstrap of the daily Sharpe difference (strategy − buy-and-hold), block 20 days,
   10,000 resamples, seed 0: P(difference ≤ 0) < 5%.
3. Strategy max drawdown is no worse than buy-and-hold's.

## Reported but not used to decide
Lookbacks 14 and 56 days and fees 0.26% / 0.16%, to show sensitivity. A pass at 28 days with failures
elsewhere is fragile; a fail at 28 days is a fail even if another setting looks better.

## Known limits
About 700 daily returns: low statistical power. A pass is weak evidence; a fail does not rule out trend
in general. A pass leads to a paper run under the Risk Governor, not to live trading.

## Result (appended after the run): FAIL

`python scripts/trend_test.py data/kraken_BTCUSD_5m_trades.csv`

Daily bars 2024-11-12 -> 2026-11-08 (727); evaluated 697 days; time in market 52%; switches 72

| | Sharpe | total return | max drawdown |
|---|---|---|---|
| trend 28d, 0.40% | -0.54 | -28.4% | -44.9% |
| buy and hold | -0.08 | -21.9% | -53.1% |

- [FAIL] Sharpe above buy-and-hold
- [FAIL] bootstrap P(diff <= 0) = 0.780 < 0.05
- [PASS] drawdown no worse than buy-and-hold

H2 VERDICT: FAIL

Sensitivity (reported, not used to decide):

| lookback | fee | Sharpe | total return | max DD | switches | P(diff<=0) |
|---|---|---|---|---|---|---|
| 14d | 0.40% | -0.06 | -9.5% | -36.1% | 98 | 0.556 |
| 14d | 0.26% | +0.20 | +3.8% | -29.8% | 98 | 0.381 |
| 14d | 0.16% | +0.39 | +14.6% | -24.9% | 98 | 0.265 |
| 28d | 0.40% | -0.54 | -28.4% | -44.9% | 72 | 0.780 |
| 28d | 0.26% | -0.34 | -20.8% | -40.5% | 72 | 0.672 |
| 28d | 0.16% | -0.19 | -14.8% | -37.1% | 72 | 0.582 |
| 56d | 0.40% | -0.36 | -21.2% | -36.4% | 40 | 0.701 |
| 56d | 0.26% | -0.25 | -16.6% | -34.4% | 40 | 0.631 |
| 56d | 0.16% | -0.16 | -13.2% | -32.9% | 40 | 0.577 |

Before costs, the 28-day rule made −3.7% (Sharpe +0.05) against buy-and-hold's −21.9%: it stepped
aside for part of the fall, but 72 switches at 0.41% each cost about 30%. The 14-day rule looks better
(gross Sharpe +0.71, net +0.39 at 0.16%), but that was found by looking at this data, so it is a new
hypothesis to test on data not yet seen, not a pass.
