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
